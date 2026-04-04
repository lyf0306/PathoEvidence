import asyncio
import json
import logging
import os
import re
import textwrap
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

from .config import (
    settings,
    get_claude_openai,
    get_deepseek_r1,
    get_deepseek_v3,
    get_gpt4_1,
    get_gpt4_1_mini,
    get_local_model,
)
from .connect_mcp import OrigeneMCPToolClient, mcp_servers
from .search_system_support import (
    compress_all_llm,
    extract_and_convert_list,
    parse_single,
    safe_json_from_text,
    SourcesReference,
)
from .tool_executor import ToolExecutor
from .tool_selector import ToolSelector
from .utilties.search_utilities import (
    invoke_with_timeout_and_retry,
    write_log_process_safe,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(
    log_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def remove_think_tags(text: str) -> str:
    """Robustly remove <think> tags from model output."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in cleaned:
        cleaned = re.sub(r".*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


class ReferencePool:
    """Reference pool for citations, supporting baseline offset."""
    def __init__(self, baseline_max_index: int = 0) -> None:
        self.pool: List[SourcesReference] = []
        self.link2idx: dict[str, int] = {}
        self.base_idx = baseline_max_index # 记录 graph-ec 传来的最大文献序号

    def add(self, title: str, citation: str, link: str) -> int:
        if not link:
            return -1
        if link in self.link2idx:
            return self.link2idx[link]
        # 新文献从 base_idx + 1 开始接力编号
        idx = self.base_idx + len(self.pool) + 1
        self.link2idx[link] = idx
        self.pool.append(
            SourcesReference(title=title or link, subtitle=citation or "", link=link)
        )
        return idx
    
    def get_ref_by_idx(self, idx: int):
        # 换算回本地 pool 的实际索引
        actual_idx = idx - self.base_idx - 1
        if 0 <= actual_idx < len(self.pool):
            return self.pool[actual_idx]
        return None


class AdvancedSearchSystem:
    def __init__(
        self,
        max_iterations=2,
        questions_per_iteration=4,
        is_report=True,
        chosen_tools: list[str] = None,
        error_log_path: str = "",
        using_model = "deepseek",  
        treatment_context: str = "",
        structured_task: dict = None, # 接收从 main.py 传来的结构化任务
    ):
        self.structured_task = structured_task or {}
        
        # 初始化带偏移量的引用池
        baseline_refs = self.structured_task.get("baseline_references", {})
        max_idx = baseline_refs.get("max_index", 0)
        self.ref_pool = ReferencePool(baseline_max_index=max_idx) 
        
        self.chosen_tools = chosen_tools
        self.is_report = is_report
        self.max_iterations = max_iterations
        self.questions_per_iteration = questions_per_iteration
        self.treatment_context = treatment_context
        self.knowledge_chunks = []
        self.all_links_of_system = []
        self.questions_by_iteration = {}

        if error_log_path == "":
            error_log_path = os.path.join(
                log_dir, f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
        self.error_log_path = error_log_path

        # === 模型初始化逻辑 ===
        self.using_model = using_model
        
        if self.using_model == "local":
            logger.info("🤖 Using Local vLLM Model (DeepSeek-R1-32B / qwen-test)")
            try:
                local_llm = get_local_model(temperature=0.1)
                local_fast_llm = get_local_model(temperature=0.1) 
                
                self.model = local_llm              
                self.reasoning_model = local_llm    
                self.tool_planning_model = local_llm 
                self.report_model = local_llm       
                self.fast_model = local_fast_llm    
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                raise e

        elif self.using_model == "deepseek":
            self.model = get_deepseek_r1()
            self.reasoning_model = get_deepseek_r1()
            self.tool_planning_model = get_deepseek_v3()
            self.fast_model = get_deepseek_v3()
            self.report_model = get_deepseek_r1()
            
        else:
            self.model = get_gpt4_1()
            self.reasoning_model = get_gpt4_1()
            self.tool_planning_model = get_gpt4_1()
            self.fast_model = get_gpt4_1_mini()
            self.report_model = get_gpt4_1()

    async def initialize(self):
        """Initialize tools in Pure API Mode (Lightweight)."""
        try:
            self.mcp_tool_client = OrigeneMCPToolClient(mcp_servers, self.chosen_tools)
            await self.mcp_tool_client.initialize()
            self.mcp_tool_dict = self.mcp_tool_client.tool2source
            
            self.tool_selector = ToolSelector(
                self.tool_planning_model,
                self.reasoning_model,
                self.mcp_tool_client,
                tool_info_data=None,
                embedding_api_key=None,
                embedding_cache=None,
                available_tools=self.chosen_tools,
            )
            
            self.tool_executor = ToolExecutor(
                self.mcp_tool_client, self.error_log_path, self.fast_model
            )
            
            logger.info("✅ System initialized in PURE API MODE (Official Databases Only).")
            
        except Exception as e:
            logger.error(f"Failed to initialize search system: {e}")
            raise e

    async def _get_follow_up_questions(self, current_knowledge: str, query: str) -> List[str]:
        now = datetime.now().strftime("%Y-%m-%d")
        upstream_report = self.treatment_context
        
        prompt = f"""
        你是一名顶级的“循证医学检索转化专家”（Clinical Evidence Coordinator）。
        上游的 RAG 系统生成了一份【初步 MDT 会诊报告】。

        你的核心任务是：
        1. 【无损提取】：精准理解报告中的“患者全息画像”、“主干治疗方案”及“PICO具体问题”。
        2. 【🚨 临床方案主动质疑（Clinical Challenge）】：主动审视上游推荐的化疗药物（如顺铂）是否属于毒性较大或已过时的传统药物。如果是，必须生成与现代更优替代药物（如卡铂）的“头对头对比”检索词！
        3. 【🚨 破解旧文献陷阱（通用时效性策略！）】：当上游要求查询某经典重磅试验（如 PORTEC、GOG 系列）的“最新数据”时，PubMed 通常会把多年前的初版报告排在第一。为了强制召回【任何年限】的最新随访结果，你绝对不能把年份写死（比如不能只写 10-year，因为有些试验可能是 7年或 5年随访），你**必须使用通用的随访关键词簇，并配合年份标签！**
           - ❌ 错误示范1：PORTEC-3 overall survival (太宽泛，会搜出2018年旧文)
           - ❌ 错误示范2：PORTEC-3 10-year overall survival (太死板，如果该试验只有5年随访数据就会漏检)
           - ✅ 正确示范：PORTEC-3 AND ("follow-up"[Title/Abstract] OR "long-term"[Title/Abstract] OR "updated"[Title/Abstract] OR "final"[Title/Abstract]) AND 2020:2026[dp]
        4. 【转化检索词】：将这些意图转化为能在 PubMed 纯英文数据库中进行精准匹配的高级 Boolean 检索词。

        【上游传入的初步 MDT 会诊报告】：
        {upstream_report}
        
        【当前已验证的知识】：
        {current_knowledge}

        【🚨 检索词设计红线】：
        1. 必须使用纯正的英文医学缩写，绝不能用自然语言长句问问题！
        2. 如果发现了需要质疑的老旧药物，必须生成对比检索词（如：Endometrial cancer Cisplatin Carboplatin efficacy toxicity）。
        3. 必须精确生成 {self.questions_per_iteration} 个英文检索词组。

        【强制输出 JSON 格式】：
        你必须输出合法的 JSON 格式。
        ```json
        {{
            "lossless_extraction": {{
                "patient_status": "极简提炼患者的核心分期、病理、合并症",
                "proposed_treatment": "无损提取上游推荐的主干方案",
                "pico_questions": "提取上游在文末留下的具体待查问题"
            }},
            "clinical_challenge": "记录药物替换质疑，或对抗旧文献的通用策略（如：我使用了 'follow-up' OR 'updated' 组合词，并加上了近期年份标签，以防止漏检）",
            "analysis": "结合全局方案、PICO问题和你的主动质疑，简述检索词构建策略。",
            "sub_queries": [
                "keyword query 1", 
                "keyword query 2",
                "keyword query 3"
            ]
        }}
        ```
        💡 请直接输出 JSON 代码块！
        """

        try:
            response = await invoke_with_timeout_and_retry(
                self.tool_planning_model, prompt, timeout=1200.0, max_retries=3
            )
            
            response_text = remove_think_tags(response.content)
            
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                parsed = safe_json_from_text(response_text[json_start:json_end])
                if parsed:
                    # 打印主动质疑的结果，方便后台监控
                    challenge = parsed.get("clinical_challenge", "")
                    logger.info(f"💡 [翻译官临床质询与高级语法控制]: {challenge}")
                    
                    questions = parsed.get("sub_queries", [])
                    return questions[:self.questions_per_iteration]
            
            return extract_and_convert_list(response_text)[:self.questions_per_iteration]
            
        except Exception as e:
            logger.warning(f"Failed to generate questions: {e}")
            return []
        
    async def _answer_query(
        self,
        current_knowledge: str,
        query: str,
        current_iteration: int,
        max_iterations: int,
    ) -> str:
        """Synthesize API findings."""
        existing_refs = [
            f"[{idx}] {ref.link} — {ref.title}"
            for idx, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1)
        ]
        refs_block = "\n".join(existing_refs) or "*None yet*"

        prompt = textwrap.dedent(f"""
        ## Task: Evidence Synthesis
        
        Validating clinical plan using **OFFICIAL API DATA** (PubMed/FDA/CT.gov).
        
        ## Verified API Data
        {current_knowledge}
        
        ## Sources
        {refs_block}
        
        ## Instructions
        1. **Validate**: Does 2024-Present evidence support the focus areas?
        2. **Update**: Identify newer trials/approvals.
        3. **Detail**: Note specific regimens (Drug/Dose) found in evidence.
        
        ## Output Template
        ## Evidence Status
        - **Decision Point**: [e.g. Adjuvant Therapy]
          - **Status**: [✅ Supported / ⚠️ Controversy / ❓ No Data]
          - **Key Evidence**: [Summarize findings from PubMed/CT.gov [^^n]]
        
        ## Missing Data
        [What couldn't be verified?]
        """)

        try:
            response = await invoke_with_timeout_and_retry(
                self.model, prompt, timeout=1200.0, max_retries=3
            )
            content = remove_think_tags(response.content)
            return content
        except Exception as e:
            logger.error(f"Failed to synthesize answer: {e}")
            return "Error synthesizing evidence."

    def _reindex_references(self, content: str) -> Tuple[str, str]:
        """
        重排引用序号，并生成与 graph-ec 完全一致的参考文献格式。
        核心逻辑：根据正文实际引用的文献，在 graph-ec 的最大序号之后【连续且不跳号】地重新编号。
        """
        # 1. 匹配括号内全都是数字、逗号、空格或^符号的字符串 (如 [10], [10, 12], [^^10])
        matches_iter = re.finditer(r"\[([\d\s\^\,]+)\]", content)
        
        all_cited_ids = []
        for m in matches_iter:
            inner_text = m.group(1)
            ids = [int(s) for s in re.findall(r"\d+", inner_text)]
            if ids:
                all_cited_ids.extend(ids)

        # 2. 去重，保留正文中实际引用到的文献顺序
        unique_cited_ids = list(dict.fromkeys(all_cited_ids))
        
        old_id_to_new_id = {}
        new_references_list = []
        
        # 动态分配连续的新序号，从 graph-ec 的最大序号之后开始接力
        current_new_id = self.ref_pool.base_idx + 1 
        
        for old_id in unique_cited_ids:
            ref_obj = self.ref_pool.get_ref_by_idx(old_id)
            if ref_obj:
                old_id_to_new_id[old_id] = current_new_id
                new_references_list.append((current_new_id, ref_obj))
                current_new_id += 1
        
        # 3. 替换正文中的旧序号为新分配的连续序号
        def replace_match(match):
            inner_text = match.group(1)
            old_ids = [int(s) for s in re.findall(r"\d+", inner_text)]
            if not old_ids:
                return match.group(0)
                
            new_ids = []
            for oid in old_ids:
                if oid in old_id_to_new_id:
                    new_ids.append(str(old_id_to_new_id[oid]))
                else:
                    new_ids.append(str(oid)) 
            
            if new_ids:
                return f"[{', '.join(new_ids)}]"
            return match.group(0)
            
        new_content = re.sub(r"\[([\d\s\^\,]+)\]", replace_match, content)
        
        # 4. 生成完全模仿 graph-ec 排版格式的文本
        refs_text = "\n==================================================\n" 
        
        if new_references_list:
            for new_idx, ref in new_references_list:
                title = ref.title.replace("\n", " ").strip() if ref.title else ref.link
                if len(title) > 300: 
                    title = title[:300] + "..."
                
                pmid_val = "Unknown"
                if "pubmed.ncbi.nlm.nih.gov/" in ref.link:
                    pmid_match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', ref.link)
                    if pmid_match:
                        pmid_val = pmid_match.group(1)
                
                # 严格对齐 graph-ec 的打印格式
                if pmid_val != 'Unknown':
                    refs_text += f"[{new_idx}] PMID: {pmid_val}\n"
                else:
                    refs_text += f"[{new_idx}] URL: {ref.link[:50]}...\n"
                    
                refs_text += f"    Title: {title}\n"
                refs_text += f"    Guidelines: 前沿证据合成 (Deep Research)\n"
                refs_text += "-" * 10 + "\n"
                
        return new_content, refs_text

    async def _generate_detailed_report(
        self, current_knowledge: str, findings: List[Dict], query: str, iteration: int
    ):
        # 1. 准备文献标签池
        pool_text = ""
        for i, r in enumerate(self.ref_pool.pool, 1):
            pool_text += f"[^^{self.ref_pool.base_idx + i}] {r.title}\n"

        # =====================================================================
        # 🤖 智能体 1：子宫内膜癌临床试验专家 (防无限死循环重试版)
        # =====================================================================
        trial_prompt = textwrap.dedent(f"""
        你是一名顶尖的妇科肿瘤（特长：子宫内膜癌）循证医学分析专家。请仔细阅读以下【最新查证的前沿循证数据】，并结合【当前患者真实病情】，为 MDT 会诊报告撰写《核心临床试验循证解析》部分。

        【当前患者真实病情草稿】（🚨 你的核心过滤标准！）：
        {self.treatment_context}
        
        【最新查证的前沿循证数据】（均为 PubMed 摘要）：
        ---
        {current_knowledge}
        ---
        文献标签池：
        {pool_text}

        【🚨 核心任务与严格红线】（一旦违背将导致严重医疗事故，请务必遵守！）：
        1. **【绝对贴合该患者分期与特征】**：仔细核对草稿中患者的【具体分期（如I期或III期）】、【组织学类型】和【高危因素】！你必须剔除与该患者风险级别完全不符的文献。只保留高度匹配当前患者真实病情的临床试验。
        2. **【杜绝学术废话】**：绝对禁止输出“研究空白”、“未来展望”等学术套话！
        3. **【严禁捏造与套用数据】**：如果文献中没有明确提到具体的 ORR、OS、PFS 等值，你只能描述结论，【严禁】自己捏造数据！
        4. **【🚨 绝对数量限额（最多3个）】**：在满足条件1的前提下，你**最多只能挑选 1 到 3 个（或套）最具决定性临床权重的核心试验**！绝对禁止列出第4个！
        5. **【🚨 格式防崩溃红线】**：为了防止生成死循环，【绝对禁止使用 "1.", "2.", "3." 这种数字序列来对试验编号】！请直接使用 `#### 试验名称/方案名称` 作为小标题！
        6. **【全中文与角标】**：专业缩写（OS, DFS等）保留，其余翻译为中文，并在句尾正确标上真实的文献角标 [^^x]。
        """)
        
        # =====================================================================
        # 🤖 智能体 1.5：子宫内膜癌随访专家 (动态提取合并症版)
        # =====================================================================
        followup_prompt = textwrap.dedent(f"""
        你是一名经验丰富的妇科肿瘤个案管理专家。请根据【患者初步会诊草稿】和【最新查证的前沿循证数据】，撰写一份极其专业的子宫内膜癌术后随访方案。
        
        【患者初步会诊草稿】：
        {self.treatment_context}

        【最新查证的前沿循证数据】：
        {current_knowledge}

        【输出要求】（直接输出正文，严禁输出任何Markdown大标题或总结废话）：
        1. **随访频率**：结合子宫内膜癌指南及前沿数据，明确不同时间段的具体时间间隔。
        2. **常规随访内容**：列出妇科专科查体（如阴道残端检查）及肿瘤标志物（如 CA125、HE4）。
        3. **影像学复查原则**：强调基于症状驱动的复查，反对无症状常规全身 PET-CT。
        4. **可能提示复发的警示症状**：列出阴道不规则流血、盆腹腔疼痛等子宫内膜癌常见复发表现。
        5. **生活方式与毒性管理**：⚠️【极端字数红线】必须仔细阅读草稿，提取该患者**真实的既往合并症**以及化疗可能的毒性。**必须高度凝练成一两句话连贯交待，绝对禁止分点展开或列清单！**

        【参考示例】（🚨 警告：仅为句式参考！绝不能照抄“糖尿病/高血压”等字眼，必须根据输入患者的真实合并症动态生成！）
        - **随访频率**：前2-3年每3-6个月随访一次；此后每6个月一次至第五年；之后每年一次。
        - **常规随访内容**：常规妇科查体及阴道残端细胞学检查，必要时检测 CA125、HE4。
        - **影像学复查原则**：仅在出现临床症状或肿瘤标志物异常升高时，行盆腹腔增强 CT 或 MRI 检查。
        - **可能提示复发的警示症状**：出现阴道不规则流血、持续性盆腹疼痛、下肢严重水肿等。
        - **生活方式与毒性管理**：针对患者既往的[真实合并症A]及[真实合并症B]史，建议严格控制代谢指标；随访中严密监测治疗方案可能的靶器官毒性，并建议心理科干预焦虑。
        """)

        # ---------------------------------------------------------------------
        # ⚡ 异步护栏封装：隔离重试，保障并发效率
        # ---------------------------------------------------------------------
        async def _run_agent1_with_guardrail():
            for attempt in range(3):
                try:
                    res = await invoke_with_timeout_and_retry(self.report_model, trial_prompt, timeout=800.0, max_retries=3)
                    content = remove_think_tags(res.content).strip()
                    
                    # 🛡️ 护栏检查：1. 是否超长(死循环特征)  2. 是否输出了超过3个试验
                    if len(content) > 3000 or content.count("####") > 4 or "#### 4" in content:
                        logger.warning(f"⚠️ 触发护栏：Agent 1 试验数量超标或陷入死循环 (尝试 {attempt+1}/3)，打回重做...")
                        if attempt < 2: 
                            continue
                    return content
                except Exception as e:
                    logger.error(f"Agent 1 error: {e}")
                    if attempt == 2: return "临床试验数据解析生成失败，请查阅原始文献。"

        async def _run_agent15():
            try:
                res = await invoke_with_timeout_and_retry(self.report_model, followup_prompt, timeout=800.0, max_retries=3)
                return remove_think_tags(res.content).strip()
            except Exception as e:
                logger.error(f"Agent 1.5 error: {e}")
                return "随访方案生成失败，请参考指南常规随访。"

        logger.info("🤖 [Agent 1 & 1.5] 正在并发执行：生成试验深度解析 & 定制随访规划...")
        trial_analysis, followup_plan = await asyncio.gather(_run_agent1_with_guardrail(), _run_agent15())

        # =====================================================================
        # 🤖 智能体 2：子宫内膜癌 MDT 首席主笔 (内置妇瘤专科药理学智慧)
        # =====================================================================
        main_prompt = textwrap.dedent(f"""
        你是一名具备顶尖国际视野的妇科肿瘤 MDT 首席专家。
        【初步会诊草稿】：
        {self.treatment_context}
        【助手整理的临床试验深度解析】：
        {trial_analysis}

        你的任务是：输出最终版的 MDT 报告主干。
        
        ## 🛑 首席专家“元临床思维”法则 (核心红线)
        1. **【现代临床用药纠偏（极度重要）】**：大模型的固有知识可能死板遵守了经典妇瘤试验原始的顺铂设计。但在现代真实临床实践中，基于“低毒高效”原则及患者的合并症，临床通常默认将高毒性老药替换为更安全的**卡铂（Carboplatin）**。因此，如果草稿推荐了含铂化疗，**请直接在主方案中写明优选卡铂！绝对不要先写顺铂再去纠错！**
        2. **【全局用药一致性】**：严密比对患者的合并症，整篇报告的主干方案、分析理由必须统一口径！绝不能前后矛盾！
        3. **【病理审慎法则】**：若草稿提供的病理信息缺乏关键量化细节，必须在【病情分析】末尾添加“💡 临床病理复核建议”。
        4. **【预后数据强制量化】**：在【预后分析】中，绝对禁止说空话！你必须从助手的《临床试验深度解析》中直接提取具体的生存数据（如OS/DFS百分比），结合该患者特有的高危因素写出极其具体的预后评估！
        5. **【双占位符机制】**：在“3. 核心临床试验循证解析”部分原封不动输出 `{{{{TRIAL_PLACEHOLDER}}}}`；在“四、 随访方案”部分原封不动输出 `{{{{FOLLOWUP_PLACEHOLDER}}}}`。绝对不要自己写这两个部分的内容！
        6. 绝对不要写参考文献！绝对不要输出 JSON！写完第四部分立刻停止！

        ## 📝 必须使用的固定输出模板（严格照抄 Markdown 结构，将尖括号 < > 内的说明替换为真实的专业分析，绝对禁止保留尖括号及其内部的说明文字！）：

        # 妇科肿瘤 MDT 最终版会诊报告

        ## 一、 病情分析
        ### 1. 病历及病理摘要
        <在此处详尽总结患者病情、高危因素及 FIGO 分期。并在末尾加上💡临床病理复核建议>

        ### 2. 各大核心指南推荐及风险分层
        - **核心指南风险分层**：<填写具体指南的风险分层>
        
        - **核心指南推荐方案 1**：
        `<在此处用代码块包裹原指南推荐的治疗方案。⚠️格式红线：代码块内必须提取并使用高度规格化的医学路径公式（如 "Systemic therapy ± EBRT ± VBT" 或 "Observation"），绝对禁止使用自然语言长句描述！>`
        **分析**：<在此处写出分析。如果有基于患者合并症的用药替换，必须在这里明确指出倾向于使用低毒性的现代替代方案及其理由>
        
        - **其他权威指南推荐方案**（若证据中有）：
        `<在此处用代码块包裹该指南推荐方案，同样必须使用规格化的医学路径公式格式>`
        **分析**：<在此处写出该指南的指导意见分析>

        ### 3. 核心临床试验循证解析
        {{{{TRIAL_PLACEHOLDER}}}}

        ## 二、 术后处理
        ### 1. 肿瘤专科主方案
        <明确写出最终决定的放化疗或观察方案。如果用到了药物平替或因低危决定免除化疗，必须写上“💡 方案优化与说明”并阐述理由>
        
        ### 2. 多学科及合并症管理
        <根据草稿中患者真实的合并症，分点列出各相关科室的随诊建议>

        ## 三、 预后分析
        <必须带有具体的 OS/DFS 百分比数据、HR值！结合患者特有的高危因素深入讨论，标注文献>

        ## 四、 随访方案
        {{{{FOLLOWUP_PLACEHOLDER}}}}
        
        💡 请先在 <think> 标签内思考：我是否统一了前后的用药方案？我是否去掉了所有的尖括号 < >？确认无误后再输出正文！
        """)

        logger.info("🤖 [Agent 2] 正在统筹生成 MDT 报告主干并确保全局逻辑一致性...")
        
        max_guardrail_retries = 3
        content = ""
        
        for attempt in range(max_guardrail_retries):
            try:
                response = await invoke_with_timeout_and_retry(
                    self.report_model, main_prompt, timeout=1200.0, max_retries=3
                )
                content = remove_think_tags(response.content)
                
                banned_phrases = [
                    "代码块写出", 
                    "若后续换药", 
                    "倾向于使用低毒性", 
                    "填写具体指南",
                    "严禁遗漏",
                    "高度规格化的医学路径公式",
                    "<", ">"  
                ]
                
                lazy_generation_detected = any(phrase in content for phrase in banned_phrases)
                
                if lazy_generation_detected:
                    logger.warning(f"⚠️ 触发护栏：Agent 2 照抄指令或遗留尖括号 (尝试 {attempt + 1}/{max_guardrail_retries})，打回重做...")
                    if attempt < max_guardrail_retries - 1: continue
                    else: logger.error("❌ 达到最大重试次数，模型依然照抄指令，强制继续流程。")
                
                if "## 五" in content: content = content.split("## 五")[0].strip()
                if "# 五" in content: content = content.split("# 五")[0].strip()
                if "参考文献" in content: content = content.split("参考文献")[0].strip()
                if "References" in content: content = content.split("References")[0].strip()
                
                break
                
            except Exception as e:
                logger.error(f"Agent 2 生成报错: {e}")
                if attempt == max_guardrail_retries - 1:
                    content = "报告生成失败"

        # =================================================================
        # 🔗 拼接双占位符 (Trial + Followup) 并重排参考文献
        # =================================================================
        try:
            if "{{TRIAL_PLACEHOLDER}}" in content:
                content = content.replace("{{TRIAL_PLACEHOLDER}}", trial_analysis)
            else:
                if "## 二、 术后处理" in content:
                    content = content.replace("## 二、 术后处理", f"### 3. 核心临床试验循证解析\n{trial_analysis}\n\n## 二、 术后处理")
                else:
                    content += f"\n\n### 3. 核心临床试验循证解析\n{trial_analysis}"
                    
            if "{{FOLLOWUP_PLACEHOLDER}}" in content:
                content = content.replace("{{FOLLOWUP_PLACEHOLDER}}", followup_plan)
            else:
                if "## 四、 随访方案" in content:
                    content = content.replace("## 四、 随访方案", f"## 四、 随访方案\n{followup_plan}\n")
                else:
                    content += f"\n\n## 四、 随访方案\n{followup_plan}"

            new_content, refs_section = self._reindex_references(content)
            
            full_report = new_content + "\n" + refs_section
            return full_report, full_report 
            
        except Exception as e:
            logger.error(f"Failed to post-process and reindex references: {e}")
            fallback_refs = "\n==================================================\n"
            for i, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1):
                title = ref.title if ref.title else ref.link
                fallback_refs += f"[{i}] URL: {ref.link}\n    Title: {title}\n----------\n"
            return "报告后处理失败", "报告后处理失败"
        
        
    async def _extract_knowledge(self, facts_md: str, refs_in_round: List[Dict]):
        """Extract key info from tool outputs."""
        prompt = f"""
        Extract key clinical facts from these API results.
        
        Facts:
        {facts_md}
        
        Refs:
        {json.dumps(refs_in_round)}
        
        Output JSON:
        {{
            "key_information": "- **Fact**... (<url>)",
            "cleaned_refs": [{{"url": "...", "description": "Title/Summary"}}]
        }}
        """
        try:
            resp = await invoke_with_timeout_and_retry(self.model, prompt, timeout=1200.0)
            cleaned_content = remove_think_tags(resp.content)
            data = safe_json_from_text(cleaned_content) or {}
            return data.get("key_information", ""), data.get("cleaned_refs", [])
        except Exception:
            return facts_md, refs_in_round

    async def process_multiple_knowledge_chunks(self, query: str, current_key_info: str) -> str:
        """Consolidate knowledge."""
        if not self.knowledge_chunks:
            return current_key_info
        
        prompt = f"""
        Consolidate these research findings into a concise summary.
        Keep citations <URL>.
        
        Findings:
        {current_key_info}
        """
        try:
            resp = await invoke_with_timeout_and_retry(self.fast_model, prompt, timeout=1200.0)
            return remove_think_tags(resp.content)
        except Exception:
            return current_key_info

    async def analyze_topic(self, query: str) -> Dict:
        """Main execution loop."""
        logger.info(f"Starting Pure API Validation")
        
        current_knowledge = ""
        iteration = 0
        findings = []

        await self.initialize()

        while iteration < self.max_iterations:
            questions = await self._get_follow_up_questions(current_knowledge, query)
            if not questions:
                questions = [query]
                
            self.questions_by_iteration[iteration] = questions
            
            fullquery_tool_results = []
            
            for question in questions:
                try:
                    tool_calls = await self.tool_selector.run(question)
                except Exception as e:
                    logger.warning(f"Tool selection failed: {e}")
                    tool_calls = []

                if not tool_calls:
                    continue
                
                # [代码修改点]：此处已彻底删除强制向 ClinicalTrials 注入 "AND AREA[LastUpdatePostDate]RANGE[01/01/2024, MAX]" 时间限制的逻辑，解放系统查阅经典方案的历史文献能力！

                try:
                    tool_results = await self.tool_executor.run(tool_calls) or []
                except Exception as e:
                    logger.error(f"Tool execution failed for question '{question}': {e}")
                    tool_results = []
                
                if tool_results:
                    try:
                        parsed_list = await asyncio.gather(
                            *(parse_single(res, query=question) for res in tool_results)
                        )
                        compressed = await compress_all_llm(
                            self.fast_model, parsed_list, limit=3, query=query
                        )
                        fullquery_tool_results.extend(compressed)
                    except Exception as e:
                        logger.warning(f"Error parsing tool results: {e}")

            iteration += 1
            facts, refs_raw = [], []
            for item in fullquery_tool_results:
                facts.extend(item.get("extracted_facts", []))
                refs_raw.extend(item.get("references", []))

            unique_refs = {}
            for ref in refs_raw:
                url = ref.get("url", "").strip()
                if url and url.startswith("http") and url not in unique_refs:
                    unique_refs[url] = ref
            refs = list(unique_refs.values())
            self.all_links_of_system.extend([r["url"] for r in refs])

            facts_md = "\n".join(f"- {f}" for f in facts)
            key_info, cleaned_refs = await self._extract_knowledge(facts_md, refs)
            
            self.knowledge_chunks.append({"key_info": key_info})
            current_knowledge = await self.process_multiple_knowledge_chunks(query, key_info)

            for ref in cleaned_refs:
                idx = self.ref_pool.add(
                    title=ref.get("description", ref["url"]),
                    citation="",
                    link=ref["url"]
                )
                current_knowledge = current_knowledge.replace(ref["url"], f"[^^{idx}]")

            final_answer = await self._answer_query(
                current_knowledge, query, iteration, self.max_iterations
            )
            current_knowledge = final_answer

        final_report = ""
        if self.is_report:
            try:
                final_report_tuple = await self._generate_detailed_report(
                    current_knowledge, findings, query, iteration
                )
                if isinstance(final_report_tuple, tuple):
                    final_report = final_report_tuple[1]
                else:
                    final_report = str(final_report_tuple)
            except Exception as e:
                logger.warning(f"Failed to generate detailed report: {e}")
                fallback_refs = "\n==================================================\n"
                for i, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1):
                    title = ref.title if ref.title else "Source"
                    fallback_refs += f"[{i}] URL: {ref.link}\n    Title: {title}\n----------\n"
                final_report = current_knowledge + fallback_refs

        return {
            "findings": findings,
            "iterations": iteration,
            "questions": self.questions_by_iteration,
            "current_knowledge": current_knowledge,
            "final_report": final_report,
        }
