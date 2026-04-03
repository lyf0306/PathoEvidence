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
                local_llm = get_local_model(temperature=0.5)
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
        # 🤖 智能体 1：临床试验解析专家 (解开字数束缚，恢复长篇深度解析)
        # =====================================================================
        trial_prompt = textwrap.dedent(f"""
        你是一名顶尖的循证医学文献分析专家。请仔细阅读以下【最新查证的前沿循证数据】（均为英文 PubMed 摘要）：
        ---
        {current_knowledge}
        ---
        文献标签池：
        {pool_text}

        【🚨 核心任务与红线】：
        1. **同类项合并**：必须将同一个试验（如多篇 PORTEC-3 的随访分析）合并在一个大标题下（如 `#### PORTEC-3 试验综合解析`）。绝对禁止用 1.2.3. 给单篇文献编号！
        2. **打破字数与格式束缚**：不要只写简短的要点！请用长篇大论的专业医学段落，极其详尽地描述该试验的：① 入组人群特征；② 具体干预方案（放化疗剂量等）；③ 精准的生存数据（如 5/10年 OS、DFS 的具体百分比、HR值及P值）；④ 特定分子分型（如 p53abn、MMRd）的获益差异；⑤ 毒性评估。
        3. **全中文输出**：专业缩写（OS, DFS, HR 等）保留，其余全部翻译为严谨的中文。必须在句尾标上文献角标 [^^x]。
        """)

        logger.info("🤖 [Agent 1] 正在提炼并翻译临床试验数据 (恢复深度长文模式)...")
        try:
            trial_res = await invoke_with_timeout_and_retry(self.report_model, trial_prompt, timeout=800.0, max_retries=3)
            trial_analysis = remove_think_tags(trial_res.content).strip()
        except Exception:
            trial_analysis = "临床试验数据解析生成失败，请查阅原始文献。"

        # =====================================================================
        # 🤖 智能体 2：MDT 首席主笔 (统管全文，确保逻辑一致性与数据强制量化)
        # =====================================================================
        main_prompt = textwrap.dedent(f"""
        你是一名具备顶尖国际视野的妇科肿瘤 MDT 首席专家。
        【初步会诊草稿】：
        {self.treatment_context}
        【助手整理的临床试验深度解析】：
        {trial_analysis}

        你的任务是：输出最终版的 MDT 报告。
        
        ## 🛑 首席专家“元临床思维”法则 (核心红线)
        1. **【全局用药一致性与纠错】**：比对患者的合并症（高龄、高血压、糖尿病、脑梗等）。如果草稿使用了高毒性老药（如顺铂），你必须在最终方案中将其替换为低毒性现代方案（如卡铂+紫杉醇）。**🚨 极度重要：一旦你决定替换，整篇报告（包括“第一部分：指南分析”和“第二部分：主方案”）都必须统一口径！绝不能在前面说顺铂，后面又说卡铂，绝对不能前后矛盾！**
        2. **【病理审慎法则】**：若病理信息缺乏关键量化细节（如未写明转移灶大小是 ITC 还是宏转移），必须在【病情分析】末尾添加“💡 临床病理复核建议”。
        3. **【预后数据强制量化】**：在【预后分析】中，**绝对禁止说空话！**你必须从助手的《临床试验深度解析》中直接提取具体的生存数据（如“根据文献，5年 OS 为 XX%，DFS 为 XX%，HR=X.XX”），结合该患者的高危因素写出极其具体的预后评估！
        4. **【占位符机制】**：在“3. 核心临床试验循证解析”部分，你只需原封不动地输出 `{{{{TRIAL_PLACEHOLDER}}}}`，不要自己写任何分析内容！
        5. 绝对不要写参考文献！绝对不要输出 JSON！写完第四部分立刻停止！

        ## 📝 必须使用的固定输出模板（严格照抄格式，填充正文）：

        # 妇科肿瘤 MDT 最终版会诊报告

        ## 一、 病情分析
        ### 1. 病历及病理摘要
        （详尽总结患者病情、高危因素及 FIGO 分期）
        （💡 临床病理复核建议：...）

        ### 2. 各大核心指南推荐及风险分层
        - **ESGO 2025 风险分层**：...
        
        - **ESGO 指南推荐**：
        `（代码块写出方案，注意：若后续换药，此处的分析也要明确指出倾向于使用低毒性的现代替代方案）`
        **分析**：...
        
        - **NCCN / 其他权威指南推荐**（若证据中有）：
        `（代码块写出方案）`
        **分析**：...

        ### 3. 核心临床试验循证解析
        {{{{TRIAL_PLACEHOLDER}}}}

        ## 二、 术后处理
        ### 1. 肿瘤专科主方案
        （明确化疗药物、剂量、放疗靶区。若进行了顺铂换卡铂等操作，必须在此处写“💡 方案优化与纠错说明：阐述结合并发症做出的替换理由”）
        
        ### 2. 多学科及合并症管理
        （必须分点 1. 2. 3. 说明）

        ## 三、 预后分析
        （🚨 必须带有具体的 OS/DFS 百分比数据、HR值！结合患者高危因素深入讨论，标注文献）

        ## 四、 随访方案
        （必须分点说明！）
        - **随访频率**：...
        - **常规随访内容**：...
        - **影像学复查原则**：（必须写明：仅在出现临床症状或标志物异常时进行，反对无症状常规 CT/PET-CT）
        - **可能提示复发的警示症状**：...
        - **生活方式与心理调整**：...
        
        💡 请先在 <think> 标签内思考：我是否统一了前后的用药方案？我是否在预后分析里填入了具体的百分比数据？确认无误后再输出正文！
        """)

        logger.info("🤖 [Agent 2] 正在统筹生成 MDT 报告主干并确保全局逻辑一致性...")
        try:
            response = await invoke_with_timeout_and_retry(
                self.report_model, main_prompt, timeout=1200.0, max_retries=3
            )
            content = remove_think_tags(response.content)
            
            # 强制物理截断防暴走
            if "## 五" in content: content = content.split("## 五")[0].strip()
            if "# 五" in content: content = content.split("# 五")[0].strip()
            if "参考文献" in content: content = content.split("参考文献")[0].strip()
            if "References" in content: content = content.split("References")[0].strip()
            
            # =================================================================
            # 🔗 拼接占位符
            # =================================================================
            if "{{TRIAL_PLACEHOLDER}}" in content:
                content = content.replace("{{TRIAL_PLACEHOLDER}}", trial_analysis)
            else:
                if "## 二、 术后处理" in content:
                    content = content.replace("## 二、 术后处理", f"### 3. 核心临床试验循证解析\n{trial_analysis}\n\n## 二、 术后处理")
                else:
                    content += f"\n\n### 3. 核心临床试验循证解析\n{trial_analysis}"

            # 调用重排函数：旧标号无损保留，新标号连续重排，并在文末自动附上 Ref 列表
            new_content, refs_section = self._reindex_references(content)
            
            full_report = new_content + "\n" + refs_section
            return full_report, full_report 
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            fallback_refs = "\n==================================================\n"
            for i, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1):
                title = ref.title if ref.title else ref.link
                fallback_refs += f"[{i}] URL: {ref.link}\n    Title: {title}\n----------\n"
            return "报告生成失败", "报告生成失败"

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
