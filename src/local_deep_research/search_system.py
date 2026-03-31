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
        
        patient_profile = self.structured_task.get("patient_profile", "")
        pathway = self.structured_task.get("primary_pathway", "未提取到主路径")
        details = "\n".join([f"- {i}" for i in self.structured_task.get("pathway_details", [])])
        alts = "\n".join([f"- {i}" for i in self.structured_task.get("alternatives_and_exclusions", [])])
        
        interventions = f"【主路径 (Primary Pathway)】: {pathway}\n【细节 (Details)】:\n{details}\n【被排除/备选方案 (Excluded/Alternatives)】:\n{alts}"
        
        prompt = f"""
        # Clinical Evidence Coordinator & API Planner
        
        You are an expert Clinical Decision Support Coordinator. 
        Your task is to analyze a Chinese clinical case, bridge the language gap, and generate precise English API queries for medical databases.
        
        ## 1. Clinical Context (Chinese Input)
        - **Patient Profile**: {patient_profile}
        - **Proposed Interventions (Baseline Plan)**: {interventions}
        - **Current Verification Knowledge**: {current_knowledge}
        
        ## 2. 🛑 API KEYWORD OPTIMIZATION STRATEGY (CRITICAL)
        - **Tier 1: Baseline Drug Optimization (Head-to-Head Comparisons)**
          Identify toxic legacy drugs in the baseline (e.g., Cisplatin). Use your internal medical knowledge to identify its modern, less toxic alternative (e.g., Carboplatin), and generate a pure keyword query putting them side-by-side.
          *Rule:* Do NOT use complex tags like [ptyp] or [pdat] here. Just use the core keywords so the engine can find landmark Phase III trials (like GOG 209).
          *GOOD PubMed Query:* "Endometrial cancer Cisplatin Carboplatin"
        - **Tier 2: Frontier Exploration (Recent 1-2 Years)**
          Search for new targeted/immunotherapies based on high-risk factors ONLY IF appropriate for the ADJUVANT setting.
          *Rule:* You MUST add "2024" or "2025" to filter for the latest breakthroughs.
          *GOOD PubMed Query:* "Endometrial cancer adjuvant Pembrolizumab 2024"
        
        Generate exactly {self.questions_per_iteration} highly targeted API queries.
        
        ## Output Format (JSON)
        {{
            "analysis": "Identify legacy drugs needing comparison, and new targets for exploration.",
            "sub_queries": [
                "Endometrial cancer Cisplatin Carboplatin", 
                "Endometrial cancer adjuvant Pembrolizumab 2024"
            ]
        }}
        """

        try:
            response = await invoke_with_timeout_and_retry(
                self.reasoning_model, prompt, timeout=1200.0, max_retries=3
            )
            
            response_text = remove_think_tags(response.content)
            
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                parsed = safe_json_from_text(response_text)
                if parsed:
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
        pool_objs = [
            {"idx": self.ref_pool.base_idx + i, "url": r.link, "desc": r.title} 
            for i, r in enumerate(self.ref_pool.pool, 1)
        ]
        pool_json = json.dumps(pool_objs, indent=2)

        prompt = textwrap.dedent(f"""
        你是一名严谨、克制且具备国际视野的肿瘤学多学科会诊（MDT）首席专家。你的任务是基于【基于既有指南的初版方案】和【2024年以后的最新检索证据】，撰写一份最终版的“深度循证”术后辅助治疗方案。

        ### 1. 患者真实画像与指南初版方案 (Original Guideline-based Plan)
        {self.treatment_context}
        
        ### 2. 最新前沿证据池 (Latest Frontier Evidence - 2024+)
        {current_knowledge}
        
        ### 3. 新增引用标签池 (New Citation Pool)
        ```json
        {pool_json}
        ```

        ## 🛑 证据层级与医疗绝对红线 (EVIDENCE HIERARCHY & RED LINES)
        1. **【指南与前沿文献的区别】**：初版方案中的文献代表既有的权威指南。新增引用池中的文献代表**最新前沿临床研究（Frontier Clinical Research）**，用来验证、优化或补充指南。
        2. **【🌟 强制循证引用规则（最核心！）】**：你在正文中提出的任何意见、治疗方案或细节，**必须**标出对应的【真实数字序号】（如使用初版的 `[1]`、`[3]` 或 新增池的 `[^^6]`）。**绝对禁止在输出中直接照抄英文字母 `[x]`！你必须根据上下文，将其替换为真实的阿拉伯数字序号！**
        3. **【优先整合 2024+ 前沿证据】**：如果发现了适用于该患者的最新权威文献（如新药获益、毒性对比），务必在正文中论述并打上 `[^^数字]`。
        4. **【允许指南兜底】**：如果没有检索到有价值的新证据，或新证据不适用于该患者，则仅保留初版的指南文献 `[数字]` 即可。
        5. **【路径排他原则】**：必须给出一条主次分明的综合推荐路径，绝不能将独立的疗法机械叠加。

        ## 📝 强制输出结构 (MANDATORY OUTPUT STRUCTURE - CRITICAL)
        你必须直接输出【最终版深度循证 MDT 治疗方案】正文！不要输出你的内部思考。严禁自己编造文献列表。

        ### ✅ 必须严格采用以下 5 个模块的 Markdown 结构：
        
        ### 1. **MDT 综合辅助决策主路径 (Synthesized MDT Pathway)**
        （必须首先用一行代码块或高亮粗体表达顶层决策！例如：`全身系统性治疗 (Systemic therapy) ± 盆腔外照射 (EBRT) ± 阴道近距离放疗 (VBT)`）

        ### 2. **各大主流指南 vs. 最新前沿研究 (Guidelines vs. Frontier Evidence)**
        （这是最重要的循证综述部分。请分两块论述：）
        - **既有指南共识**：（简述各大指南对该患者的主流共识及分歧点，必须替换为真实的初版文献序号）
          - **ESGO指南意见**：... [填入真实指南数字序号]
          - **FIGO指南意见**：... [填入真实指南数字序号]
          - **NCCN 指南意见**：... [填入真实指南数字序号]
          - **国内指南意见 (中华医学会妇科肿瘤学分会指南、中国肿瘤整合诊治指南)**：... [填入真实指南数字序号]
          - **其他指南意见**：... [填入真实指南数字序号]
        - **最新前沿研究补充**：（基于新增证据池，阐述近年2024+的新试验是否支持或优化了上述方案。**此处必须强制穿插使用 `[^^数字]` 标签**。如无相关新证据，写“当前检索未发现足以改变指南的最新前沿证据”。）

        ### 3. **主路径方案细化与执行细节**
        （在综合主路径框架下，详述药物选择、剂量、周期。**此处每一项具体推荐的结尾都必须包含真实的文献引用 `[数字]` 或 `[^^数字]`！绝对禁止出现带有字母 x 的 `[x]`！**）

        ### 4. **备选/暂不推荐方案讨论 (Alternatives & Exclusions)**
        （列出那些因条件不符被明确排除的疗法，明确指出不采纳原因）

        ### 5. **随访计划与注意事项**
        （整合最新的随访监测证据及不良反应管理，也需附上真实文献引用 `[数字]` 或 `[^^数字]`）
        """)

        try:
            response = await invoke_with_timeout_and_retry(
                self.report_model, prompt, timeout=1200.0, max_retries=3
            )
            content = remove_think_tags(response.content)
            
            # 调用重排函数：旧标号会被无损保留，新标号会被连续重排
            new_content, refs_section = self._reindex_references(content)
            
            full_report = new_content + refs_section
            return full_report, full_report 
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            fallback_refs = "\n==================================================\n"
            for i, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1):
                title = ref.title if ref.title else ref.link
                fallback_refs += f"[{i}] URL: {ref.link}\n    Title: {title}\n----------\n"
            final_report = f"## 报告生成失败 (Report Generation Failed)\n\n{current_knowledge}\n" + fallback_refs
            return final_report, final_report

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
