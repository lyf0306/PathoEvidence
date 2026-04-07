import asyncio
import logging
import sys
import os
import json
import re

from .config import settings, get_local_model, get_gpt4_1_mini
from .search_system import AdvancedSearchSystem
from .utilties.search_utilities import invoke_with_timeout_and_retry

# 配置基础日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_local_model_health(timeout: float = 2.0) -> bool:
    """检测本地模型(vLLM)是否可用"""
    print("🔍 Checking local model connectivity...", end=" ", flush=True)
    try:
        llm = get_local_model(temperature=0.1)
        if hasattr(llm, "request_timeout"):
            llm.request_timeout = timeout
        llm.invoke("Hi") 
        print("✅ ONLINE")
        return True
    except Exception as e:
        print(f"❌ OFFLINE (Error: {str(e)[:50]}...)")
        return False

def read_context_from_file(file_path: str) -> str:
    """Helper: 从文件读取治疗方案内容"""
    try:
        file_path = file_path.strip('"').strip("'")
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return ""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading context file: {e}")
        return ""

def parse_graph_ec_report(raw_text: str):
    """
    分离 graph-ec 的正文和参考文献，并用正则安全提取最大文献序号
    """
    separator = "==================== 参考文献 (References) ===================="
    max_index = 0
    ref_text = ""
    report_body = raw_text
    
    if separator in raw_text:
        parts = raw_text.split(separator)
        report_body = parts[0].strip()
        ref_text = parts[1].strip()
        
        # 匹配 "[数字] PMID:" 或 "[数字] DocID:" 或 "[数字] URL:"
        matches = re.findall(r'\[(\d+)\]\s*(?:PMID|DocID|URL)', ref_text)
        if matches:
            max_index = max([int(m) for m in matches])
            
    return report_body, max_index, ref_text, separator

async def extract_structured_task(raw_text: str, fast_llm) -> dict:
    """
    无损结构化翻译官：仅负责将 Markdown 报告拆解为 JSON 字段，保留所有医学细节。
    """
    print(" [Parser] Converting clinical text to structured JSON format without losing details...")
    prompt = f"""
你是一个极其严格的、毫无感情的【医疗病历结构化信息提取器】。你的唯一任务是从长篇且无序的中文临床病历中，精准地“复印”和“归纳”事实，填入预设的 JSON 格式中。

=========================================
【👇需要你提取的患者原始病历数据👇】
{raw_text}
=========================================

🚨 【致命红线（绝对禁止违反）】：
1. **【禁止医学推理】**：你不是医生！严禁根据前期症状推断分期，严禁根据分期自行判定高低危，严禁擅自将原方案篡改为“观察”。原病历写了化疗几次，就是几次！
2. **【精准分期识别】**：必须一字不差地复制术后诊断中的罗马数字分期（如 IIIA1、IIIC、IVB），包含年份（如 FIGO 2023）。绝对禁止视线跳跃导致漏看（如把 IIIA1 错看成 IA1）。
3. **【寻址提取法则】**：
   - 找合并症：必须去【既往史】和【辅助检查】段落里找。
   - 找主方案：必须直接去病历文末寻找【术后处理】、【诊疗计划】、【MDT建议】段落，严格摘录原话。

【强制输出的 JSON 嵌套模板】（请严格按照此结构输出，不要遗漏任何层级，不要输出多余的非 JSON 文本）：
{{
  "patient_profile": {{
    "基本信息": "提取年龄、绝经状态、ECOG/KPS评分等。",
    "术后诊断_照抄原文": [
      "提取病历中【术后诊断】下的每一条内容，必须包含最完整的分期、病理类型、浸润深度、脉管癌栓等，形成数组形式原样照抄"
    ],
    "合并症与异常检查_归纳": {{
      "既往史与合并症": "高度概括高血压、糖尿病、冠心病、手术史等。",
      "重要辅助检查异常": "概括影像学或内镜发现的其他异常（如肺结节、胃炎、主动脉钙化、肝血管瘤等），这些对评估化疗毒性极其重要。"
    }}
  }},
  "primary_pathway": "严格摘录病历文末医生给出的【肿瘤专科治疗主方案】的原话（例如：TC方案化疗6次，序贯盆腔EBRT）。",
  "pathway_details": {{
    "肿瘤专科方案细节": [
      "放化疗的具体周期数、剂量",
      "治疗后的影像学复查计划（时间与项目）"
    ],
    "多学科及合并症管理": [
      "结合既往史，提取病历中提到的各科室随诊建议（如心内科、内分泌科随诊监测等）"
    ],
    "随访大纲": [
      "提取病历中提到的随访频率、随访检查项目"
    ]
  }},
  "alternatives_and_exclusions": [
    "提取病历中提及的分子分型追踪（如等结果回报后调整）、特定病毒随访（如HPV+复查TCT）、或明确被排除的方案。若无则填无。"
  ]
}}

💡 提取要求：仔细阅读上方的【患者原始病历数据】，深呼吸，确保没有任何重要信息被遗漏后，直接输出 JSON。
"""
    try:
        # 给翻译官也加上长超时时间，防止被强制中断
        response = await invoke_with_timeout_and_retry(fast_llm, prompt, timeout=1200.0)
        raw_resp = response.content
        
        # ================== 🔥 新增：打印原始输出供 Debug ==================
        print(f"\n{'='*20} 🔍 [Debug] 翻译官原始输出 {'='*20}")
        print(raw_resp)
        print(f"{'='*64}\n")
        # =================================================================
        
        # 1. 剔除可能的 <think> 标签及其内部的思维链
        cleaned_resp = re.sub(r"<think>.*?</think>", "", raw_resp, flags=re.DOTALL | re.IGNORECASE).strip()
        
        # 2. 🔥 终极无敌截取法：找到第一个 { 和最后一个 }
        start_idx = cleaned_resp.find('{')
        end_idx = cleaned_resp.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            cleaned_resp = cleaned_resp[start_idx:end_idx+1]
        
        # 解析 JSON (加上 strict=False 允许字符串内存在换行符)
        structured_data = json.loads(cleaned_resp, strict=False)
        print("✅ [Parser] Extracted patient profile and interventions successfully.")
        return structured_data
    except Exception as e:
        print(f"⚠️ [Parser] Parsing failed: {e}. Fallback to default structure.")
        return {
            "patient_profile": "Extraction failed, rely on raw text.",
            "proposed_interventions": ["Validate the proposed treatment plan"],
            "clinical_uncertainties": "Check latest evidence"
        }

async def run_evidence_update(treatment_context: str):
    """
    执行单一任务：临床证据更新
    """
    if check_local_model_health():
        print("🚀 Using Local vLLM Model (Free & Private).")
        current_mode = "local"
        fast_llm = get_local_model(temperature=0.1)
    else:
        print("⚠️ Local model unavailable. Switching to Cloud API (DeepSeek/GPT).")
        current_mode = "deepseek"
        try: 
            fast_llm = get_gpt4_1_mini()
        except: 
            fast_llm = get_local_model(temperature=0.1)

    # 1. 纯代码切分文献，提取最大序号（这一步专为最后的完美排版拼接做准备）
    report_body, max_index, baseline_refs, separator = parse_graph_ec_report(treatment_context)
    print(f"✅ [Parser] Found {max_index} baseline references from graph-ec.")

    # 2. LLM 提取无损结构化数据（传入全文 treatment_context，让大模型看到上下文！）
    structured_task = await extract_structured_task(treatment_context, fast_llm)
    
    # 动态将 max_index 注入 task 中，传递给底层 ReferencePool
    structured_task["baseline_references"] = {"max_index": max_index}
    
    print(f"\n🔄 Clinical Evidence Update System Activated.")
    print(f"   Context Length: {len(report_body)} characters")
    print("   Targeting Sources: PubMed (2024+), ClinicalTrials.gov\n")

    # 🔥 核心修改：严格限制加载的工具列表，防止 Token 爆炸导致超时！
    # 严格限制加载的工具列表，防止 Token 爆炸导致超时！
    my_target_tools = [
        # 1. 查最新医学论文 (NCBI/PubMed)
        "search_recent_pubmed",  
        
        # 2. 查最新临床试验 (ClinicalTrials.gov)
        "get_studies",           
        
        # 3. 查 FDA 权威数据：药物不良反应
        "get_adverse_reactions_by_drug_name", 
        
        # 4. 查 FDA 权威数据：药物警告与注意事项
        "get_warnings_and_cautions_by_drug_name" 
    ]

    # 3. 初始化搜索系统
    system = AdvancedSearchSystem(
        max_iterations=settings.detailed.iteration, 
        questions_per_iteration=settings.detailed.questions_per_iteration,
        is_report=True,
        treatment_context=report_body, 
        structured_task=structured_task,
        using_model=current_mode,
        chosen_tools=my_target_tools  # <- 传入工具白名单
    )

    try:
        await system.initialize()
        
        # 启动检索与合成的主流程
        query = "Validate and optimize this clinical treatment plan using the latest evidence."
        results = await system.analyze_topic(query)
        print(f"\n✅ Evidence synthesis task completed.")
        
        if results.get("final_report"):
             print("\n" + "="*60)
             print("   FINAL COMBINED CLINICAL REPORT   ")
             print("="*60 + "\n")
             
             final_resp_text = results['final_report']
             
             # 4. 完美拼接：解析 search_system 吐出的终极报告和新文献
             new_evidence_text = final_resp_text
             new_refs_text = ""
             split_marker = "=================================================="
             
             if split_marker in final_resp_text:
                 parts = final_resp_text.split(split_marker)
                 new_evidence_text = parts[0].strip()
                 new_refs_text = parts[1].strip()
             
             # 将重写后的终极正文、旧文献列表、新文献列表按序无缝缝合
             combined_report = f"### 🏥 循证校验与优化的最终治疗方案 (Deep EBM Synthesized Plan)\n\n" \
                               f"{new_evidence_text}\n\n" \
                               f"{separator}\n" \
                               f"{baseline_refs}\n"
             
             # 如果找到了新文献才往后贴
             if new_refs_text:
                 combined_report += f"{new_refs_text}\n"
             
             print(combined_report)
             
             # 保存到文件
             report_path = "evidence_update_report.md"
             with open(report_path, "w", encoding="utf-8") as f:
                 f.write(combined_report)
             print(f"\n📄 Report saved to: {os.path.abspath(report_path)}")

    except Exception as e:
        logger.error(f"Run failed: {e}")
        print(f"\n❌ Error during execution: {e}")

async def main():
    """主程序入口"""
    print("==================================================")
    print("   OriGene Clinical Evidence Validator (Auto-Hybrid)")
    print("==================================================")
    print("Strategy: Extract Context -> Deep Search -> Auto Gap-Closing Citations")
    print("Type 'quit' to exit at any time.")

    while True:
        print("\n--------------------------------------------------")
        print("Select Input Method:")
        print("1) Paste Treatment Plan Text (Markdown with References)")
        print("2) Load Plan from File (.txt/.md)")
        
        choice = input("\nEnter number (1 or 2): ").strip()
        
        if choice.lower() == 'quit':
            break

        treatment_context = ""

        if choice == "2":
            path = input("Enter file path: ").strip()
            if path.lower() == 'quit': break
            treatment_context = read_context_from_file(path)
        elif choice == "1":
            print("\n👇 Please paste the Clinical Treatment Plan (Markdown) below.")
            print("Type 'END' on a new line when finished:\n")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            treatment_context = "\n".join(lines)
        else:
            print("Invalid selection. Please enter 1 or 2.")
            continue
        
        if not treatment_context.strip():
            print("❌ Empty context provided. Please try again.")
            continue

        # 执行核心任务
        await run_evidence_update(treatment_context)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
