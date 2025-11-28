#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDDA Framework - Conference Paper Results Generator
生成適合學術論文的詳細實驗結果
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# ============================================================================
# 格式化輸出函數
# ============================================================================

class PaperFormatter:
    """學術論文格式化輸出"""
    
    @staticmethod
    def section(title: str, level: int = 1):
        """章節標題"""
        if level == 1:
            print("\n" + "="*100)
            print(f"  {title}")
            print("="*100)
        elif level == 2:
            print("\n" + "-"*100)
            print(f"  {title}")
            print("-"*100)
        else:
            print(f"\n### {title}")
    
    @staticmethod
    def table_header(columns: List[str], widths: List[int]):
        """表格標題"""
        header = " | ".join([col.ljust(w) for col, w in zip(columns, widths)])
        print("\n" + header)
        print("-" * len(header))
    
    @staticmethod
    def table_row(values: List[str], widths: List[int]):
        """表格行"""
        row = " | ".join([str(val).ljust(w) for val, w in zip(values, widths)])
        print(row)
    
    @staticmethod
    def metric(name: str, value: Any, unit: str = ""):
        """指標輸出"""
        print(f"  {name:40s}: {value} {unit}")
    
    @staticmethod
    def latex_table_start(caption: str, label: str):
        """LaTeX 表格開始"""
        print("\n% LaTeX Table")
        print("\\begin{table}[htbp]")
        print("\\centering")
        print(f"\\caption{{{caption}}}")
        print(f"\\label{{{label}}}")
    
    @staticmethod
    def latex_table_end():
        """LaTeX 表格結束"""
        print("\\end{table}")


fmt = PaperFormatter()


# ============================================================================
# 輔助函數
# ============================================================================

def safe_get_feature_attr(feature, attr_name, default=None):
    """
    安全地從 Feature 對象或字典中獲取屬性
    
    Args:
        feature: Feature 對象或字典
        attr_name: 屬性名稱
        default: 預設值
    
    Returns:
        屬性值或預設值
    """
    if isinstance(feature, dict):
        return feature.get(attr_name, default)
    else:
        return getattr(feature, attr_name, default)


# ============================================================================
# 主要測試函數
# ============================================================================

def generate_paper_results():
    """生成論文結果"""
    
    fmt.section("CDDA Framework - Experimental Results", level=1)
    print(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System: Cognitive Discrepancy-Driven Agent (CDDA)")
    print(f"Architecture: Dual-LLM A2A Pattern")
    print(f"  - Agent A (Orchestrator): Phi-4-mini")
    print(f"  - Agent B (Consultant): Llama3.1-Aloe-Beta-8B")
    
    # 導入模組
    fmt.section("1. System Initialization", level=2)
    
    print("\n[1.1] Importing CDDA Framework Components...")
    try:
        from app.agents.cdda_agent import CDDAAgent
        from app.core.models.context_models import DiagnosticReport, Feature
        import glob
        print("[OK] All modules imported successfully")
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    
    # 掃描數據集
    print("\n[1.2] Dataset Validation...")
    subject_labels = {}
    data_folders = glob.glob("data/MRI_processed/*/sub-*")
    
    for folder_path in data_folders:
        parts = folder_path.replace('\\', '/').split('/')
        if len(parts) >= 3:
            subject_id = parts[-1]
            label = parts[-2]
            nii_files = list(Path(folder_path).glob("*.nii.gz"))
            if len(nii_files) >= 3:
                subject_labels[subject_id] = label
    
    subject_list = sorted(subject_labels.keys())
    
    # 統計數據集
    label_counts = {}
    for label in subject_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    fmt.metric("Total Valid Subjects", len(subject_list))
    for label, count in sorted(label_counts.items()):
        fmt.metric(f"  - {label} subjects", count)
    
    # 選擇測試受試者
    selected_subject = subject_list[0]
    ground_truth = subject_labels[selected_subject]
    
    fmt.metric("Selected Test Subject", selected_subject)
    fmt.metric("Ground Truth Label", ground_truth)
    
    # 初始化 CDDA Agent
    fmt.section("2. CDDA Agent Configuration", level=2)
    
    config_params = {
        "Orchestrator Model": "Phi-4-mini",
        "Orchestrator Path": "D:/hf_models/Phi-4-mini-instruct",
        "Consultant Model": "Llama3.1-Aloe-Beta-8B",
        "Consultant Path": "D:/hf_models/Llama3.1-Aloe-Beta-8B",
        "Quantization": "4-bit (NF4)",
        "LLM Mode": "Rule-based (for reproducibility)",
        "UQ Threshold": "0.8",
        "Z-score Threshold": "±2.5"
    }
    
    print("\nConfiguration Parameters:")
    for key, value in config_params.items():
        fmt.metric(key, value)
    
    print("\n[2.1] Initializing CDDA Agent...")
    init_start = time.time()
    
    try:
        agent = CDDAAgent(
            orchestrator_model="phi-4-mini",
            orchestrator_model_path="D:/hf_models/Phi-4-mini-instruct",
            consultant_model="llama3.1-aloe-beta-8b",
            consultant_model_path=r"D:\hf_models\Llama3.1-Aloe-Beta-8B",
            use_llm=True,  # Rule-based for reproducibility
            use_4bit=True,
            verbose=False  # Reduce noise for paper results
        )
        init_time = time.time() - init_start
        print(f"[OK] Agent initialized successfully ({init_time:.2f}s)")
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        return False
    
    # 執行分析
    fmt.section("3. Analysis Execution", level=2)
    
    print(f"\n[3.1] Running CDDA Analysis for {selected_subject}...")
    print("Pipeline Stages:")
    print("  1. Agent A: Orchestration (MCP resource reading, tool invocation)")
    print("  2. Agent B: Clinical synthesis (report generation)")
    print("  3. Post-processing: Executive summary generation")
    print("\n" + "="*100)
    print("DETAILED ANALYSIS LOG (Real-time Output)")
    print("="*100 + "\n")
    
    analysis_start = time.time()
    
    try:
        # 啟用詳細輸出
        agent.verbose = True
        if hasattr(agent, 'agent_a'):
            agent.agent_a.config.verbose = True
        if hasattr(agent, 'agent_b'):
            agent.agent_b.config.verbose = True
        
        result = agent.run_analysis(selected_subject)
        analysis_time = time.time() - analysis_start
        
        print("\n" + "="*100)
        print(f"[OK] Analysis completed successfully (Total time: {analysis_time:.2f}s)")
        print("="*100)
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 詳細結果分析
    fmt.section("4. Experimental Results", level=1)
    
    # 4.1 診斷性能
    fmt.section("4.1 Diagnostic Performance", level=2)
    
    diagnosis_map = {
        'AD': 'Alzheimer\'s Disease',
        'MCI': 'Mild Cognitive Impairment',
        'NC': 'Normal Cognition'
    }
    
    print("\nTable 1: Diagnostic Results")
    fmt.table_header(
        ["Metric", "Value", "Interpretation"],
        [30, 20, 40]
    )
    
    fmt.table_row(
        ["Ground Truth", diagnosis_map.get(ground_truth, ground_truth), "Clinical diagnosis"],
        [30, 20, 40]
    )
    fmt.table_row(
        ["AI Prediction", diagnosis_map.get(result.prediction, result.prediction), 
         "[OK] Correct" if ground_truth == result.prediction else "[X] Incorrect"],
        [30, 20, 40]
    )
    fmt.table_row(
        ["Confidence Score", f"{result.confidence:.4f}", 
         "High" if result.confidence > 0.8 else "Medium" if result.confidence > 0.6 else "Low"],
        [30, 20, 40]
    )
    fmt.table_row(
        ["Uncertainty (UQ) Score", f"{result.uq_score:.4f}",
         "High" if result.uq_score > 0.8 else "Medium" if result.uq_score > 0.5 else "Low"],
        [30, 20, 40]
    )
    fmt.table_row(
        ["Agent Decision Mode", result.agent_decision, 
         "Adaptive decision-making"],
        [30, 20, 40]
    )
    


    
    # 4.2 特徵重要性分析
    fmt.section("4.2 Feature Importance Analysis (SHAP + Z-score)", level=2)
    
    if result.context_object and result.context_object.diagnostic_report:
        top_features = result.context_object.diagnostic_report.top_features[:10]
        
        print("\nTable 2: Top 10 Diagnostic Drivers")
        fmt.table_header(
            ["Rank", "Brain Region (ROI)", "SHAP Value", "Z-score", "Clinical Significance"],
            [6, 35, 12, 10, 35]
        )
        
        for feat in top_features:
            # 安全獲取屬性
            rank = safe_get_feature_attr(feat, 'rank', 0)
            roi_name = safe_get_feature_attr(feat, 'roi_name', 'Unknown')
            shap_value = safe_get_feature_attr(feat, 'shap_value', 0)
            z_score = safe_get_feature_attr(feat, 'z_score', 0)
            
            # 臨床意義
            if abs(z_score) > 2.5:
                significance = "Anomalous (|Z| > 2.5)"
            elif z_score < -1.5:
                significance = "Atrophy pattern"
            elif z_score > 1.5:
                significance = "Preserved volume"
            else:
                significance = "Normal range"
            
            fmt.table_row(
                [str(rank), roi_name, f"{shap_value:+.4f}", f"{z_score:+.2f}", significance],
                [6, 35, 12, 10, 35]
            )
        

        
        # 統計分析
        print("\nStatistical Summary of Features:")
        shap_values = [safe_get_feature_attr(f, 'shap_value', 0) for f in top_features]
        z_scores = [safe_get_feature_attr(f, 'z_score', 0) for f in top_features]
        
        fmt.metric("Mean SHAP Value", f"{np.mean(shap_values):.4f}")
        fmt.metric("Std SHAP Value", f"{np.std(shap_values):.4f}")
        fmt.metric("Mean Z-score", f"{np.mean(z_scores):.4f}")
        fmt.metric("Std Z-score", f"{np.std(z_scores):.4f}")
        fmt.metric("Anomalous Features (|Z| > 2.5)", sum(1 for z in z_scores if abs(z) > 2.5))
    
    # 4.3 Agent 決策分析
    fmt.section("4.3 Agent Decision-Making Analysis", level=2)
    
    print("\nAgent A (Orchestrator) Actions:")
    if result.context_object:
        fmt.metric("MCP Actions Executed", len(result.context_object.mcp_actions))
        fmt.metric("Decision Rationale", result.context_object.decision_rationale)
        
        # 顯示 Agent A 的推理步驟
        if result.context_object.agent_a_reasoning:
            print("\nAgent A Reasoning Steps:")
            for i, step in enumerate(result.context_object.agent_a_reasoning[:10], 1):
                print(f"  {i}. {step}")
        
        # MCP 動作詳情
        if result.context_object.mcp_actions:
            print("\nMCP Action Log:")
            fmt.table_header(
                ["#", "Action Type", "Target", "Status", "Timestamp"],
                [4, 20, 40, 10, 20]
            )
            
            for i, action in enumerate(result.context_object.mcp_actions, 1):
                action_dict = action.to_dict() if hasattr(action, 'to_dict') else action
                fmt.table_row(
                    [str(i), 
                     action_dict.get('type', 'N/A'),
                     action_dict.get('target', 'N/A')[:40],
                     action_dict.get('status', 'N/A'),
                     action_dict.get('timestamp', 'N/A')[:20]],
                    [4, 20, 40, 10, 20]
                )
    
    # 工具調用結果
    if result.context_object and result.context_object.tool_results:
        tool_results = result.context_object.tool_results
        
        if 'counterfactual' in tool_results:
            print("\n" + "="*100)
            print("[COUNTERFACTUAL SIMULATION RESULTS]")
            print("="*100)
            cf = tool_results['counterfactual']
            
            print("\nTable 3: Counterfactual Analysis")
            fmt.table_header(
                ["Metric", "Original", "Counterfactual", "Delta"],
                [25, 20, 20, 15]
            )
            
            fmt.table_row(
                ["Prediction", 
                 cf.get('original_prediction', 'N/A'),
                 cf.get('new_prediction', 'N/A'),
                 "N/A"],
                [25, 20, 20, 15]
            )
            
            fmt.table_row(
                ["Confidence",
                 f"{cf.get('original_confidence', 0):.4f}",
                 f"{cf.get('new_confidence', 0):.4f}",
                 f"{cf.get('confidence_delta', 0):+.4f}"],
                [25, 20, 20, 15]
            )
            
            print(f"\n[INTERPRETATION]")
            print(f"  {cf.get('interpretation', 'N/A')}")
            
            # 計算影響程度
            impact = abs(cf.get('confidence_delta', 0))
            if impact > 0.1:
                impact_level = "High Impact (>10%)"
            elif impact > 0.05:
                impact_level = "Medium Impact (5-10%)"
            else:
                impact_level = "Low Impact (<5%)"
            
            print(f"\n[IMPACT ASSESSMENT]: {impact_level}")
            
            # 顯示被遮蔽的特徵
            if 'masked_features' in cf:
                print(f"\n[MASKED FEATURES]:")
                for i, feat in enumerate(cf.get('masked_features', [])[:5], 1):
                    feat_name = safe_get_feature_attr(feat, 'roi_name', 'Unknown')
                    print(f"  {i}. {feat_name}")
            
            print("="*100)
        
        if 'knowledge_context' in tool_results:
            print("\n" + "="*100)
            print("[KNOWLEDGE GRAPH QUERY RESULTS]")
            print("="*100)
            kc = tool_results['knowledge_context']
            anomalous_regions = kc.get('query_regions', [])
            
            fmt.metric("Anomalous Regions Detected", len(anomalous_regions))
            
            if anomalous_regions:
                print(f"\n[ANOMALOUS REGIONS]:")
                for i, region in enumerate(anomalous_regions, 1):
                    print(f"  {i}. {region}")
                
                print(f"\n[CLINICAL CONTEXT SUMMARY]:")
                print(f"  {kc.get('summary', 'N/A')}")
                
                # 顯示詳細的區域上下文
                if 'contexts' in kc:
                    print(f"\n[DETAILED REGION CONTEXTS]:")
                    for i, ctx in enumerate(kc.get('contexts', [])[:3], 1):
                        print(f"\n  Region {i}: {ctx.get('region', 'Unknown')}")
                        if 'context' in ctx:
                            context_info = ctx['context']
                            print(f"    - Full Name: {context_info.get('full_name', 'N/A')}")
                            print(f"    - Function: {context_info.get('function', 'N/A')}")
                            print(f"    - Clinical Significance: {context_info.get('clinical_significance', 'N/A')}")
                            if context_info.get('related_conditions'):
                                print(f"    - Related Conditions: {', '.join(context_info['related_conditions'][:3])}")
            
            print("="*100)

    
    # 4.4 執行摘要（Post-processing）
    fmt.section("4.4 Executive Summary (Post-Processing)", level=2)
    
    if 'executive_summary' in result.metadata:
        summary = result.metadata['executive_summary']
        
        print("\n" + "="*100)
        print("EXECUTIVE SUMMARY (Post-Processing by Agent A)")
        print("="*100)
        
        print(f"\n[HEADLINE]")
        print(f"  {summary.get('headline', 'N/A')}")
        
        print(f"\n[KEY FINDINGS]")
        for i, finding in enumerate(summary.get('key_findings', []), 1):
            print(f"  {i}. {finding}")
        
        print(f"\n[RECOMMENDED ACTIONS]")
        for i, action in enumerate(summary.get('recommended_actions', []), 1):
            print(f"  {i}. {action}")
        
        print(f"\n[RISK LEVEL]: {summary.get('risk_level', 'N/A')}")
        
        print("\n" + "="*100)
        
        # 評估摘要質量
        print("\nSummary Quality Metrics:")
        fmt.metric("Findings Count", len(summary.get('key_findings', [])))
        fmt.metric("Actions Count", len(summary.get('recommended_actions', [])))
        fmt.metric("Risk Stratification", summary.get('risk_level', 'N/A'))
    else:
        print("\n[WARNING] Executive summary not available in metadata")
    
    # 4.5 臨床報告分析
    fmt.section("4.5 Clinical Report Analysis", level=2)
    
    if hasattr(result, 'clinical_report') and result.clinical_report:
        report = result.clinical_report
        
        print("\nReport Statistics:")
        fmt.metric("Total Characters", len(report))
        fmt.metric("Total Words", len(report.split()))
        fmt.metric("Total Lines", report.count('\n') + 1)
        
        # 關鍵詞分析
        keywords = {
            'Alzheimer': report.lower().count('alzheimer'),
            'MCI': report.lower().count('mci'),
            'Cognitive': report.lower().count('cognitive'),
            'Hippocampus': report.lower().count('hippocampus'),
            'Atrophy': report.lower().count('atrophy'),
            'Uncertainty': report.lower().count('uncertainty'),
            'Confidence': report.lower().count('confidence')
        }
        
        print("\nKeyword Frequency:")
        for keyword, count in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                fmt.metric(f"  '{keyword}'", count)
        
        print("\n" + "="*100)
        print("COMPLETE CLINICAL REPORT")
        print("="*100)
        print(report)
        print("="*100)
        print(f"[Report Statistics: {len(report)} characters, {len(report.split())} words]")
        print("="*100)
    
    # 4.6 推理鏈分析
    fmt.section("4.6 Reasoning Chain Analysis", level=2)
    
    if result.reasoning_chain:
        total_steps = len(result.reasoning_chain)
        print("\nReasoning Chain Statistics:")
        fmt.metric("Total Log Entries", total_steps)
        
        # --- 定義分類邏輯 ---
        def classify_step(step):
            s = step.strip()
            if not s or s.startswith('=') or s.startswith('-'):
                return "Formatting"
            if '[Agent A]' in s or 'AGENT A' in s or 'Orchestrating' in s:
                return "Agent A"
            if '[Agent B]' in s or 'AGENT B' in s or 'Synthesizing' in s:
                return "Agent B"
            if 'MCP' in s or 'read_resource' in s or 'call_tool' in s:
                return "MCP"
            if 'HANDOFF' in s:
                return "Handoff"
            return "System/Logic" # 捕捉所有剩餘的推理描述 (如 Decision, Context validated)

        # --- 執行分類統計 ---
        counts = {
            "Agent A": 0,
            "Agent B": 0,
            "MCP": 0,
            "Handoff": 0,
            "System/Logic": 0,
            "Formatting": 0
        }
        
        for step in result.reasoning_chain:
            category = classify_step(step)
            counts[category] += 1
            
        # --- 顯示統計數據 ---
        # 這裡只顯示有意義的步驟 (排除格式化線條)
        effective_steps = total_steps - counts["Formatting"]
        
        fmt.metric("Effective Reasoning Steps", effective_steps, f"(Excluding {counts['Formatting']} formatting lines)")
        print("-" * 40)
        fmt.metric("Agent A (Orchestrator)", counts["Agent A"])
        fmt.metric("Agent B (Consultant)", counts["Agent B"])
        fmt.metric("MCP / Tools", counts["MCP"])
        fmt.metric("Handoff Events", counts["Handoff"])
        fmt.metric("System Logic / Rationale", counts["System/Logic"])
        
        # --- 顯示佔比表格 ---
        print("\nReasoning Chain Breakdown (Effective Steps):")
        fmt.table_header(
            ["Component", "Count", "Percentage"],
            [30, 10, 15]
        )
        
        # 計算百分比 (分母為有效步驟)
        if effective_steps > 0:
            for cat in ["Agent A", "Agent B", "MCP", "Handoff", "System/Logic"]:
                count = counts[cat]
                pct = (count / effective_steps) * 100
                fmt.table_row(
                    [cat, str(count), f"{pct:.1f}%"],
                    [30, 10, 15]
                )
        
        # --- 顯示關鍵推理步驟 (不變) ---
        print("\nKey Reasoning Steps (Sample):")
        key_steps = [s for s in result.reasoning_chain if any(keyword in s for keyword in 
                      ['Decision', 'Trigger', 'Simulation', 'Anomaly', 'Context', 'rationale'])]
        for i, step in enumerate(key_steps[:10], 1):
            # 清理掉多餘的時間戳記以便閱讀
            clean_step = step.split(']')[-1].strip() if ']' in step else step.strip()
            print(f"  {i}. {clean_step[:100]}{'...' if len(clean_step) > 100 else ''}")
        
        # --- 顯示完整推理鏈 (不變) ---
        print("\n" + "="*100)
        print("COMPLETE REASONING CHAIN (Full Transparency)")
        print("="*100)
        for i, step in enumerate(result.reasoning_chain, 1):
            if step.startswith("="*80) or step.startswith("="*100):
                print(f"\n{'='*100}")
                print(step.strip('=').strip())
                print("="*100)
            elif step.startswith("-"*80):
                print(f"\n{'-'*100}")
                print(step.strip('-').strip())
                print("-"*100)
            else:
                print(f"{i:4d}. {step}")
    # 4.7 性能指標
    fmt.section("4.7 Performance Metrics", level=2)
    
    total_time = init_time + analysis_time
    
    print("\nTable 4: System Performance")
    fmt.table_header(
        ["Component", "Time (s)", "Percentage", "Notes"],
        [30, 12, 12, 40]
    )
    
    fmt.table_row(
        ["Initialization", f"{init_time:.2f}", f"{init_time/total_time*100:.1f}%", "Model loading, toolkit setup"],
        [30, 12, 12, 40]
    )
    fmt.table_row(
        ["Analysis Pipeline", f"{analysis_time:.2f}", f"{analysis_time/total_time*100:.1f}%", "Agent A + Agent B + Post-processing"],
        [30, 12, 12, 40]
    )
    fmt.table_row(
        ["Total Time", f"{total_time:.2f}", "100.0%", "End-to-end execution"],
        [30, 12, 12, 40]
    )
    
    # 吞吐量計算
    throughput = 3600 / analysis_time  # subjects per hour
    fmt.metric("\nThroughput", f"{throughput:.2f} subjects/hour")
    
    # 記憶體使用（如果可用）
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        fmt.metric("Memory Usage (RSS)", f"{memory_info.rss / 1024**3:.2f} GB")
        fmt.metric("Memory Usage (VMS)", f"{memory_info.vms / 1024**3:.2f} GB")
    except:
        print("  (Memory metrics not available)")
    
    # 4.8 系統架構摘要
    fmt.section("4.8 System Architecture Summary", level=2)
    
    print("\nCDDA Framework Components:")
    print("  1. Layer 1: ML Prediction (Random Forest + SHAP)")
    print("  2. Layer 2: Uncertainty Quantification + Anomaly Detection")
    print("  3. Layer 3: Agent A (Orchestrator) - MCP Client")
    print("  4. Layer 3: Agent B (Consultant) - Clinical Synthesis")
    print("  5. Layer 4: Knowledge Graph (GraphRAG)")
    print("  6. Post-Processing: Executive Summary Generation")
    
    print("\nA2A Pattern Implementation:")
    print("  - Agent A: Reads resources, invokes tools, compiles context")
    print("  - Handoff: ContextObject transfer (no tool access for Agent B)")
    print("  - Agent B: Synthesizes clinical narrative from context")
    print("  - Post-processing: Agent A generates executive summary")
    
    print("\nKey Features:")
    print("  [+] Adaptive decision-making (3 pathways)")
    print("  [+] Counterfactual simulation for high uncertainty")
    print("  [+] Knowledge graph integration for anomalies")
    print("  [+] Complete reasoning chain transparency")
    print("  [+] Executive summary for rapid review")

    
    # 保存結果
    fmt.section("5. Results Export", level=1)
    
    print("\n[5.1] Generating Paper-Ready Output Files...")
    
    output_dir = Path("output/paper_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 5.1 完整文本報告
    report_file = output_dir / f"paper_results_{selected_subject}_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("CDDA Framework - Experimental Results for Conference Paper\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Subject ID: {selected_subject}\n")
        f.write(f"Ground Truth: {ground_truth}\n")
        f.write(f"AI Prediction: {result.prediction}\n")
        f.write(f"Accuracy: {'Correct' if ground_truth == result.prediction else 'Incorrect'}\n\n")
        
        f.write("-"*100 + "\n")
        f.write("DIAGNOSTIC PERFORMANCE\n")
        f.write("-"*100 + "\n\n")
        f.write(f"Confidence Score: {result.confidence:.4f}\n")
        f.write(f"Uncertainty Score: {result.uq_score:.4f}\n")
        f.write(f"Decision Mode: {result.agent_decision}\n\n")
        
        f.write("-"*100 + "\n")
        f.write("TOP 10 DIAGNOSTIC DRIVERS\n")
        f.write("-"*100 + "\n\n")
        
        if result.context_object and result.context_object.diagnostic_report:
            for feat in result.context_object.diagnostic_report.top_features[:10]:
                rank = safe_get_feature_attr(feat, 'rank', 0)
                roi_name = safe_get_feature_attr(feat, 'roi_name', 'Unknown')
                shap_value = safe_get_feature_attr(feat, 'shap_value', 0)
                z_score = safe_get_feature_attr(feat, 'z_score', 0)
                f.write(f"{rank:2d}. {roi_name:35s} SHAP={shap_value:+.4f}  Z={z_score:+.2f}\n")
        
        f.write("\n" + "-"*100 + "\n")
        f.write("CLINICAL REPORT\n")
        f.write("-"*100 + "\n\n")
        f.write(result.clinical_report if hasattr(result, 'clinical_report') else "N/A")
        
        f.write("\n\n" + "-"*100 + "\n")
        f.write("COMPLETE REASONING CHAIN\n")
        f.write("-"*100 + "\n\n")
        for step in result.reasoning_chain:
            f.write(step + "\n")
    
    print(f"[OK] Text report saved: {report_file}")
    
    # 5.2 JSON 格式（用於進一步分析）
    json_file = output_dir / f"paper_results_{selected_subject}_{timestamp}.json"
    
    results_dict = {
        "metadata": {
            "experiment_date": datetime.now().isoformat(),
            "subject_id": selected_subject,
            "ground_truth": ground_truth,
            "system_version": "CDDA v1.0",
            "orchestrator": "Phi-4-mini",
            "consultant": "Llama3.1-Aloe-Beta-8B"
        },
        "diagnostic_performance": {
            "prediction": result.prediction,
            "confidence": float(result.confidence),
            "uncertainty": float(result.uq_score),
            "decision_mode": result.agent_decision,
            "accuracy": ground_truth == result.prediction
        },
        "feature_importance": [],
        "agent_actions": {
            "mcp_actions_count": len(result.context_object.mcp_actions) if result.context_object else 0,
            "decision_rationale": result.context_object.decision_rationale if result.context_object else ""
        },
        "performance_metrics": {
            "initialization_time": float(init_time),
            "analysis_time": float(analysis_time),
            "total_time": float(total_time),
            "throughput_per_hour": float(3600 / analysis_time)
        },
        "reasoning_chain_stats": {
            "total_steps": len(result.reasoning_chain),
            "agent_a_steps": len([s for s in result.reasoning_chain if 'AGENT A' in s]),
            "agent_b_steps": len([s for s in result.reasoning_chain if 'AGENT B' in s]),
            "mcp_steps": len([s for s in result.reasoning_chain if 'MCP' in s])
        }
    }
    
    # 添加特徵重要性
    if result.context_object and result.context_object.diagnostic_report:
        for feat in result.context_object.diagnostic_report.top_features[:10]:
            results_dict["feature_importance"].append({
                "rank": safe_get_feature_attr(feat, 'rank', 0),
                "roi_name": safe_get_feature_attr(feat, 'roi_name', 'Unknown'),
                "shap_value": float(safe_get_feature_attr(feat, 'shap_value', 0)),
                "z_score": float(safe_get_feature_attr(feat, 'z_score', 0))
            })
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] JSON data saved: {json_file}")
    
    # 5.3 LaTeX 表格文件
    latex_file = output_dir / f"paper_tables_{selected_subject}_{timestamp}.tex"
    
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write("% CDDA Framework - LaTeX Tables for Conference Paper\n")
        f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Table 1: Diagnostic Performance
        f.write("% Table 1: Diagnostic Performance\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Diagnostic Performance Metrics}\n")
        f.write("\\label{tab:diagnostic_performance}\n")
        f.write("\\begin{tabular}{lll}\n")
        f.write("\\hline\n")
        f.write("Metric & Value & Interpretation \\\\\n")
        f.write("\\hline\n")
        f.write(f"Ground Truth & {diagnosis_map.get(ground_truth, ground_truth)} & Clinical diagnosis \\\\\n")
        f.write(f"AI Prediction & {diagnosis_map.get(result.prediction, result.prediction)} & "
               f"{'Correct' if ground_truth == result.prediction else 'Incorrect'} \\\\\n")
        f.write(f"Confidence & {result.confidence:.4f} & "
               f"{'High' if result.confidence > 0.8 else 'Medium' if result.confidence > 0.6 else 'Low'} \\\\\n")
        f.write(f"Uncertainty & {result.uq_score:.4f} & "
               f"{'High' if result.uq_score > 0.8 else 'Medium' if result.uq_score > 0.5 else 'Low'} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Table 2: Feature Importance
        f.write("% Table 2: Top Diagnostic Drivers\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Top 10 Diagnostic Drivers (SHAP + Z-score Analysis)}\n")
        f.write("\\label{tab:feature_importance}\n")
        f.write("\\begin{tabular}{clrr}\n")
        f.write("\\hline\n")
        f.write("Rank & Brain Region & SHAP & Z-score \\\\\n")
        f.write("\\hline\n")
        
        if result.context_object and result.context_object.diagnostic_report:
            for feat in result.context_object.diagnostic_report.top_features[:10]:
                rank = safe_get_feature_attr(feat, 'rank', 0)
                roi = safe_get_feature_attr(feat, 'roi_name', 'Unknown').replace('_', '\\_')
                shap = safe_get_feature_attr(feat, 'shap_value', 0)
                z = safe_get_feature_attr(feat, 'z_score', 0)
                
                f.write(f"{rank} & {roi} & {shap:+.4f} & {z:+.2f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Table 3: Performance
        f.write("% Table 3: System Performance\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{System Performance Metrics}\n")
        f.write("\\label{tab:performance}\n")
        f.write("\\begin{tabular}{lrr}\n")
        f.write("\\hline\n")
        f.write("Component & Time (s) & Percentage \\\\\n")
        f.write("\\hline\n")
        f.write(f"Initialization & {init_time:.2f} & {init_time/total_time*100:.1f}\\% \\\\\n")
        f.write(f"Analysis Pipeline & {analysis_time:.2f} & {analysis_time/total_time*100:.1f}\\% \\\\\n")
        f.write(f"Total & {total_time:.2f} & 100.0\\% \\\\\n")
        f.write("\\hline\n")
        f.write(f"\\multicolumn{{3}}{{l}}{{Throughput: {3600/analysis_time:.2f} subjects/hour}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"[OK] LaTeX tables saved: {latex_file}")
    
    # 總結
    fmt.section("6. Summary", level=1)
    
    print("\nExperimental Results Summary:")
    print(f"  Subject: {selected_subject}")
    print(f"  Ground Truth: {ground_truth}")
    print(f"  Prediction: {result.prediction} ({'[OK] Correct' if ground_truth == result.prediction else '[X] Incorrect'})")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Uncertainty: {result.uq_score:.4f}")
    print(f"  Analysis Time: {analysis_time:.2f}s")
    print(f"  Decision Mode: {result.agent_decision}")
    
    print("\nOutput Files Generated:")
    print(f"  1. Text Report: {report_file.name}")
    print(f"  2. JSON Data: {json_file.name}")
    print(f"  3. LaTeX Tables: {latex_file.name}")
    
    print("\n[SUCCESS] These files are ready for inclusion in your conference paper!")
    
    return True


# ============================================================================
# 執行
# ============================================================================

if __name__ == "__main__":
    try:
        success = generate_paper_results()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[Error] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
