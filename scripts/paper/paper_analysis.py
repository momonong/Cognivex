#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDDA Paper Analysis Script

完整的系統測試腳本，用於論文撰寫。
輸出所有分析過程、中間結果和最終成果。

使用方法:
    python scripts/paper_analysis.py --subject sub-0005
    python scripts/paper_analysis.py --subject sub-0005 --output output/paper_results/
    python scripts/paper_analysis.py --subjects sub-0001 sub-0002 sub-0003
"""

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


class PaperAnalysisLogger:
    """論文分析日誌記錄器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 創建子目錄
        self.logs_dir = self.output_dir / "logs"
        self.reports_dir = self.output_dir / "reports"
        self.reasoning_dir = self.output_dir / "reasoning_chains"
        self.metrics_dir = self.output_dir / "metrics"
        self.visualizations_dir = self.output_dir / "visualizations"
        
        for dir_path in [self.logs_dir, self.reports_dir, 
                         self.reasoning_dir, self.metrics_dir,
                         self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化主日誌文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log_file = self.logs_dir / f"analysis_log_{timestamp}.txt"
        self.summary_file = self.output_dir / f"analysis_summary_{timestamp}.md"
        
        self.log_buffer = []
    
    def log(self, message: str, level: str = "INFO"):
        """記錄日誌消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(log_entry)
        self.log_buffer.append(log_entry)
        
        # 寫入文件
        with open(self.main_log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def section(self, title: str):
        """記錄章節標題"""
        separator = "=" * 80
        self.log(separator)
        self.log(title.center(80))
        self.log(separator)
    
    def subsection(self, title: str):
        """記錄子章節標題"""
        separator = "-" * 80
        self.log(separator)
        self.log(title)
        self.log(separator)


def print_system_info(logger: PaperAnalysisLogger):
    """打印系統信息"""
    logger.section("SYSTEM INFORMATION")
    
    import torch
    import platform
    
    logger.log(f"Python Version: {sys.version}")
    logger.log(f"Platform: {platform.platform()}")
    logger.log(f"PyTorch Version: {torch.__version__}")
    logger.log(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.log(f"CUDA Version: {torch.version.cuda}")
        logger.log(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    logger.log("")


def analyze_subject(
    agent: CDDAAgent,
    subject_id: str,
    logger: PaperAnalysisLogger,
    ground_truth: Optional[str] = None
) -> Dict:
    """
    分析單個受試者並記錄所有細節
    
    Args:
        agent: CDDA Agent 實例
        subject_id: 受試者 ID
        logger: 日誌記錄器
        ground_truth: 真實標籤 (可選)
    
    Returns:
        包含所有分析結果的字典
    """
    logger.section(f"ANALYZING SUBJECT: {subject_id}")
    
    if ground_truth:
        logger.log(f"Ground Truth: {ground_truth}")
    
    logger.log("")
    
    # ========================================================================
    # Phase 1: 初始化
    # ========================================================================
    logger.subsection("Phase 1: System Initialization")
    
    init_start = time.time()
    logger.log("Initializing CDDA Agent components...")
    logger.log(f"  - ToolKit: CNN-RF Model + SHAP + UQ + Anomaly Detection")
    logger.log(f"  - GraphRAG: Knowledge Graph Query Engine")
    logger.log(f"  - MCP Server: Resource & Tool Provider")
    logger.log(f"  - Agent A: Phi-4-mini (Orchestrator)")
    logger.log(f"  - Agent B: Llama3.1-Aloe-Beta-8B (Consultant)")
    init_time = time.time() - init_start
    logger.log(f"Initialization completed in {init_time:.2f}s")
    logger.log("")
    
    # ========================================================================
    # Phase 2: 運行分析
    # ========================================================================
    logger.subsection("Phase 2: Running CDDA Analysis")
    
    analysis_start = time.time()
    
    try:
        result = agent.run_analysis(subject_id)
        analysis_time = time.time() - analysis_start
        
        logger.log(f"Analysis completed successfully in {analysis_time:.2f}s")
        logger.log("")
        
    except Exception as e:
        logger.log(f"Analysis failed: {e}", level="ERROR")
        import traceback
        logger.log(traceback.format_exc(), level="ERROR")
        return None
    
    # ========================================================================
    # Phase 3: 提取並記錄結果
    # ========================================================================
    logger.subsection("Phase 3: Analysis Results")
    
    # 3.1 基本診斷結果
    logger.log("=== DIAGNOSTIC RESULTS ===")
    logger.log(f"Subject ID: {result.subject_id}")
    logger.log(f"Prediction: {result.prediction}")
    logger.log(f"Confidence: {result.confidence:.4f} ({result.confidence*100:.2f}%)")
    logger.log(f"Uncertainty Score (UQ): {result.uq_score:.4f}")
    logger.log(f"Agent Decision: {result.agent_decision}")
    
    if ground_truth:
        correct = (result.prediction == ground_truth)
        logger.log(f"Ground Truth: {ground_truth}")
        logger.log(f"Prediction Correct: {correct}")
    
    logger.log("")
    
    # 3.2 特徵重要性分析
    logger.log("=== TOP FEATURES (SHAP + Z-SCORE) ===")
    
    top_features = result.context_object.diagnostic_report.top_features[:10]
    
    logger.log(f"{'Rank':<6} {'ROI Name':<25} {'SHAP Value':<12} {'Z-Score':<10} {'Significance'}")
    logger.log("-" * 80)
    
    for feat in top_features:
        rank = feat.rank if hasattr(feat, 'rank') else 0
        roi_name = feat.roi_name if hasattr(feat, 'roi_name') else 'Unknown'
        shap_value = feat.shap_value if hasattr(feat, 'shap_value') else 0
        z_score = feat.z_score if hasattr(feat, 'z_score') else 0
        
        # 判斷臨床意義
        if abs(z_score) > 2.5:
            significance = "Anomalous"
        elif z_score < -1.5:
            significance = "Atrophy"
        elif z_score > 1.5:
            significance = "Preserved"
        else:
            significance = "Normal"
        
        logger.log(f"{rank:<6} {roi_name:<25} {shap_value:+.6f}   {z_score:+.4f}    {significance}")
    
    logger.log("")
    
    # 3.3 異常檢測結果
    anomaly_status = result.context_object.diagnostic_report.anomaly_status
    
    logger.log("=== ANOMALY DETECTION ===")
    logger.log(f"Anomaly Detected: {anomaly_status.has_anomaly}")
    
    if anomaly_status.has_anomaly:
        logger.log(f"Anomalous Regions ({len(anomaly_status.anomalous_regions)}):")
        for region in anomaly_status.anomalous_regions:
            logger.log(f"  - {region}")
    else:
        logger.log("No statistical anomalies detected.")
    
    logger.log("")
    
    # 3.4 工具調用結果
    tool_results = result.context_object.tool_results or {}
    
    if tool_results:
        logger.log("=== TOOL INVOCATION RESULTS ===")
        
        # 反事實模擬結果
        if 'counterfactual' in tool_results:
            cf = tool_results['counterfactual']
            logger.log("Counterfactual Simulation:")
            logger.log(f"  Original Prediction: {cf.get('original_prediction')} ({cf.get('original_confidence', 0):.4f})")
            logger.log(f"  New Prediction: {cf.get('new_prediction')} ({cf.get('new_confidence', 0):.4f})")
            logger.log(f"  Confidence Delta: {cf.get('confidence_delta', 0):+.4f}")
            logger.log(f"  Masked Features:")
            for feat in cf.get('masked_features', []):
                roi_name = feat.get('roi_name', 'Unknown')
                logger.log(f"    - {roi_name}")
            logger.log(f"  Interpretation: {cf.get('interpretation', 'N/A')}")
        
        # 知識圖譜查詢結果
        if 'knowledge_context' in tool_results:
            kc = tool_results['knowledge_context']
            logger.log("Knowledge Graph Query:")
            logger.log(f"  Query Regions: {', '.join(kc.get('query_regions', []))}")
            logger.log(f"  Summary: {kc.get('summary', 'N/A')}")
            
            contexts = kc.get('contexts', [])
            if contexts:
                logger.log(f"  Retrieved Contexts ({len(contexts)}):")
                for ctx in contexts[:3]:  # 只顯示前 3 個
                    region = ctx.get('region', 'Unknown')
                    context_info = ctx.get('context', {})
                    logger.log(f"    - {region}:")
                    logger.log(f"      Function: {context_info.get('function', 'N/A')}")
                    logger.log(f"      Clinical Significance: {context_info.get('clinical_significance', 'N/A')}")
        
        logger.log("")
    
    # 3.5 執行摘要
    if 'executive_summary' in result.metadata:
        summary = result.metadata['executive_summary']
        
        logger.log("=== EXECUTIVE SUMMARY ===")
        logger.log(f"Headline: {summary.get('headline', 'N/A')}")
        logger.log(f"Risk Level: {summary.get('risk_level', 'N/A')}")
        
        logger.log("Key Findings:")
        for finding in summary.get('key_findings', []):
            logger.log(f"  - {finding}")
        
        logger.log("Recommended Actions:")
        for action in summary.get('recommended_actions', []):
            logger.log(f"  - {action}")
        
        logger.log("")

    # ========================================================================
    # Phase 4: Agent 推理鏈分析
    # ========================================================================
    logger.subsection("Phase 4: Agent Reasoning Chain Analysis")
    
    logger.log("=== COMPLETE REASONING CHAIN ===")
    logger.log(f"Total Reasoning Steps: {len(result.reasoning_chain)}")
    logger.log("")
    
    # 分析推理鏈結構
    agent_a_steps = 0
    agent_b_steps = 0
    mcp_actions = 0
    
    for step in result.reasoning_chain:
        if '[Agent A]' in step:
            agent_a_steps += 1
        elif '[Agent B]' in step:
            agent_b_steps += 1
        elif 'read_resource' in step or 'call_tool' in step:
            mcp_actions += 1
    
    logger.log(f"Agent A Reasoning Steps: {agent_a_steps}")
    logger.log(f"Agent B Reasoning Steps: {agent_b_steps}")
    logger.log(f"MCP Actions: {mcp_actions}")
    logger.log("")
    
    # 打印完整推理鏈
    logger.log("Full Reasoning Chain:")
    for i, step in enumerate(result.reasoning_chain, 1):
        logger.log(f"{i:3d}. {step}")
    
    logger.log("")
    
    # ========================================================================
    # Phase 5: MCP 動作詳細分析
    # ========================================================================
    logger.subsection("Phase 5: MCP Actions Analysis")
    
    mcp_actions_list = result.context_object.mcp_actions
    
    logger.log(f"=== MCP ACTIONS ({len(mcp_actions_list)}) ===")
    
    for i, action in enumerate(mcp_actions_list, 1):
        action_dict = action.to_dict() if hasattr(action, 'to_dict') else action
        
        logger.log(f"Action {i}:")
        logger.log(f"  Type: {action_dict.get('type', 'unknown')}")
        logger.log(f"  Target: {action_dict.get('target', 'unknown')}")
        logger.log(f"  Status: {action_dict.get('status', 'unknown')}")
        logger.log(f"  Timestamp: {action_dict.get('timestamp', 'N/A')}")
        
        if action_dict.get('status') == 'error':
            error = action_dict.get('error', {})
            logger.log(f"  Error: {error.get('message', 'Unknown error')}")
        
        if action_dict.get('arguments'):
            logger.log(f"  Arguments: {json.dumps(action_dict['arguments'], indent=4)}")
        
        logger.log("")
    
    # ========================================================================
    # Phase 6: 臨床報告
    # ========================================================================
    logger.subsection("Phase 6: Clinical Report")
    
    logger.log("=== CLINICAL REPORT (AGENT B SYNTHESIS) ===")
    logger.log("")
    logger.log(result.clinical_report)
    logger.log("")
    
    # ========================================================================
    # Phase 7: 性能指標
    # ========================================================================
    logger.subsection("Phase 7: Performance Metrics")
    
    total_time = init_time + analysis_time
    throughput = 3600 / analysis_time if analysis_time > 0 else 0
    
    logger.log("=== PERFORMANCE METRICS ===")
    logger.log(f"Initialization Time: {init_time:.2f}s")
    logger.log(f"Analysis Time: {analysis_time:.2f}s")
    logger.log(f"Total Time: {total_time:.2f}s")
    logger.log(f"Throughput: {throughput:.2f} subjects/hour")
    logger.log("")
    
    # ========================================================================
    # Phase 8: 保存結果到文件
    # ========================================================================
    logger.subsection("Phase 8: Saving Results to Files")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 8.1 保存完整結果 (JSON)
    result_file = logger.output_dir / f"result_{subject_id}_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.log(f"Saved complete result to: {result_file}")
    
    # 8.2 保存推理鏈 (JSON)
    reasoning_file = logger.reasoning_dir / f"reasoning_{subject_id}_{timestamp}.json"
    reasoning_data = {
        'subject_id': subject_id,
        'timestamp': timestamp,
        'reasoning_chain': result.reasoning_chain,
        'agent_a_steps': agent_a_steps,
        'agent_b_steps': agent_b_steps,
        'mcp_actions': mcp_actions
    }
    with open(reasoning_file, 'w', encoding='utf-8') as f:
        json.dump(reasoning_data, f, indent=2, ensure_ascii=False)
    logger.log(f"Saved reasoning chain to: {reasoning_file}")
    
    # 8.3 保存推理鏈 (純文本)
    reasoning_txt_file = logger.reasoning_dir / f"reasoning_{subject_id}_{timestamp}.txt"
    with open(reasoning_txt_file, 'w', encoding='utf-8') as f:
        f.write(f"REASONING CHAIN FOR {subject_id}\n")
        f.write("=" * 80 + "\n\n")
        for i, step in enumerate(result.reasoning_chain, 1):
            f.write(f"{i:3d}. {step}\n")
    logger.log(f"Saved reasoning chain (text) to: {reasoning_txt_file}")
    
    # 8.4 保存臨床報告 (Markdown)
    report_file = logger.reports_dir / f"report_{subject_id}_{timestamp}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Clinical Report: {subject_id}\n\n")
        f.write(f"**Generated**: {timestamp}\n\n")
        f.write(f"**Prediction**: {result.prediction}\n\n")
        f.write(f"**Confidence**: {result.confidence:.4f}\n\n")
        f.write(f"**Uncertainty**: {result.uq_score:.4f}\n\n")
        f.write(f"**Agent Decision**: {result.agent_decision}\n\n")
        
        if ground_truth:
            f.write(f"**Ground Truth**: {ground_truth}\n\n")
        
        f.write("---\n\n")
        f.write("## Clinical Report\n\n")
        f.write(result.clinical_report)
        f.write("\n\n---\n\n")
        
        # 添加執行摘要
        if 'executive_summary' in result.metadata:
            summary = result.metadata['executive_summary']
            f.write("## Executive Summary\n\n")
            f.write(f"**Headline**: {summary.get('headline', 'N/A')}\n\n")
            f.write(f"**Risk Level**: {summary.get('risk_level', 'N/A')}\n\n")
            f.write("**Key Findings**:\n")
            for finding in summary.get('key_findings', []):
                f.write(f"- {finding}\n")
            f.write("\n**Recommended Actions**:\n")
            for action in summary.get('recommended_actions', []):
                f.write(f"- {action}\n")
    
    logger.log(f"Saved clinical report to: {report_file}")
    
    # 8.5 保存特徵重要性 (CSV)
    features_file = logger.metrics_dir / f"features_{subject_id}_{timestamp}.csv"
    with open(features_file, 'w', encoding='utf-8') as f:
        f.write("Rank,ROI_Name,SHAP_Value,Z_Score,Feature_Value,Significance\n")
        for feat in top_features:
            rank = feat.rank if hasattr(feat, 'rank') else 0
            roi_name = feat.roi_name if hasattr(feat, 'roi_name') else 'Unknown'
            shap_value = feat.shap_value if hasattr(feat, 'shap_value') else 0
            z_score = feat.z_score if hasattr(feat, 'z_score') else 0
            feature_value = feat.feature_value if hasattr(feat, 'feature_value') else 0
            
            if abs(z_score) > 2.5:
                significance = "Anomalous"
            elif z_score < -1.5:
                significance = "Atrophy"
            elif z_score > 1.5:
                significance = "Preserved"
            else:
                significance = "Normal"
            
            f.write(f"{rank},{roi_name},{shap_value},{z_score},{feature_value},{significance}\n")
    
    logger.log(f"Saved feature importance to: {features_file}")
    
    # 8.6 保存性能指標 (JSON)
    metrics_file = logger.metrics_dir / f"metrics_{subject_id}_{timestamp}.json"
    metrics_data = {
        'subject_id': subject_id,
        'timestamp': timestamp,
        'ground_truth': ground_truth,
        'prediction': result.prediction,
        'confidence': result.confidence,
        'uq_score': result.uq_score,
        'agent_decision': result.agent_decision,
        'correct': (result.prediction == ground_truth) if ground_truth else None,
        'performance': {
            'init_time': init_time,
            'analysis_time': analysis_time,
            'total_time': total_time,
            'throughput': throughput
        },
        'reasoning_stats': {
            'total_steps': len(result.reasoning_chain),
            'agent_a_steps': agent_a_steps,
            'agent_b_steps': agent_b_steps,
            'mcp_actions': mcp_actions
        },
        'top_features': [
            {
                'rank': feat.rank if hasattr(feat, 'rank') else 0,
                'roi_name': feat.roi_name if hasattr(feat, 'roi_name') else 'Unknown',
                'shap_value': feat.shap_value if hasattr(feat, 'shap_value') else 0,
                'z_score': feat.z_score if hasattr(feat, 'z_score') else 0
            }
            for feat in top_features[:5]
        ]
    }
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    logger.log(f"Saved performance metrics to: {metrics_file}")
    logger.log("")
    
    logger.log("=" * 80)
    logger.log(f"Analysis for {subject_id} completed successfully!")
    logger.log("=" * 80)
    logger.log("")
    
    return {
        'subject_id': subject_id,
        'result': result,
        'metrics': metrics_data,
        'files': {
            'result': str(result_file),
            'reasoning_json': str(reasoning_file),
            'reasoning_txt': str(reasoning_txt_file),
            'report': str(report_file),
            'features': str(features_file),
            'metrics': str(metrics_file)
        }
    }


def generate_summary_report(
    all_results: List[Dict],
    logger: PaperAnalysisLogger,
    ground_truths: Optional[Dict[str, str]] = None
):
    """
    生成所有受試者的總結報告
    
    Args:
        all_results: 所有分析結果列表
        logger: 日誌記錄器
        ground_truths: 真實標籤字典
    """
    logger.section("GENERATING SUMMARY REPORT")
    
    if not all_results:
        logger.log("No results to summarize.", level="WARNING")
        return
    
    # 統計數據
    total_subjects = len(all_results)
    
    predictions = {}
    confidences = []
    uq_scores = []
    agent_decisions = {}
    correct_predictions = 0
    total_with_gt = 0
    
    init_times = []
    analysis_times = []
    total_times = []
    
    for res in all_results:
        metrics = res['metrics']
        
        # 預測統計
        pred = metrics['prediction']
        predictions[pred] = predictions.get(pred, 0) + 1
        
        confidences.append(metrics['confidence'])
        uq_scores.append(metrics['uq_score'])
        
        # 決策統計
        decision = metrics['agent_decision']
        agent_decisions[decision] = agent_decisions.get(decision, 0) + 1
        
        # 準確率統計
        if metrics['correct'] is not None:
            total_with_gt += 1
            if metrics['correct']:
                correct_predictions += 1
        
        # 性能統計
        perf = metrics['performance']
        init_times.append(perf['init_time'])
        analysis_times.append(perf['analysis_time'])
        total_times.append(perf['total_time'])
    
    # 計算平均值
    avg_confidence = sum(confidences) / len(confidences)
    avg_uq = sum(uq_scores) / len(uq_scores)
    avg_init_time = sum(init_times) / len(init_times)
    avg_analysis_time = sum(analysis_times) / len(analysis_times)
    avg_total_time = sum(total_times) / len(total_times)
    
    accuracy = (correct_predictions / total_with_gt * 100) if total_with_gt > 0 else None
    
    # 打印總結
    logger.log("=== SUMMARY STATISTICS ===")
    logger.log(f"Total Subjects Analyzed: {total_subjects}")
    logger.log("")
    
    logger.log("Prediction Distribution:")
    for pred, count in sorted(predictions.items()):
        percentage = count / total_subjects * 100
        logger.log(f"  {pred}: {count} ({percentage:.1f}%)")
    logger.log("")
    
    logger.log("Agent Decision Distribution:")
    for decision, count in sorted(agent_decisions.items()):
        percentage = count / total_subjects * 100
        logger.log(f"  {decision}: {count} ({percentage:.1f}%)")
    logger.log("")
    
    logger.log("Diagnostic Metrics:")
    logger.log(f"  Average Confidence: {avg_confidence:.4f}")
    logger.log(f"  Average Uncertainty (UQ): {avg_uq:.4f}")
    
    if accuracy is not None:
        logger.log(f"  Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_with_gt})")
    logger.log("")
    
    logger.log("Performance Metrics:")
    logger.log(f"  Average Initialization Time: {avg_init_time:.2f}s")
    logger.log(f"  Average Analysis Time: {avg_analysis_time:.2f}s")
    logger.log(f"  Average Total Time: {avg_total_time:.2f}s")
    logger.log(f"  Average Throughput: {3600/avg_analysis_time:.2f} subjects/hour")
    logger.log("")
    
    # 保存總結報告到 Markdown
    summary_md = logger.summary_file
    
    with open(summary_md, 'w', encoding='utf-8') as f:
        f.write("# CDDA Paper Analysis Summary Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Subjects**: {total_subjects}\n\n")
        f.write("---\n\n")
        
        f.write("## 1. Prediction Distribution\n\n")
        f.write("| Prediction | Count | Percentage |\n")
        f.write("|------------|-------|------------|\n")
        for pred, count in sorted(predictions.items()):
            percentage = count / total_subjects * 100
            f.write(f"| {pred} | {count} | {percentage:.1f}% |\n")
        f.write("\n")
        
        f.write("## 2. Agent Decision Distribution\n\n")
        f.write("| Decision | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        for decision, count in sorted(agent_decisions.items()):
            percentage = count / total_subjects * 100
            f.write(f"| {decision} | {count} | {percentage:.1f}% |\n")
        f.write("\n")
        
        f.write("## 3. Diagnostic Metrics\n\n")
        f.write(f"- **Average Confidence**: {avg_confidence:.4f}\n")
        f.write(f"- **Average Uncertainty (UQ)**: {avg_uq:.4f}\n")
        if accuracy is not None:
            f.write(f"- **Accuracy**: {accuracy:.2f}% ({correct_predictions}/{total_with_gt})\n")
        f.write("\n")
        
        f.write("## 4. Performance Metrics\n\n")
        f.write(f"- **Average Initialization Time**: {avg_init_time:.2f}s\n")
        f.write(f"- **Average Analysis Time**: {avg_analysis_time:.2f}s\n")
        f.write(f"- **Average Total Time**: {avg_total_time:.2f}s\n")
        f.write(f"- **Average Throughput**: {3600/avg_analysis_time:.2f} subjects/hour\n")
        f.write("\n")
        
        f.write("## 5. Individual Results\n\n")
        f.write("| Subject ID | Prediction | Confidence | UQ Score | Decision | Ground Truth | Correct |\n")
        f.write("|------------|------------|------------|----------|----------|--------------|----------|\n")
        
        for res in all_results:
            metrics = res['metrics']
            subject_id = metrics['subject_id']
            pred = metrics['prediction']
            conf = metrics['confidence']
            uq = metrics['uq_score']
            decision = metrics['agent_decision']
            gt = metrics.get('ground_truth', 'N/A')
            correct = '✓' if metrics.get('correct') else ('✗' if metrics.get('correct') is not None else 'N/A')
            
            f.write(f"| {subject_id} | {pred} | {conf:.4f} | {uq:.4f} | {decision} | {gt} | {correct} |\n")
        
        f.write("\n")
        
        f.write("## 6. Files Generated\n\n")
        for res in all_results:
            subject_id = res['subject_id']
            files = res['files']
            
            f.write(f"### {subject_id}\n\n")
            f.write(f"- **Complete Result**: `{files['result']}`\n")
            f.write(f"- **Reasoning Chain (JSON)**: `{files['reasoning_json']}`\n")
            f.write(f"- **Reasoning Chain (Text)**: `{files['reasoning_txt']}`\n")
            f.write(f"- **Clinical Report**: `{files['report']}`\n")
            f.write(f"- **Feature Importance**: `{files['features']}`\n")
            f.write(f"- **Performance Metrics**: `{files['metrics']}`\n")
            f.write("\n")
    
    logger.log(f"Summary report saved to: {summary_md}")
    logger.log("")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="CDDA Paper Analysis Script - 完整的系統測試用於論文撰寫"
    )
    
    parser.add_argument(
        '--subject',
        type=str,
        help='單個受試者 ID (例如: sub-0005)'
    )
    
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        help='多個受試者 ID (例如: sub-0001 sub-0002 sub-0003)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/paper_results',
        help='輸出目錄 (默認: output/paper_results)'
    )
    
    parser.add_argument(
        '--orchestrator-path',
        type=str,
        default='D:/hf_models/Phi-4-mini-instruct',
        help='Orchestrator 模型路徑'
    )
    
    parser.add_argument(
        '--consultant-path',
        type=str,
        default='D:/hf_models/Llama3.1-Aloe-Beta-8B',
        help='Consultant 模型路徑'
    )
    
    parser.add_argument(
        '--use-4bit',
        action='store_true',
        default=True,
        help='使用 4-bit 量化'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='禁用 LLM 模式 (使用規則決策)'
    )
    
    parser.add_argument(
        '--ground-truth-file',
        type=str,
        help='包含真實標籤的 JSON 文件 (格式: {"sub-0001": "AD", ...})'
    )
    
    args = parser.parse_args()
    
    # 確定要分析的受試者列表
    if args.subject:
        subjects = [args.subject]
    elif args.subjects:
        subjects = args.subjects
    else:
        # 如果沒有指定，掃描所有可用受試者
        import glob
        data_folders = glob.glob("data/MRI_processed/*/sub-*")
        subjects = sorted(list(set([Path(f).name for f in data_folders])))
        
        if not subjects:
            print("ERROR: No subjects found in data/MRI_processed/")
            print("Please specify subjects using --subject or --subjects")
            sys.exit(1)
        
        print(f"Found {len(subjects)} subjects in data/MRI_processed/")
        print("Analyzing all subjects...")
    
    # 加載真實標籤 (如果提供)
    ground_truths = {}
    if args.ground_truth_file:
        with open(args.ground_truth_file, 'r') as f:
            ground_truths = json.load(f)
    else:
        # 嘗試從數據目錄結構推斷
        import glob
        for folder_path in glob.glob("data/MRI_processed/*/sub-*"):
            parts = folder_path.replace('\\', '/').split('/')
            if len(parts) >= 3:
                subject_id = parts[-1]
                label = parts[-2]
                ground_truths[subject_id] = label
    
    # 初始化日誌記錄器
    logger = PaperAnalysisLogger(args.output)
    
    # 打印系統信息
    print_system_info(logger)
    
    # 初始化 CDDA Agent
    logger.section("INITIALIZING CDDA AGENT")
    
    logger.log(f"Orchestrator Model: {args.orchestrator_path}")
    logger.log(f"Consultant Model: {args.consultant_path}")
    logger.log(f"Use 4-bit Quantization: {args.use_4bit}")
    logger.log(f"LLM Mode: {not args.no_llm}")
    logger.log("")
    
    try:
        agent = CDDAAgent(
            orchestrator_model="phi-4-mini",
            orchestrator_model_path=args.orchestrator_path,
            consultant_model="llama3.1-aloe-beta-8b",
            consultant_model_path=args.consultant_path,
            use_llm=not args.no_llm,
            use_4bit=args.use_4bit,
            verbose=False  # 關閉內部詳細輸出，使用我們的日誌
        )
        
        logger.log("CDDA Agent initialized successfully!")
        logger.log("")
        
    except Exception as e:
        logger.log(f"Failed to initialize CDDA Agent: {e}", level="ERROR")
        import traceback
        logger.log(traceback.format_exc(), level="ERROR")
        sys.exit(1)
    
    # 分析所有受試者
    all_results = []
    
    for i, subject_id in enumerate(subjects, 1):
        logger.log("")
        logger.log("=" * 80)
        logger.log(f"SUBJECT {i}/{len(subjects)}: {subject_id}")
        logger.log("=" * 80)
        logger.log("")
        
        ground_truth = ground_truths.get(subject_id)
        
        result = analyze_subject(
            agent=agent,
            subject_id=subject_id,
            logger=logger,
            ground_truth=ground_truth
        )
        
        if result:
            all_results.append(result)
        
        logger.log("")
    
    # 生成總結報告
    if len(all_results) > 1:
        generate_summary_report(all_results, logger, ground_truths)
    
    # 最終總結
    logger.section("ANALYSIS COMPLETE")
    
    logger.log(f"Total Subjects Analyzed: {len(all_results)}/{len(subjects)}")
    logger.log(f"Output Directory: {logger.output_dir}")
    logger.log(f"Summary Report: {logger.summary_file}")
    logger.log("")
    logger.log("All results have been saved to the output directory.")
    logger.log("You can now use these files for your paper!")
    logger.log("")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {logger.output_dir}")
    print(f"Summary report: {logger.summary_file}")
    print("\nGenerated files:")
    print(f"  - Logs: {logger.logs_dir}")
    print(f"  - Reports: {logger.reports_dir}")
    print(f"  - Reasoning Chains: {logger.reasoning_dir}")
    print(f"  - Metrics: {logger.metrics_dir}")
    print("\nYou can now use these files for your paper!")


if __name__ == "__main__":
    main()
