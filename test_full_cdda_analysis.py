#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整 CDDA 分析測試腳本
模擬前端介面的完整流程，包含詳細的過程輸出
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ============================================================================
# 輔助函數
# ============================================================================

def print_header(title):
    """打印標題"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_section(title):
    """打印章節"""
    print("\n" + "-"*80)
    print(f"  {title}")
    print("-"*80)


def print_step(step_num, description):
    """打印步驟"""
    print(f"\n[步驟 {step_num}] {description}")


def print_success(message):
    """打印成功訊息"""
    print(f"✓ {message}")


def print_error(message):
    """打印錯誤訊息"""
    print(f"✗ {message}")


def print_info(message):
    """打印資訊"""
    print(f"ℹ {message}")


def print_result(key, value):
    """打印結果"""
    print(f"  • {key}: {value}")


# ============================================================================
# 主測試流程
# ============================================================================

def main():
    """主測試函數"""
    
    print_header("CDDA 完整分析測試")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # 階段 1: 環境檢查與初始化
    # ========================================================================
    
    print_section("階段 1: 環境檢查與初始化")
    
    # 步驟 1.1: 導入模組
    print_step("1.1", "導入必要模組")
    try:
        from app.agents.cdda_agent import CDDAAgent
        print_success("CDDAAgent 導入成功")
        
        from app.core.models.context_models import DiagnosticReport, Feature
        print_success("數據模型導入成功")
        
        import glob
        print_success("標準庫導入成功")
        
    except Exception as e:
        print_error(f"模組導入失敗: {e}")
        return False
    
    # 步驟 1.2: 掃描可用受試者
    print_step("1.2", "掃描可用受試者資料")
    try:
        subject_labels = {}
        data_folders = glob.glob("data/MRI_processed/*/sub-*")
        
        print_info(f"找到 {len(data_folders)} 個受試者資料夾")
        
        for folder_path in data_folders:
            parts = folder_path.split('\\')  # Windows
            if len(parts) < 3:
                parts = folder_path.split('/')  # Unix
            
            if len(parts) >= 3:
                subject_id = parts[-1]
                label = parts[-2]
                
                # 檢查是否有完整的 MRI 文件
                nii_files = list(Path(folder_path).glob("*.nii.gz"))
                if len(nii_files) >= 3:
                    subject_labels[subject_id] = label
                    print_info(f"  ✓ {subject_id} ({label}): {len(nii_files)} 個 MRI 文件")
                else:
                    print_info(f"  ✗ {subject_id} ({label}): 只有 {len(nii_files)} 個文件 (需要 ≥3)")
        
        subject_list = sorted(subject_labels.keys())
        
        if not subject_list:
            print_error("沒有找到有完整數據的受試者")
            return False
        
        print_success(f"找到 {len(subject_list)} 個有效受試者")
        print_info(f"有效受試者列表: {', '.join(subject_list[:5])}{'...' if len(subject_list) > 5 else ''}")
        
    except Exception as e:
        print_error(f"掃描受試者失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 步驟 1.3: 選擇測試受試者
    print_step("1.3", "選擇測試受試者")
    
    # 使用第一個有效受試者
    selected_subject = subject_list[0]
    ground_truth = subject_labels[selected_subject]
    
    print_result("選擇的受試者", selected_subject)
    print_result("真實標籤", ground_truth)
    
    # ========================================================================
    # 階段 2: CDDA Agent 初始化
    # ========================================================================
    
    print_section("階段 2: CDDA Agent 初始化")
    
    # 步驟 2.1: 設定參數
    print_step("2.1", "設定 CDDA 參數")
    
    use_llm = True  # 使用規則式模式以加快測試
    use_4bit = True
    verbose = True
    
    print_result("LLM 模式", "啟用" if use_llm else "規則式 (Rule-based)")
    print_result("4-bit 量化", "啟用" if use_4bit else "停用")
    print_result("詳細輸出", "啟用" if verbose else "停用")
    
    # 步驟 2.2: 初始化 Agent
    print_step("2.2", "初始化 CDDA Agent")
    
    start_time = time.time()
    
    try:
        agent = CDDAAgent(
            orchestrator_model="phi-4-mini",
            orchestrator_model_path="D:/hf_models/Phi-4-mini-instruct",
            consultant_model="llama3.1-aloe-beta-8b",
            consultant_model_path=r"D:\hf_models\Llama3.1-Aloe-Beta-8B",
            use_llm=use_llm,
            use_4bit=use_4bit,
            verbose=verbose
        )
        
        init_time = time.time() - start_time
        print_success(f"CDDA Agent 初始化成功 (耗時: {init_time:.2f} 秒)")
        
    except Exception as e:
        print_error(f"Agent 初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # 階段 3: 執行分析
    # ========================================================================
    
    print_section("階段 3: 執行 CDDA 分析")
    
    print_step("3.1", f"開始分析受試者: {selected_subject}")
    print_info("這個過程可能需要 30-60 秒...")
    
    analysis_start = time.time()
    
    try:
        result = agent.run_analysis(selected_subject)
        
        analysis_time = time.time() - analysis_start
        print_success(f"分析完成！(耗時: {analysis_time:.2f} 秒)")
        
    except Exception as e:
        print_error(f"分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # 階段 4: 結果展示
    # ========================================================================
    
    print_section("階段 4: 分析結果")
    
    # 步驟 4.1: 基本診斷資訊
    print_step("4.1", "基本診斷資訊")
    
    diagnosis_map = {
        'AD': '阿茲海默症 (Alzheimer\'s Disease)',
        'MCI': '輕度認知障礙 (Mild Cognitive Impairment)',
        'NC': '正常認知 (Normal Cognition)'
    }
    
    print_result("受試者編號", result.subject_id)
    print_result("真實標籤", diagnosis_map.get(ground_truth, ground_truth))
    print_result("AI 預測", diagnosis_map.get(result.prediction, result.prediction))
    print_result("預測正確性", "✓ 正確" if ground_truth == result.prediction else "✗ 錯誤")
    print_result("信心度", f"{result.confidence:.1%}")
    print_result("不確定性評分", f"{result.uq_score:.3f}")
    print_result("分析模式", result.agent_decision)
    
    # 步驟 4.2: 執行摘要
    print_step("4.2", "執行摘要 (Executive Summary)")
    
    if 'executive_summary' in result.metadata:
        summary = result.metadata['executive_summary']
        
        print("\n📋 標題:")
        print(f"  {summary.get('headline', 'N/A')}")
        
        print("\n🔍 關鍵發現:")
        for i, finding in enumerate(summary.get('key_findings', []), 1):
            print(f"  {i}. {finding}")
        
        print("\n💡 建議行動:")
        for i, action in enumerate(summary.get('recommended_actions', []), 1):
            print(f"  {i}. {action}")
        
        print(f"\n⚠️ 風險等級: {summary.get('risk_level', 'N/A')}")
    else:
        print_info("執行摘要不可用")
    
    # 步驟 4.3: 關鍵特徵
    print_step("4.3", "Top 5 診斷驅動因子")
    
    if result.context_object and result.context_object.diagnostic_report:
        top_features = result.context_object.diagnostic_report.top_features[:5]
        
        print("\n排名 | 腦區 | SHAP 值 | Z-score")
        print("-" * 60)
        
        for feat in top_features:
            # 安全地獲取屬性
            if isinstance(feat, dict):
                roi_name = feat.get('roi_name', 'Unknown')
                shap_value = feat.get('shap_value', 0)
                z_score = feat.get('z_score', 0)
                rank = feat.get('rank', 0)
            else:
                roi_name = getattr(feat, 'roi_name', 'Unknown')
                shap_value = getattr(feat, 'shap_value', 0)
                z_score = getattr(feat, 'z_score', 0)
                rank = getattr(feat, 'rank', 0)
            
            print(f"{rank:4d} | {roi_name:30s} | {shap_value:+.4f} | {z_score:+.2f}")
    else:
        print_info("特徵資訊不可用")
    
    # 步驟 4.4: 工具調用結果
    print_step("4.4", "Agent 工具調用")
    
    if result.context_object and result.context_object.tool_results:
        tool_results = result.context_object.tool_results
        
        if 'counterfactual' in tool_results:
            cf = tool_results['counterfactual']
            print("\n🔬 反事實模擬:")
            print_result("原始預測", f"{cf.get('original_prediction', 'N/A')} ({cf.get('original_confidence', 0):.1%})")
            print_result("模擬後預測", f"{cf.get('new_prediction', 'N/A')} ({cf.get('new_confidence', 0):.1%})")
            print_result("信心度變化", f"{cf.get('confidence_delta', 0):+.1%}")
            print_result("解釋", cf.get('interpretation', 'N/A')[:100] + "...")
        
        if 'knowledge_context' in tool_results:
            kc = tool_results['knowledge_context']
            anomalous_regions = kc.get('query_regions', [])
            print("\n⚠️ 異常檢測:")
            print_result("異常腦區數量", len(anomalous_regions))
            if anomalous_regions:
                print_result("異常腦區", ', '.join(anomalous_regions[:5]))
                summary = kc.get('summary', '')
                if summary:
                    print_result("臨床背景", summary[:150] + "...")
    else:
        print_info("標準診斷流程，未調用額外工具")
    
    # 步驟 4.5: 臨床報告
    print_step("4.5", "臨床報告預覽")
    
    if hasattr(result, 'clinical_report') and result.clinical_report:
        report_preview = result.clinical_report[:500]
        print("\n" + "─"*80)
        print(report_preview)
        if len(result.clinical_report) > 500:
            print(f"\n... (還有 {len(result.clinical_report) - 500} 個字符)")
        print("─"*80)
    else:
        print_info("臨床報告不可用")
    
    # 步驟 4.6: 推理鏈統計
    print_step("4.6", "推理鏈統計")
    
    if result.reasoning_chain:
        print_result("推理步驟總數", len(result.reasoning_chain))
        
        # 統計各階段
        agent_a_steps = sum(1 for step in result.reasoning_chain if 'AGENT A' in step)
        agent_b_steps = sum(1 for step in result.reasoning_chain if 'AGENT B' in step)
        mcp_actions = sum(1 for step in result.reasoning_chain if 'MCP' in step)
        
        print_result("Agent A 步驟", agent_a_steps)
        print_result("Agent B 步驟", agent_b_steps)
        print_result("MCP 動作", mcp_actions)
    
    # 步驟 4.7: 元數據
    print_step("4.7", "分析元數據")
    
    print_result("時間戳", result.timestamp)
    print_result("使用 LLM", result.metadata.get('use_llm', False))
    print_result("Agent A 步驟", result.metadata.get('agent_a_steps', 0))
    print_result("Agent B 步驟", result.metadata.get('agent_b_steps', 0))
    print_result("MCP 動作", result.metadata.get('mcp_actions', 0))
    
    # ========================================================================
    # 階段 5: 性能統計
    # ========================================================================
    
    print_section("階段 5: 性能統計")
    
    total_time = time.time() - start_time
    
    print_result("總執行時間", f"{total_time:.2f} 秒")
    print_result("初始化時間", f"{init_time:.2f} 秒 ({init_time/total_time*100:.1f}%)")
    print_result("分析時間", f"{analysis_time:.2f} 秒 ({analysis_time/total_time*100:.1f}%)")
    
    # ========================================================================
    # 階段 6: 保存結果
    # ========================================================================
    
    print_section("階段 6: 保存結果")
    
    print_step("6.1", "生成測試報告")
    
    try:
        output_dir = Path("output/test_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"test_report_{selected_subject}_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CDDA 完整分析測試報告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"受試者: {selected_subject}\n")
            f.write(f"真實標籤: {ground_truth}\n")
            f.write(f"AI 預測: {result.prediction}\n")
            f.write(f"信心度: {result.confidence:.1%}\n")
            f.write(f"不確定性: {result.uq_score:.3f}\n")
            f.write(f"分析模式: {result.agent_decision}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("完整臨床報告\n")
            f.write("-"*80 + "\n\n")
            f.write(result.clinical_report if hasattr(result, 'clinical_report') else "N/A")
            f.write("\n\n")
            
            f.write("-"*80 + "\n")
            f.write("完整推理鏈\n")
            f.write("-"*80 + "\n\n")
            for step in result.reasoning_chain:
                f.write(step + "\n")
        
        print_success(f"報告已保存: {report_file}")
        
    except Exception as e:
        print_error(f"保存報告失敗: {e}")
    
    # ========================================================================
    # 完成
    # ========================================================================
    
    print_header("測試完成")
    
    print_success("所有測試階段完成！")
    print_info(f"總耗時: {total_time:.2f} 秒")
    
    # 最終評估
    print("\n" + "="*80)
    print("最終評估:")
    print("="*80)
    
    if ground_truth == result.prediction:
        print("✓ 診斷正確")
    else:
        print("✗ 診斷錯誤")
    
    if result.confidence > 0.8:
        print("✓ 高信心度")
    elif result.confidence > 0.6:
        print("⚠ 中等信心度")
    else:
        print("✗ 低信心度")
    
    if result.uq_score < 0.5:
        print("✓ 低不確定性")
    elif result.uq_score < 0.8:
        print("⚠ 中等不確定性")
    else:
        print("✗ 高不確定性")
    
    print("="*80)
    
    return True


# ============================================================================
# 執行測試
# ============================================================================

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[中斷] 測試被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[錯誤] 未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
