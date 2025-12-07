#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test Script for Binary Statistics

快速測試 binary_statistics.py 的功能
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.paper.binary_statistics import BinaryStatistics
from app.core.models.context_models import (
    AgentResult, ContextObject, DiagnosticReport, 
    Feature, AnomalyStatus
)


def create_mock_result(subject_id: str, prediction: str, confidence: float, 
                       uq_score: float, ground_truth: str, 
                       agent_decision: str = "STANDARD_REPORT") -> AgentResult:
    """創建模擬的 AgentResult 用於測試"""
    
    # 創建模擬特徵
    features = [
        Feature(
            roi_name=f"Hippocampus_L",
            feature_name=f"Hippocampus_L_GM",
            feature_value=0.5,
            z_score=-2.5,
            shap_value=0.15,
            rank=1
        ),
        Feature(
            roi_name=f"Amygdala_R",
            feature_name=f"Amygdala_R_GM",
            feature_value=0.6,
            z_score=-1.8,
            shap_value=0.12,
            rank=2
        )
    ]
    
    # 創建異常狀態
    anomaly_status = AnomalyStatus(
        has_anomaly=(uq_score > 0.8),
        anomalous_regions=["Hippocampus_L"] if uq_score > 0.8 else [],
        threshold_used=2.5
    )
    
    # 創建診斷報告
    diagnostic_report = DiagnosticReport(
        subject_id=subject_id,
        prediction_result=prediction,
        confidence=confidence,
        uq_score=uq_score,
        top_features=features,
        anomaly_status=anomaly_status
    )
    
    # 創建 ContextObject
    context_object = ContextObject(
        subject_id=subject_id,
        diagnostic_report=diagnostic_report,
        decision_rationale="Test decision",
        signals={'uq_score': uq_score},
        agent_a_reasoning=[
            f"[System] Inference complete for {subject_id} using rf_model_{subject_id}.joblib"
        ],
        mcp_actions=[]
    )
    
    # 創建 AgentResult
    result = AgentResult(
        subject_id=subject_id,
        agent_decision=agent_decision,
        prediction=prediction,
        confidence=confidence,
        uq_score=uq_score,
        context_object=context_object,
        clinical_report="Test clinical report",
        reasoning_chain=[
            f"[System] Inference complete for {subject_id} using rf_model_{subject_id}.joblib",
            "[Agent A] Orchestration complete",
            "[Agent B] Clinical synthesis complete"
        ]
    )
    
    return result


def test_basic_functionality():
    """測試基本功能"""
    print("\n" + "="*80)
    print("Testing Binary Statistics - Basic Functionality")
    print("="*80)
    
    # 創建統計分析器
    stats = BinaryStatistics("output/test_binary_stats")
    
    # 添加模擬結果
    test_cases = [
        # (subject_id, prediction, confidence, uq, ground_truth, agent_decision)
        ("sub-001", "AD", 0.95, 0.3, "AD", "STANDARD_REPORT"),           # TP - Standard
        ("sub-002", "NC", 0.92, 0.4, "NC", "STANDARD_REPORT"),           # TN - Standard
        ("sub-003", "AD", 0.55, 0.85, "AD", "SIMULATION_TRIGGERED"),     # TP - Corrected by CF!
        ("sub-004", "NC", 0.88, 0.5, "AD", "STANDARD_REPORT"),           # FN - Standard
        ("sub-005", "AD", 0.91, 0.35, "AD", "STANDARD_REPORT"),          # TP - Standard
        ("sub-006", "NC", 0.78, 0.82, "NC", "ANOMALY_INVESTIGATION"),    # TN - Corrected by KG!
    ]
    
    for subject_id, pred, conf, uq, gt, decision in test_cases:
        result = create_mock_result(subject_id, pred, conf, uq, gt, decision)
        stats.add_result(
            subject_id=subject_id,
            result=result,
            ground_truth=gt,
            init_time=0.5,
            analysis_time=2.0
        )
        print(f"✓ Added: {subject_id} - Pred: {pred}, GT: {gt}, Conf: {conf:.2f}, UQ: {uq:.2f}")
    
    # 計算統計
    stats.calculate_statistics()
    
    # 檢查 LOOCV 驗證
    loocv_stats = stats.statistics['loocv_integrity']
    print(f"\n[LOOCV Verification]")
    print(f"  Verified: {loocv_stats['loocv_verified']}")
    print(f"  Coverage: {loocv_stats.get('coverage_percentage', 0):.2f}%")
    
    # 檢查二分類指標
    metrics = stats.statistics['binary_metrics']
    print(f"\n[Binary Metrics]")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # 檢查系統價值分析 (NEW!)
    system_value = stats.statistics['system_value']
    print(f"\n[System Value Analysis]")
    print(f"  Total Interventions: {len(system_value['intervention_cases'])}")
    print(f"  Total Corrections: {system_value['total_corrections']}")
    print(f"  Corrected by Counterfactual: {len(system_value['corrected_by_counterfactual'])}")
    print(f"  Corrected by Knowledge: {len(system_value['corrected_by_knowledge'])}")
    if system_value['intervention_accuracy'] > 0:
        print(f"  Intervention Accuracy: {system_value['intervention_accuracy']:.4f}")
        print(f"  Standard Accuracy: {system_value['standard_accuracy']:.4f}")
        print(f"  Accuracy Improvement: {system_value['accuracy_improvement']:+.4f}")
    
    # 生成報告
    print(f"\n[Generating Reports]")
    report_file = stats.save_report("test_report.txt")
    print(f"  ✓ Text report: {report_file}")
    
    json_file = stats.save_json("test_stats.json")
    print(f"  ✓ JSON data: {json_file}")
    
    csv_file = stats.save_csv("test_results.csv")
    print(f"  ✓ CSV data: {csv_file}")
    
    latex_file = stats.save_latex_table("test_table.tex")
    print(f"  ✓ LaTeX table: {latex_file}")
    
    print(f"\n✓ All tests passed!")
    print("="*80)


def test_loocv_verification():
    """測試 LOOCV 驗證邏輯"""
    print("\n" + "="*80)
    print("Testing LOOCV Verification Logic")
    print("="*80)
    
    stats = BinaryStatistics("output/test_loocv")
    
    # 測試案例 1: 正確使用專屬模型
    reasoning_chain_1 = [
        "Inference complete for sub-001 using rf_model_sub-001.joblib"
    ]
    status_1, model_1 = stats._verify_model_usage("sub-001", reasoning_chain_1)
    print(f"\nTest 1: Correct LOOCV model")
    print(f"  Status: {status_1} (expected: loocv_verified)")
    print(f"  Model: {model_1}")
    assert status_1 == "loocv_verified", "Test 1 failed!"
    
    # 測試案例 2: 使用通用模型
    reasoning_chain_2 = [
        "Inference complete for sub-002 using rf_model_NC_vs_AD.joblib"
    ]
    status_2, model_2 = stats._verify_model_usage("sub-002", reasoning_chain_2)
    print(f"\nTest 2: Global fallback model")
    print(f"  Status: {status_2} (expected: fallback_global)")
    print(f"  Model: {model_2}")
    assert status_2 == "fallback_global", "Test 2 failed!"
    
    # 測試案例 3: 無法判斷
    reasoning_chain_3 = [
        "Some other log message without model info"
    ]
    status_3, model_3 = stats._verify_model_usage("sub-003", reasoning_chain_3)
    print(f"\nTest 3: Unknown model")
    print(f"  Status: {status_3} (expected: unknown)")
    print(f"  Model: {model_3}")
    assert status_3 == "unknown", "Test 3 failed!"
    
    print(f"\n✓ All LOOCV verification tests passed!")
    print("="*80)


def test_latex_generation():
    """測試 LaTeX 表格生成"""
    print("\n" + "="*80)
    print("Testing LaTeX Table Generation")
    print("="*80)
    
    stats = BinaryStatistics("output/test_latex")
    
    # 添加一些模擬數據
    for i in range(10):
        result = create_mock_result(
            f"sub-{i:03d}", 
            "AD" if i % 2 == 0 else "NC",
            0.9,
            0.4,
            "AD" if i % 2 == 0 else "NC"
        )
        stats.add_result(f"sub-{i:03d}", result, "AD" if i % 2 == 0 else "NC", 0.5, 2.0)
    
    stats.calculate_statistics()
    
    # 生成 LaTeX 表格
    latex_table = stats.generate_latex_table()
    
    print("\n[Generated LaTeX Table]")
    print(latex_table)
    
    # 驗證包含關鍵元素
    assert "\\begin{table}" in latex_table
    assert "Accuracy" in latex_table
    assert "Precision" in latex_table
    assert "\\end{table}" in latex_table
    
    print(f"\n✓ LaTeX generation test passed!")
    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BINARY STATISTICS TEST SUITE")
    print("="*80)
    
    try:
        # 運行所有測試
        test_basic_functionality()
        test_loocv_verification()
        test_latex_generation()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
