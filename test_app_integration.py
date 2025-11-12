"""
測試 app.py 整合 - 驗證所有組件正常載入
"""

import sys
from pathlib import Path

print("="*70)
print("🧪 測試 app.py 整合")
print("="*70)

# Test 1: 測試 imports
print("\n[Test 1] 測試所有 imports...")
try:
    # 測試主要 imports
    import streamlit as st
    from nilearn import plotting, image as nimg
    
    # 測試 workflow
    from app.graph.workflow import app
    
    # 測試 UI 組件
    from app.ui.structural_mri_components import (
        render_analysis_mode_selector,
        render_structural_results
    )
    
    print("✅ 所有 imports 成功")
    
except Exception as e:
    print(f"❌ Import 失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: 測試 workflow 路由
print("\n[Test 2] 測試 workflow 路由...")
try:
    from app.graph.workflow import route_by_analysis_mode
    
    # 測試不同模式
    test_cases = [
        ({"analysis_mode": "structural"}, "structural_mri_inference"),
        ({"analysis_mode": "functional"}, "inference"),
        ({}, "inference")
    ]
    
    all_passed = True
    for state, expected in test_cases:
        result = route_by_analysis_mode(state)
        if result == expected:
            mode = state.get("analysis_mode", "default")
            print(f"   ✅ {mode:12s} -> {result}")
        else:
            print(f"   ❌ {state} -> {result} (expected {expected})")
            all_passed = False
    
    if all_passed:
        print("✅ Workflow 路由測試通過")
    else:
        print("❌ Workflow 路由測試失敗")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Workflow 測試失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: 測試 UI 組件可以被調用（不需要 Streamlit 運行）
print("\n[Test 3] 測試 UI 組件...")
try:
    from app.ui.structural_mri_components import (
        render_analysis_mode_selector,
        render_structural_results
    )
    
    # 檢查函數存在且可調用
    assert callable(render_analysis_mode_selector)
    assert callable(render_structural_results)
    
    print("✅ UI 組件可用")
    
except Exception as e:
    print(f"❌ UI 組件測試失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: 測試結構性 MRI agents
print("\n[Test 4] 測試結構性 MRI agents...")
try:
    from app.agents.structural_mri_inference import run_structural_mri_inference
    from app.agents.structural_feature_analyzer import analyze_feature_importance
    from app.agents.structural_visualizer import generate_structural_visualizations
    
    print("✅ 所有 agents 可用")
    
except Exception as e:
    print(f"❌ Agents 測試失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: 測試核心 ML 處理模組
print("\n[Test 5] 測試核心 ML 處理模組...")
try:
    from app.core.ml_processing.model_loader import MLModelLoader
    from app.core.ml_processing.feature_extractor import ROIFeatureExtractor
    from app.core.ml_processing.roi_names_zh import get_roi_display_name
    
    # 測試中文名稱
    zh_name = get_roi_display_name("Hippocampus_L", "zh")
    assert zh_name == "海馬迴（左）", f"Expected '海馬迴（左）', got '{zh_name}'"
    
    print("✅ 核心 ML 模組可用")
    print(f"   - 中文名稱測試: Hippocampus_L -> {zh_name}")
    
except Exception as e:
    print(f"❌ 核心模組測試失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: 檢查模型檔案
print("\n[Test 6] 檢查模型檔案...")
try:
    from app.core.ml_processing.config import ML_MODEL_CONFIG
    
    model_dir = Path(ML_MODEL_CONFIG["model_dir"])
    
    print(f"   模型目錄: {model_dir}")
    
    if model_dir.exists():
        model_files = list(model_dir.glob("*.pkl"))
        print(f"   找到 {len(model_files)} 個模型檔案:")
        for f in model_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"      - {f.name} ({size_mb:.2f} MB)")
        
        if len(model_files) >= 3:
            print("✅ 模型檔案完整")
        else:
            print("⚠️  模型檔案不完整（需要 3 個檔案）")
    else:
        print(f"⚠️  模型目錄不存在: {model_dir}")
        print("   提示: 可能需要先訓練模型")
    
except Exception as e:
    print(f"⚠️  模型檔案檢查失敗: {e}")

# Test 7: 模擬完整流程（不實際執行）
print("\n[Test 7] 模擬完整流程...")
try:
    # 模擬 initial_state
    mock_state = {
        "subject_id": "test_subject",
        "fmri_scan_path": "test.nii.gz",
        "model_path": None,
        "model_name": "random_forest",
        "analysis_mode": "structural",
        "trace_log": [],
        "error_log": []
    }
    
    print("   模擬狀態:")
    for key, value in mock_state.items():
        print(f"      {key}: {value}")
    
    # 測試路由
    from app.graph.workflow import route_by_analysis_mode
    next_node = route_by_analysis_mode(mock_state)
    
    print(f"\n   路由結果: {next_node}")
    assert next_node == "structural_mri_inference", f"Expected 'structural_mri_inference', got '{next_node}'"
    
    print("✅ 完整流程模擬成功")
    
except Exception as e:
    print(f"❌ 流程模擬失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "="*70)
print("📊 整合測試總結")
print("="*70)

print("\n✅ 所有測試通過！")
print("\n已驗證的組件:")
print("   1. ✅ Streamlit imports")
print("   2. ✅ Workflow 路由")
print("   3. ✅ UI 組件")
print("   4. ✅ Structural MRI agents")
print("   5. ✅ 核心 ML 模組")
print("   6. ✅ 中文名稱系統")
print("   7. ✅ 完整流程模擬")

print("\n🚀 系統準備就緒！")
print("\n下一步:")
print("   1. 啟動應用: streamlit run app.py")
print("   2. 選擇 'Structural MRI (T1)' 模式")
print("   3. 選擇受試者並開始分析")

print("\n" + "="*70)
