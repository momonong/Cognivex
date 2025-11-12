# Quick Start Guide - ML Model Integration

## 🚀 5 分鐘快速開始

### 步驟 1: 驗證模型檔案

```bash
ls -la model/ml/final/
```

應該看到：
- `final_model.pkl`
- `final_scaler.pkl`
- `final_roi_list.csv`
- `final_feature_names.txt`

### 步驟 2: 測試核心功能

```python
# 測試模型載入
from app.core.ml_processing import MLModelLoader

loader = MLModelLoader()
components = loader.get_all_components()
print(f"✓ 模型載入成功！特徵數量: {len(components['feature_names'])}")
```

### 步驟 3: 測試完整 workflow

```python
from app.graph.workflow import app

# 準備測試數據
initial_state = {
    "subject_id": "test_001",
    "fmri_scan_path": "path/to/your/t1_mri.nii.gz",
    "analysis_mode": "structural",
    "trace_log": [],
    "error_log": []
}

# 執行分析
final_state = app.invoke(initial_state)

# 檢查結果
print(f"分類結果: {final_state['classification_result']}")
print(f"信心分數: {final_state['prediction_confidence']:.1%}")
print(f"Top 5 重要 ROI:")
for region in final_state['activated_regions'][:5]:
    print(f"  - {region['region_name']}: {region['activation_score']*100:.2f}%")
```

### 步驟 4: 整合到 UI

在 `app.py` 中加入：

```python
# 1. Import
from app.ui import render_analysis_mode_selector, render_structural_results

# 2. 側邊欄加入模式選擇
analysis_mode = render_analysis_mode_selector()

# 3. 在 initial_state 加入 analysis_mode
initial_state = {
    ...
    "analysis_mode": analysis_mode,
}

# 4. 結果顯示
if analysis_mode == "structural":
    render_structural_results(final_state, ground_truth)
```

### 步驟 5: 啟動應用

```bash
streamlit run app.py
```

## 🧪 快速測試

```bash
# 執行所有測試
pytest tests/ -v

# 只測試 ML 模組
pytest tests/test_ml_model_loader.py tests/test_roi_feature_extractor.py -v

# 測試 agents
pytest tests/test_structural_agents.py -v

# 測試 workflow
pytest tests/test_structural_workflow_integration.py -v
```

## 📊 預期輸出

成功執行後，你應該看到：

```
=== Loading ML Model Components ===
✓ Loaded Random Forest model from model/ml/final/final_model.pkl
  - n_estimators: 500
  - n_features: 32
✓ Loaded StandardScaler from model/ml/final/final_scaler.pkl
✓ Loaded 32 ROIs from model/ml/final/final_roi_list.csv
✓ Loaded 32 feature names from model/ml/final/final_feature_names.txt
✓ All components loaded successfully
===================================

=== Extracting ROI Features ===
Input: path/to/mri.nii.gz
ROIs to extract: 32
✓ Extracted 32 features
===================================

🎯 Prediction Results:
   Classification: AD
   Confidence: 78.5%
   Probabilities: NC=21.5%, AD=78.5%

📊 Top 5 Important Features:
   1. Cingulum_Post_R: 0.0861 (8.61%)
   2. Lingual_R: 0.0635 (6.35%)
   3. Cingulum_Mid_L: 0.0614 (6.14%)
   4. Cingulum_Post_L: 0.0610 (6.10%)
   5. SupraMarginal_L: 0.0591 (5.91%)
```

## ⚠️ 常見問題

### 問題 1: ModuleNotFoundError

```bash
pip install scikit-learn nilearn nibabel pandas matplotlib seaborn
```

### 問題 2: Atlas 下載失敗

```python
# 手動下載 atlas
from nilearn import datasets
atlas = datasets.fetch_atlas_aal(version='SPM12')
```

### 問題 3: 模型檔案找不到

檢查路徑：
```bash
pwd
ls model/ml/final/
```

## 📝 下一步

1. 閱讀完整文件: `docs/ml_model_integration_summary.md`
2. 查看整合指南: `docs/app_py_integration_guide.md`
3. 了解設計細節: `.kiro/specs/ml-model-integration/design.md`

## 🎉 完成！

你現在已經成功整合了 ML 模型到你的應用中！
