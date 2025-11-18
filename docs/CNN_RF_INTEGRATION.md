# CNN-RF 模型整合文檔

## 概述

CNN-RF 模型已完全整合到 Cognivex 系統中，提供基於 AAL3 圖譜的結構性 MRI 分析和阿茲海默症診斷。

## 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                    Cognivex App                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │  Functional  │      │  Structural  │                     │
│  │  MRI Branch  │      │  MRI Branch  │                     │
│  └──────────────┘      └──────────────┘                     │
│                              │                                │
│                        ┌─────┴─────┐                        │
│                        │           │                         │
│                   ┌────▼────┐ ┌───▼────┐                   │
│                   │ Legacy  │ │ CNN-RF │                   │
│                   │  Model  │ │ Model  │                   │
│                   └─────────┘ └────────┘                   │
│                                    │                         │
│                        ┌───────────┴───────────┐           │
│                        │                       │            │
│                   ┌────▼────┐          ┌──────▼──────┐    │
│                   │ NC vs AD│          │ NC/MCI/AD   │    │
│                   │  Binary │          │  3-class    │    │
│                   └─────────┘          └─────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 數據流程

### 1. 數據準備

```
data/MRI_processed/
├── NC/
│   └── sub-0002/
│       ├── sub-0002_GM_to_MNI.nii.gz   # 灰質
│       ├── sub-0002_FA_to_MNI.nii.gz   # 分數各向異性
│       └── sub-0002_MD_to_MNI.nii.gz   # 平均擴散率
├── MCI/
└── AD/
```

**注意**: 系統現在支持**端到端處理**，直接從原始 MRI 影像提取特徵並進行預測，不再依賴預先計算的 CSV 檔案。

### 2. 模型文件

```
model/cnn_rf/
├── rf_model_NC_vs_AD.joblib      # NC vs AD 二分類模型 ✓
└── rf_model_NC_MCI_AD.joblib     # 三分類模型
```

### 3. 端到端工作流程

```
原始 MRI 影像 (data/MRI_processed)
    ↓
定位受試者目錄 (NC/MCI/AD)
    ↓
提取 ROI 特徵 (GM, FA, MD)
    ↓
載入 CNN-RF 模型
    ↓
預測診斷
    ↓
返回結果 + 真實標籤
```

```python
# 步驟 1: 初始化狀態
initial_state = {
    "subject_id": "sub-0005",
    "analysis_mode": "structural",  # 使用結構性 MRI
    "model_type": "cnn_rf",         # 使用 CNN-RF 模型
    "model_name": "NC_vs_AD",       # 選擇模型
    "data_root": "data/MRI_processed"  # 數據根目錄
}

# 步驟 2: 執行工作流 (端到端處理)
final_state = app.invoke(initial_state)

# 步驟 3: 獲取結果
prediction = final_state['classification_result']
confidence = final_state['prediction_confidence']
true_label = final_state['true_label']
correct = final_state['correct_prediction']
```

## 使用方法

### 方法 1: 通過 Workflow (推薦)

```python
from app.graph.workflow import app

# 使用 CNN-RF 模型
state = {
    "subject_id": "sub-0005",
    "analysis_mode": "structural",
    "model_type": "cnn_rf",
    "model_name": "NC_vs_AD",
    "trace_log": [],
    "error_log": [],
}

result = app.invoke(state)
print(f"Prediction: {result['classification_result']}")
print(f"Confidence: {result['prediction_confidence']:.1%}")
```

### 方法 2: 直接調用 Agent (端到端)

```python
from app.agents.cnn_rf_inference import run_cnn_rf_inference

state = {
    "subject_id": "sub-0005",
    "model_name": "NC_vs_AD",
    "data_root": "data/MRI_processed"  # 可選，預設為此路徑
}

result = run_cnn_rf_inference(state)

# 結果包含
print(f"預測: {result['classification_result']}")
print(f"信心: {result['prediction_confidence']:.1%}")
print(f"真實標籤: {result['true_label']}")
print(f"正確: {result['correct_prediction']}")
print(f"受試者目錄: {result['subject_directory']}")
```

### 方法 3: 使用端到端 API

```python
from scripts.cnn_rf.end_to_end_inference import EndToEndPredictor

# 初始化端到端預測器
predictor = EndToEndPredictor(
    model_path="model/cnn_rf/rf_model_NC_vs_AD.joblib",
    data_root="data/MRI_processed"
)

# 直接從原始影像預測
results = predictor.predict_subject('sub-0005')

print(f"預測: {results['predicted_label']}")
print(f"信心: {results['confidence']:.1%}")
print(f"真實標籤: {results['true_label']}")
print(f"正確: {results['correct']}")
```

## 測試

### 端到端推理測試

```bash
python app/test_end_to_end_inference.py
```

這會測試：
1. 基本端到端推理（從原始 MRI 影像）
2. 帶可視化的推理
3. 多個受試者批量處理

### Workflow 整合測試

```bash
python app/test_cnn_rf_integration.py --mode single
```

輸出：
```
================================================================================
Testing CNN-RF Integration
================================================================================

🚀 Starting CNN-RF pipeline for subject: sub-0005
   Analysis mode: structural
   Model type: cnn_rf
   Model name: NC_vs_AD
================================================================================

[1/5] Loading CNN-RF model...
   ✓ Model loaded: rf_model_NC_vs_AD.joblib
   ✓ Classes: ['AD', 'NC']

[2/5] Loading ROI features...
   ✓ Loaded features for sub-0005
   ✓ Feature count: 342

[3/5] Performing prediction...
   🎯 Prediction Results:
      Classification: AD
      Confidence: 87.3%
      Probabilities:
         AD: 87.3%
         NC: 12.7%

✅ Pipeline completed successfully!
```

### 多個受試者測試

```bash
python app/test_cnn_rf_integration.py --mode multiple
```

### 模型比較測試

```bash
python app/test_cnn_rf_integration.py --mode compare
```

## 配置選項

### 模型選擇

```python
# 二分類 (NC vs AD)
state = {
    "model_name": "NC_vs_AD",
    "model_type": "cnn_rf"
}

# 三分類 (NC vs MCI vs AD)
state = {
    "model_name": "NC_MCI_AD",
    "model_type": "cnn_rf"
}
```

### 可視化選項

```python
# 包含腦區可視化
from app.agents.cnn_rf_inference import run_cnn_rf_inference_with_visualization

result = run_cnn_rf_inference_with_visualization(state)
brain_map = result['brain_map_path']  # 3D 腦區地圖
```

## 輸出結果

### 端到端輸出

```python
{
    "classification_result": "AD",           # 預測類別
    "prediction_confidence": 0.873,          # 信心分數
    "prediction_probabilities": {            # 各類別概率
        "AD": 0.873,
        "NC": 0.127
    },
    "true_label": "AD",                      # 真實標籤（從目錄結構）
    "correct_prediction": True,              # 預測是否正確
    "subject_directory": "data/MRI_processed/AD/sub-0005",  # 受試者目錄
    "roi_features": {                        # 即時提取的 ROI 特徵值
        "Precentral_L_GM": 0.256,
        "Precentral_L_FA": 0.311,
        ...
    },
    "feature_importances": {                 # 特徵重要性
        0: 0.0234,
        1: 0.0189,
        ...
    },
    "brain_map_path": "output/cnn_rf/sub-0005_brain_map.nii.gz"  # 可選
}
```

## 重要腦區 (NC vs AD)

根據模型訓練結果，以下腦區對 AD 診斷最重要：

| 排名 | 腦區 | 中文名稱 | 臨床意義 |
|------|------|----------|----------|
| 1 | Olfactory_L | 左側嗅覺皮層 | AD 早期嗅覺功能下降 |
| 2 | OFCant_L | 左側前眶額皮層 | 執行功能和決策 |
| 3 | OFCant_R | 右側前眶額皮層 | 執行功能和決策 |
| 4 | Cingulate_Post_R | 右側後扣帶回 | AD 典型受損區域 |
| 5 | ParaHippocampal_R | 右側海馬旁回 | 記憶形成關鍵區域 |
| 6 | Calcarine_R | 右側距狀裂 | 視覺處理 |
| 7 | Lingual_R | 右側舌回 | 視覺處理 |
| 8 | Fusiform_R | 右側梭狀回 | 視覺處理 |
| 9 | Caudate_L | 左側尾狀核 | 運動和認知功能 |

## 性能指標

### NC vs AD 二分類
- **準確率**: 89.1% ± 3.7%
- **AD 召回率**: 83%
- **NC 召回率**: 93%
- **F1-score**: 0.88

### 特徵數量
- **原始特徵**: 342 個 (114 ROIs × 3 modalities)
- **選擇後**: 30 個最重要特徵
- **降維比例**: 91.2%

## 與現有系統的整合

### 1. Workflow 整合

CNN-RF 已整合到主 workflow 中，通過 `analysis_mode` 和 `model_type` 參數控制：

```python
# app/graph/workflow.py

def route_by_analysis_mode(state: AgentState) -> str:
    mode = state.get("analysis_mode", "functional")
    model_type = state.get("model_type", "legacy")
    
    if mode == "structural":
        if model_type == "cnn_rf":
            return "cnn_rf_inference"  # 新的 CNN-RF 分支
        else:
            return "structural_mri_inference"  # 舊的 legacy 分支
    else:
        return "inference"  # Functional MRI 分支
```

### 2. 數據路徑更新

系統現在使用 `data/MRI_processed` 作為主要數據源：

```python
# app/core/ml_processing/cnn_rf_config.py

DATA_ROOT = Path("data/MRI_processed")
ROI_FEATURES_PATH = Path("data/roi_features.csv")
```

### 3. 向後兼容

舊的 structural MRI 推理仍然可用：

```python
# 使用舊模型
state = {
    "analysis_mode": "structural",
    "model_type": "legacy",  # 使用舊的 RF 模型
    "fmri_scan_path": "data/sMRI/AD/sub-0005/sub_0005_T1.nii.gz"
}

# 使用新模型
state = {
    "analysis_mode": "structural",
    "model_type": "cnn_rf",  # 使用新的 CNN-RF 模型
    "subject_id": "sub-0005"
}
```

## 故障排除

### 問題 1: 找不到受試者目錄

```
Error: Subject sub-XXXX not found in data/MRI_processed
```

**解決方案**: 確認受試者目錄存在於 `data/MRI_processed/NC/`, `data/MRI_processed/MCI/`, 或 `data/MRI_processed/AD/` 中

### 問題 2: 模型文件不存在

```
Error: Model not found: model/cnn_rf/rf_model_NC_vs_AD.joblib
```

**解決方案**: 運行 `python scripts/cnn_rf/train_feat.py` 訓練模型

### 問題 3: 導入錯誤

```
ModuleNotFoundError: No module named 'scripts.cnn_rf'
```

**解決方案**: 確保項目根目錄在 Python 路徑中

## 主要特性

✅ **端到端處理** - 直接從原始 MRI 影像到診斷結果  
✅ **即時特徵提取** - 不依賴預先計算的 CSV 檔案  
✅ **自動驗證** - 與真實標籤比對，計算準確率  
✅ **多模態支持** - GM, FA, MD 三種模態  
✅ **AAL3 圖譜** - 166 個腦區的精細分析  

## 下一步

1. **訓練三分類模型** - 支持 MCI 診斷
2. **Web API** - 提供 REST API 接口
3. **批量處理優化** - 並行處理多個受試者
4. **可視化增強** - 3D 腦區交互式可視化
5. **模型優化** - 超參數調整和特徵選擇

## 參考文檔

- [CNN-RF README](../scripts/cnn_rf/README.md)
- [整合腳本](../scripts/cnn_rf/integrate_system.py)
- [測試腳本](../app/test_cnn_rf_integration.py)
- [配置文件](../app/core/ml_processing/cnn_rf_config.py)
