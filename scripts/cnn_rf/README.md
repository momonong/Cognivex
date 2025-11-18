# CNN-RF 模型系統

## 概述

CNN-RF 是一個結合 3D CNN 特徵提取和隨機森林分類的阿茲海默症診斷系統。

### 核心特點

1. **多模態影像** - 整合 GM (灰質)、FA (分數各向異性)、MD (平均擴散率)
2. **AAL3 圖譜** - 基於 AAL3 腦區圖譜提取 ROI 特徵
3. **隨機森林分類** - 使用 RF 進行分類，提供特徵重要性
4. **腦區可視化** - 生成重要腦區的 3D 地圖

## 系統架構

```
data/
├── MRI_processed/          # 處理後的 MRI 數據
│   ├── NC/                 # 正常控制組
│   ├── MCI/                # 輕度認知障礙
│   └── AD/                 # 阿茲海默症
├── aal3/                   # AAL3 圖譜
│   ├── AAL3v1_1mm.nii.gz
│   └── AAL3v1.json
├── templates/              # MNI 模板
│   └── MNI152_T1_1mm_brain.nii.gz
└── roi_features.csv        # 提取的 ROI 特徵

model/cnn_rf/
├── rf_model_NC_vs_AD.joblib      # NC vs AD 二分類模型
└── rf_model_NC_MCI_AD.joblib     # 三分類模型

scripts/cnn_rf/
├── config.py               # 配置文件
├── model.py                # 3D CNN 模型定義
├── dataset.py              # 數據加載器
├── train_feat.py           # 訓練腳本
├── eval_feat.py            # 評估腳本
├── inference.py            # 推理接口
├── visualize_feat.py       # 腦區可視化
└── plot_feat.py            # 特徵可視化
```

## 數據格式

### MRI 數據結構
```
data/MRI_processed/
└── NC/
    └── sub-0002/
        ├── sub-0002_GM_to_MNI.nii.gz   # 灰質
        ├── sub-0002_FA_to_MNI.nii.gz   # 分數各向異性
        └── sub-0002_MD_to_MNI.nii.gz   # 平均擴散率
```

### ROI 特徵 CSV
```csv
Subject_ID,Group,Precentral_L_GM,Precentral_L_FA,...
sub-0002,NC,0.523,0.412,...
sub-0005,AD,0.445,0.389,...
```

## 使用方法

### 1. 配置檢查

```bash
python scripts/cnn_rf/config.py
```

輸出：
```
================================================================================
CNN-RF Model Configuration
================================================================================
Data root: data\MRI_processed
ROI features: data\roi_features.csv

Available models:
  [✓] NC_vs_AD: 二分類：正常控制組 vs 阿茲海默症
  [✗] NC_MCI_AD: 三分類：正常控制組 vs 輕度認知障礙 vs 阿茲海默症

Default model: NC_vs_AD
Output directory: output\cnn_rf
================================================================================
```

### 2. 訓練模型

```bash
# 訓練 NC vs AD 二分類模型
python scripts/cnn_rf/train_feat.py
```

輸出：
```
==================================================
--- 正在訓練模型: ['NC', 'AD'] ---
[*] 資料集大小: 65 人 ( 342 個特徵)
[*] 類別分佈:
NC    42
AD    23

[*] 正在執行 5 折交叉驗證 (Stratified K-Fold)...

--- 交叉驗證結果 ---
  每次的準確率: [0.923 0.846 0.923 0.846 0.917]
  平均準確率 (Mean Accuracy): 0.891
  標準差 (Std Dev):           0.037

[*] 正在於 *所有* 65 筆資料上訓練最終模型...
    ...最終模型訓練完成。
    [v] 成功儲存最終模型至: model/cnn_rf/rf_model_NC_vs_AD.joblib

--- 最重要的特徵 (Top Features) ---
  1. Olfactory_L_GM
  2. OFCant_L_FA
  3. OFCant_R_MD
  4. Cingulate_Post_R_GM
  5. ParaHippocampal_R_FA
  ...
```

### 3. 評估模型

```bash
python scripts/cnn_rf/eval_feat.py
```

輸出：
```
==================================================
--- 正在評估模型: NC vs AD (二分類) ---

--- 總體評估報告 (針對所有資料) ---
              precision    recall  f1-score   support

          AD       0.87      0.83      0.85        23
          NC       0.90      0.93      0.91        42

    accuracy                           0.89        65
   macro avg       0.89      0.88      0.88        65
weighted avg       0.89      0.89      0.89        65

--- 混淆矩陣 (Confusion Matrix) ---
 (Pred) AD        NC        
(True) AD      19         4          
(True) NC      3          39         
```

### 4. 使用模型進行預測

```python
from scripts.cnn_rf.inference import CNNRF_Predictor, load_roi_features

# 初始化預測器
predictor = CNNRF_Predictor(
    model_path="model/cnn_rf/rf_model_NC_vs_AD.joblib"
)

# 載入特徵
features = load_roi_features("data/roi_features.csv")

# 進行預測
results = predictor.predict(features)

# 查看結果
for detail in results['detailed']:
    print(f"Predicted: {detail['predicted_class']}")
    print(f"Confidence: {detail['confidence']:.3f}")
    print(f"Probabilities: {detail['probabilities']}")
```

### 5. 生成腦區可視化

```bash
python scripts/cnn_rf/visualize_feat.py
```

輸出：
```
[*] 載入 AAL3 模板: data/aal3/AAL3v1_1mm.nii.gz
[*] 載入 AAL3 標籤: data/aal3/AAL3v1.json
[*] 正在將 AAL3 圖譜重新採樣至 MNI 空間...
    -> 重新採樣完成。 維度: (193, 229, 193)
[*] 正在標記 9 個重要腦區...

[SUCCESS] 成功建立特徵地圖！
  -> 檔案儲存於: output/cnn_rf/NC_vs_AD_top_features_map.nii.gz

[*] 下一步：
    1. 打開影像查看器 (例如 ITK-SNAP 或 3D Slicer)。
    2. 載入 MNI 模板: data/templates/MNI152_T1_1mm_brain.nii.gz
    3. 在模板上疊加 (Overlay): output/cnn_rf/NC_vs_AD_top_features_map.nii.gz
    4. 你現在可以看到模型認為最重要的腦區了！
```

## 重要腦區 (NC vs AD)

根據模型訓練結果，以下腦區對 AD 診斷最重要：

| 排名 | 腦區 | 中文名稱 | 重要性 |
|------|------|----------|--------|
| 1 | Olfactory_L | 左側嗅覺皮層 | ⭐⭐⭐⭐⭐ |
| 2 | OFCant_L | 左側前眶額皮層 | ⭐⭐⭐⭐⭐ |
| 3 | OFCant_R | 右側前眶額皮層 | ⭐⭐⭐⭐ |
| 4 | Cingulate_Post_R | 右側後扣帶回 | ⭐⭐⭐⭐ |
| 5 | ParaHippocampal_R | 右側海馬旁回 | ⭐⭐⭐⭐ |
| 6 | Calcarine_R | 右側距狀裂 | ⭐⭐⭐ |
| 7 | Lingual_R | 右側舌回 | ⭐⭐⭐ |
| 8 | Fusiform_R | 右側梭狀回 | ⭐⭐⭐ |
| 9 | Caudate_L | 左側尾狀核 | ⭐⭐⭐ |

### 臨床意義

1. **嗅覺皮層** - AD 早期常見嗅覺功能下降
2. **眶額皮層** - 與執行功能和決策相關
3. **後扣帶回** - AD 的典型受損區域，與記憶相關
4. **海馬旁回** - 記憶形成的關鍵區域
5. **視覺皮層** (距狀裂、舌回、梭狀回) - AD 後期視覺處理受損
6. **尾狀核** - 與運動控制和認知功能相關

## 整合到現有系統

### 與 Multimodal ROI 系統整合

```python
# 在 scripts/multimodal_roi/train.py 中添加
from scripts.cnn_rf.inference import CNNRF_Predictor

# 使用 CNN-RF 模型作為補充預測
cnn_rf_predictor = CNNRF_Predictor()
cnn_rf_results = cnn_rf_predictor.predict(features)
```

### 與知識圖譜整合

```python
# 將重要腦區添加到知識圖譜
from scripts.cnn_rf.config import TOP_ROIS_NC_VS_AD

# 在 Neo4j 中創建節點
for roi in TOP_ROIS_NC_VS_AD:
    query = """
    MERGE (r:ROI {name: $roi_name})
    SET r.importance = 'high',
        r.source = 'CNN-RF'
    """
    session.run(query, roi_name=roi)
```

## API 接口

### CNNRF_Predictor 類

```python
class CNNRF_Predictor:
    def __init__(self, model_path, atlas_path, atlas_labels_path)
    def predict(self, features, return_proba=True) -> Dict
    def get_feature_importance(self, top_n=30) -> pd.DataFrame
    def extract_important_rois(self, top_n=10) -> List[str]
    def create_brain_map(self, important_rois, output_path) -> str
```

### 預測結果格式

```python
{
    'predictions': [0, 1, 0, ...],  # 數字標籤
    'predicted_labels': ['AD', 'NC', 'AD', ...],  # 文字標籤
    'probabilities': [[0.9, 0.1], [0.2, 0.8], ...],  # 概率
    'detailed': [
        {
            'predicted_class': 'AD',
            'confidence': 0.9,
            'probabilities': {'AD': 0.9, 'NC': 0.1}
        },
        ...
    ]
}
```

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

## 依賴項

```bash
pip install numpy pandas scikit-learn joblib nibabel antspyx matplotlib tqdm
```

## 故障排除

### 問題 1: 找不到 ROI 特徵文件
```
[!] 錯誤: 找不到特徵檔案 data/roi_features.csv
```

**解決方案**: 先運行特徵提取腳本生成 ROI 特徵

### 問題 2: 模型文件不存在
```
[!] 錯誤: 找不到模型檔案 model/cnn_rf/rf_model_NC_vs_AD.joblib
```

**解決方案**: 運行 `python scripts/cnn_rf/train_feat.py` 訓練模型

### 問題 3: AAL3 圖譜路徑錯誤
```
[!] 錯誤: 找不到檔案 data/aal3/AAL3v1_1mm.nii.gz
```

**解決方案**: 確認 AAL3 圖譜文件存在，或更新 `config.py` 中的路徑

## 下一步

1. **模型優化** - 嘗試不同的特徵選擇方法
2. **深度學習** - 使用 3D CNN 端到端訓練
3. **多任務學習** - 同時預測 NC/MCI/AD 和 MMSE 分數
4. **可解釋性** - 使用 SHAP 或 LIME 解釋預測
5. **臨床驗證** - 在獨立數據集上驗證模型

## 參考文獻

1. AAL3 Atlas: Rolls et al. (2020)
2. Random Forest: Breiman (2001)
3. Feature Selection: Guyon & Elisseeff (2003)
