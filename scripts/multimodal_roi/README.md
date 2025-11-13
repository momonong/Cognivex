# Multi-modal ROI Feature Extraction Pipeline

完整的多模態 3D ROI 特徵提取與分類 Pipeline，基於 AAL-116 圖譜和 3D ResNet-10 Mini-CNNs。

## 🎯 核心特性

### 架構設計

```
多模態 MRI 影像 (T1, T2-FLAIR, DWI)
    ↓
AAL-116 ROI Masks (116 個腦區)
    ↓
3D Patch 提取 (116 patches × 3 modalities)
    ↓
3 個獨立的 3D ResNet-10 Mini-CNNs
    ├─ Mini-CNN_T1    → 116 × 64 features
    ├─ Mini-CNN_FLAIR → 116 × 64 features
    └─ Mini-CNN_DWI   → 116 × 64 features
    ↓
特徵串接 (Concatenation)
    ↓
22,104 維特徵向量 (116 × 3 × 64)
    ↓
XGBoost 分類器 (NC vs MCI vs AD)
    ↓
預測結果 + 特徵重要性分析
```

### 關鍵優勢

1. **高解釋性**
   - XGBoost 提供特徵重要性排名
   - 可追溯到具體的 (ROI, 模態, 特徵索引) 組合
   - 符合臨床需求的可解釋 AI

2. **多模態融合**
   - 特徵層融合策略
   - 獨立學習每個模態的特徵
   - 保留模態間的互補信息

3. **抗過擬合**
   - XGBoost 的正則化機制
   - 對無關特徵具有穩健性
   - 適合 N < p 的高維數據

4. **可擴展性**
   - 模組化設計
   - 易於添加新模態
   - 支持不同的圖譜 (AAL, DK, etc.)

## 📁 文件結構

```
scripts/multimodal_roi/
├── config.py              # 配置文件
├── resnet3d_mini.py       # 3D ResNet-10 Mini-CNN 模型
├── patch_extractor.py     # AAL-116 ROI Patch 提取器
├── dataset.py             # PyTorch Dataset 和 DataLoader
├── train.py               # 訓練 Pipeline
├── inference.py           # 推理 Pipeline
└── README.md              # 本文件

model/multimodal_roi/      # 訓練好的模型
├── best_feature_extractor.pth    # 最佳特徵提取器
├── final_feature_extractor.pth   # 最終特徵提取器
└── xgboost_classifier.pkl        # XGBoost 分類器

output/multimodal_roi/     # 訓練輸出
├── training_history.csv          # 訓練歷史
├── feature_importance.csv        # 特徵重要性
└── logs/                         # TensorBoard 日誌

cache/multimodal_roi/      # 緩存的 Patches
├── train/                        # 訓練集緩存
├── val/                          # 驗證集緩存
└── test/                         # 測試集緩存
```

## 🚀 快速開始

### 1. 環境準備

```bash
# 安裝依賴
pip install torch torchvision nibabel nilearn scikit-learn xgboost pandas tqdm tensorboard

# 或使用 Poetry
poetry install
```

### 2. 數據準備

確保數據結構如下：

```
E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/
├── NC/
│   ├── sub_001_T1.nii.gz
│   ├── sub_001_T2_FLAIR.nii.gz
│   ├── sub_001_DWI.nii.gz
│   └── ...
├── MCI/
│   ├── sub_101_T1.nii.gz
│   ├── sub_101_T2_FLAIR.nii.gz
│   ├── sub_101_DWI.nii.gz
│   └── ...
└── AD/
    ├── sub_201_T1.nii.gz
    ├── sub_201_T2_FLAIR.nii.gz
    ├── sub_201_DWI.nii.gz
    └── ...
```

**重要**：所有影像必須已經配準到 MNI 空間！

### 3. 測試組件

```bash
# 測試 3D ResNet-10 Mini-CNN
python scripts/multimodal_roi/resnet3d_mini.py

# 測試 Patch 提取器
python scripts/multimodal_roi/patch_extractor.py

# 測試 Dataset
python scripts/multimodal_roi/dataset.py
```

### 4. 訓練模型

```bash
# 完整訓練 Pipeline
python scripts/multimodal_roi/train.py
```

訓練過程：
1. **階段 1**: 訓練 3 個 Mini-CNNs 學習有意義的特徵 (約 50-100 epochs)
2. **階段 2**: 提取所有受試者的特徵向量
3. **階段 3**: 訓練 XGBoost 分類器

預期訓練時間：
- GPU (RTX 3080): 約 2-4 小時
- CPU: 約 8-12 小時

### 5. 推理和分析

```bash
# 單個受試者預測
python scripts/multimodal_roi/inference.py
```

## 📊 配置說明

### 修改配置 (`config.py`)

```python
# 數據路徑
DATA_ROOT = Path("你的數據路徑")

# 模型配置
RESNET_CONFIG = {
    "initial_filters": 32,  # 增加以提升容量
    "feature_dim": 64,      # 每個 ROI 的特徵維度
}

# XGBoost 配置
XGBOOST_CONFIG = {
    "n_estimators": 500,    # 樹的數量
    "max_depth": 6,         # 最大深度
    "learning_rate": 0.05,  # 學習率
}

# 訓練配置
BATCH_SIZE = 4              # 根據 GPU 記憶體調整
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
```

## 🔬 使用範例

### 範例 1: 訓練模型

```python
from config import *
from resnet3d_mini import MultiModalFeatureExtractor
from dataset import create_dataloaders
from train import FeatureExtractionTrainer

# 創建 DataLoaders
dataloaders = create_dataloaders(
    data_root=DATA_ROOT,
    batch_size=4,
    num_workers=4
)

# 創建模型
model = MultiModalFeatureExtractor(
    feature_dim=64,
    initial_filters=32
)

# 訓練
trainer = FeatureExtractionTrainer(
    model=model,
    dataloaders=dataloaders,
    device='cuda'
)

trainer.train()
```

### 範例 2: 推理

```python
from inference import MultiModalROIPredictor

# 初始化預測器
predictor = MultiModalROIPredictor(
    feature_extractor_path='model/multimodal_roi/best_feature_extractor.pth',
    xgboost_path='model/multimodal_roi/xgboost_classifier.pkl',
    device='cuda'
)

# 預測
result = predictor.predict(
    t1_path='path/to/T1.nii.gz',
    t2_path='path/to/T2_FLAIR.nii.gz',
    dwi_path='path/to/DWI.nii.gz'
)

print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### 範例 3: 特徵重要性分析

```python
# 分析哪些 ROI 和模態最重要
analysis = predictor.analyze_feature_importance(
    t1_path='path/to/T1.nii.gz',
    t2_path='path/to/T2_FLAIR.nii.gz',
    dwi_path='path/to/DWI.nii.gz',
    top_k=30
)

# 查看最重要的 ROI
print(analysis['roi_importance'].head(10))

# 輸出範例：
#   roi_idx  roi_name              modality  contribution
#   37       Hippocampus_L         T1        0.0234
#   38       Hippocampus_R         T1        0.0198
#   39       ParaHippocampal_L     T2_FLAIR  0.0187
#   ...
```

### 範例 4: 批次預測

```python
# 準備受試者列表
subjects = [
    {
        'subject_id': 'sub_001',
        't1_path': 'path/to/sub_001_T1.nii.gz',
        't2_path': 'path/to/sub_001_T2_FLAIR.nii.gz',
        'dwi_path': 'path/to/sub_001_DWI.nii.gz'
    },
    # ... 更多受試者
]

# 批次預測
results_df = predictor.batch_predict(
    subject_list=subjects,
    output_path='output/batch_predictions.csv'
)

print(results_df)
```

## 📈 預期結果

### 效能指標

基於文獻和類似研究，預期效能：

| 指標 | 預期值 |
|------|--------|
| 交叉驗證準確率 | 75-85% |
| NC vs AD 準確率 | 85-90% |
| 三分類準確率 (NC/MCI/AD) | 70-80% |
| ROC-AUC | 0.80-0.90 |

### 特徵重要性

預期最重要的 ROI：

1. **海馬迴 (Hippocampus)** - AD 最經典的生物標記
2. **內嗅皮質 (Entorhinal Cortex)** - 早期受損區域
3. **後扣帶迴 (Posterior Cingulate)** - DMN 核心區域
4. **顳葉 (Temporal Lobe)** - 記憶相關區域
5. **頂葉 (Parietal Lobe)** - 認知功能區域

### 模態貢獻

預期模態重要性：

- **T1**: 40-50% (結構萎縮)
- **T2-FLAIR**: 25-35% (白質病變)
- **DWI**: 20-30% (微結構變化)

## 🔧 優化建議

### 1. 提升準確率

```python
# 增加模型容量
RESNET_CONFIG = {
    "initial_filters": 64,  # 從 32 增加到 64
    "feature_dim": 128,     # 從 64 增加到 128
}

# 使用更深的網絡
block_config = [2, 2, 2, 2]  # ResNet-18 instead of ResNet-10
```

### 2. 減少過擬合

```python
# 增加正則化
XGBOOST_CONFIG = {
    "reg_alpha": 0.5,      # L1 正則化
    "reg_lambda": 2.0,     # L2 正則化
    "gamma": 0.5,          # 最小損失減少
}

# 使用數據增強
def augment_patches(patches):
    # 隨機翻轉
    # 隨機旋轉
    # 隨機縮放
    return augmented_patches
```

### 3. 加速訓練

```python
# 使用緩存
use_cache = True  # 第一次運行後會快很多

# 增加 batch size (如果 GPU 記憶體足夠)
BATCH_SIZE = 8

# 使用混合精度訓練
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

## 🐛 常見問題

### Q1: CUDA out of memory

```python
# 解決方案 1: 減少 batch size
BATCH_SIZE = 2

# 解決方案 2: 減少 patch size
PATCH_CONFIG = {
    "target_patch_size": (24, 24, 24),  # 從 32 減少到 24
}

# 解決方案 3: 使用 CPU
DEVICE = torch.device("cpu")
```

### Q2: 訓練太慢

```python
# 解決方案 1: 使用緩存
use_cache = True

# 解決方案 2: 減少 epochs
NUM_EPOCHS = 50

# 解決方案 3: 使用更少的 ROI
# 只使用最重要的 50 個 ROI
```

### Q3: 準確率太低

可能原因：
1. 數據未正確配準到 MNI 空間
2. 樣本數量太少
3. 類別嚴重不平衡
4. 模型容量不足

解決方案：
```python
# 檢查數據配準
python scripts/multimodal_roi/patch_extractor.py

# 使用類別權重
class_weights = dataset.get_class_weights()

# 增加模型容量
RESNET_CONFIG["initial_filters"] = 64
```

## 📚 參考文獻

1. **3D ResNet for Medical Imaging**
   - Tran et al., "A Closer Look at Spatiotemporal Convolutions for Action Recognition", CVPR 2018

2. **Multi-modal Fusion for AD Classification**
   - Liu et al., "Multi-Modality Cascaded Convolutional Neural Networks for Alzheimer's Disease Diagnosis", Neuroinformatics 2018

3. **XGBoost for High-dimensional Data**
   - Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016

4. **AAL Atlas**
   - Tzourio-Mazoyer et al., "Automated Anatomical Labeling of Activations in SPM Using a Macroscopic Anatomical Parcellation of the MNI MRI Single-Subject Brain", NeuroImage 2002

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

請參考主專案的 LICENSE 文件。

---

**注意**: 這是一個研究用途的實現，不應用於臨床診斷。所有預測結果僅供參考。
