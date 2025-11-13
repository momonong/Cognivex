# 多模態 ROI Pipeline - 快速參考

## 🚀 一分鐘快速開始

```bash
# 1. 測試環境
python scripts/multimodal_roi/quickstart.py

# 2. 效能基準測試 (找到最佳配置)
python scripts/multimodal_roi/benchmark.py

# 3. 訓練模型
python scripts/multimodal_roi/train.py

# 4. 推理
python scripts/multimodal_roi/inference.py
```

## 📋 命令速查表

### 測試和驗證

```bash
# 完整測試
python scripts/multimodal_roi/test_pipeline.py

# 效能基準測試 (推薦)
python scripts/multimodal_roi/benchmark.py

# 測試單個組件
python scripts/multimodal_roi/resnet3d_mini.py
python scripts/multimodal_roi/patch_extractor.py
python scripts/multimodal_roi/dataset.py

# 交互式設置
python scripts/multimodal_roi/quickstart.py
```

### 訓練

```bash
# 完整訓練 (推薦)
python scripts/multimodal_roi/train.py

# 監控訓練
tensorboard --logdir output/multimodal_roi/logs

# 快速測試訓練 (修改 config.py: NUM_EPOCHS=10)
python scripts/multimodal_roi/train.py
```

### 推理

```bash
# 單樣本推理
python scripts/multimodal_roi/inference.py

# 批次推理 (在 inference.py 中修改 subject_list)
python scripts/multimodal_roi/inference.py
```

## 📁 重要文件位置

### 輸入

```
數據: E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/
  ├── NC/
  ├── MCI/
  └── AD/

配置: scripts/multimodal_roi/config.py
```

### 輸出

```
模型: model/multimodal_roi/
  ├── best_feature_extractor.pth
  ├── final_feature_extractor.pth
  └── xgboost_classifier.pkl

結果: output/multimodal_roi/
  ├── training_history.csv
  ├── feature_importance.csv
  └── logs/

緩存: cache/multimodal_roi/
  ├── train/
  ├── val/
  └── test/
```

## ⚙️ 常用配置

### 高效能配置 (推薦 - 針對 24GB VRAM) ⚡

```python
# config.py
BATCH_SIZE = 16  # 4x 速度提升
NUM_WORKERS = 8  # 加速數據加載
torch.backends.cudnn.benchmark = True  # 自動調優
```

### 減少記憶體使用

```python
# config.py
BATCH_SIZE = 2
PATCH_CONFIG["target_patch_size"] = (24, 24, 24)
```

### 加速訓練

```python
# config.py
BATCH_SIZE = 24  # 如果 VRAM 足夠
NUM_EPOCHS = 50
use_cache = True
NUM_WORKERS = 12
```

### 提升準確率

```python
# config.py
RESNET_CONFIG["initial_filters"] = 64
RESNET_CONFIG["feature_dim"] = 128
NUM_EPOCHS = 100
BATCH_SIZE = 12  # 較大模型需要較小 batch
```

## 🐛 常見問題速查

### CUDA out of memory
```python
BATCH_SIZE = 2
PATCH_CONFIG["target_patch_size"] = (24, 24, 24)
```

### 訓練太慢
```python
use_cache = True
NUM_WORKERS = 8
NUM_EPOCHS = 50
```

### 準確率太低
```python
# 檢查數據配準
# 增加模型容量
RESNET_CONFIG["initial_filters"] = 64
# 增加訓練時間
NUM_EPOCHS = 100
```

### 找不到數據
```python
# 修改 config.py
DATA_ROOT = Path("你的數據路徑")
```

## 📊 預期輸出

### 訓練完成後

```
model/multimodal_roi/
├── best_feature_extractor.pth    (最佳模型)
├── final_feature_extractor.pth   (最終模型)
└── xgboost_classifier.pkl        (XGBoost)

output/multimodal_roi/
├── training_history.csv          (訓練歷史)
├── feature_importance.csv        (特徵重要性)
└── logs/                         (TensorBoard)
```

### 推理結果

```
Prediction: AD
Confidence: 87.3%
Probabilities:
  NC:  5.2%
  MCI: 7.5%
  AD:  87.3%

Top 10 ROIs:
  1. Hippocampus_L (T1) - 2.34%
  2. Hippocampus_R (T1) - 1.98%
  ...
```

## 📚 文檔索引

| 文檔 | 用途 |
|------|------|
| `README.md` | 完整使用指南 |
| `IMPLEMENTATION_SUMMARY.md` | 實現總結 |
| `QUICK_REFERENCE.md` | 本文件 |
| `docs/MULTIMODAL_ROI_OPTIMIZATION.md` | 優化方案詳解 |

## 🔗 Python API 速查

### 訓練

```python
from scripts.multimodal_roi import (
    create_dataloaders,
    MultiModalFeatureExtractor,
    FeatureExtractionTrainer
)

# 創建 DataLoaders
dataloaders = create_dataloaders(
    data_root="path/to/data",
    batch_size=4
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

### 推理

```python
from scripts.multimodal_roi import MultiModalROIPredictor

# 初始化
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

# 特徵重要性
analysis = predictor.analyze_feature_importance(
    t1_path, t2_path, dwi_path, top_k=30
)
```

## 💡 最佳實踐

### 1. 數據準備
- ✅ 確保所有影像已配準到 MNI 空間
- ✅ 檢查每個受試者有完整的三種模態
- ✅ 驗證數據質量 (無 NaN, Inf)

### 2. 訓練
- ✅ 先運行快速測試 (NUM_EPOCHS=10)
- ✅ 使用緩存加速後續訓練
- ✅ 監控 TensorBoard 避免過擬合
- ✅ 保存多個 checkpoint

### 3. 推理
- ✅ 使用最佳模型而非最終模型
- ✅ 批次推理提高效率
- ✅ 分析特徵重要性驗證結果

### 4. 優化
- ✅ 根據驗證集調整超參數
- ✅ 使用交叉驗證評估穩定性
- ✅ 分析錯誤案例改進模型

## 🎯 效能基準

| 指標 | 目標值 | 可接受範圍 |
|------|--------|-----------|
| 訓練準確率 | 90-95% | 85-98% |
| 驗證準確率 | 75-85% | 70-90% |
| 測試準確率 | 75-85% | 70-90% |
| 訓練時間 (GPU) | 2-4 hours | 1-6 hours |
| 推理時間 (單樣本) | 10-20 sec | 5-30 sec |
| 過擬合差距 | < 10% | < 15% |

## 📞 獲取幫助

1. **查看文檔**
   - `README.md` - 詳細使用指南
   - `IMPLEMENTATION_SUMMARY.md` - 實現細節

2. **運行測試**
   - `test_pipeline.py` - 診斷問題

3. **檢查日誌**
   - `output/multimodal_roi/logs/` - TensorBoard
   - 終端輸出 - 錯誤訊息

4. **常見問題**
   - 查看 `README.md` 的疑難排解章節

---

**提示**: 將此文件加入書籤以便快速查閱！
