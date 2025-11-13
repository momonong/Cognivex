# 疑難排解指南

## 常見問題

### 1. 配置信息重複打印

**問題**: 每次 import config.py 時都會打印配置信息

**原因**: config.py 中的 print 語句在模組級別執行

**解決**: 已修復。現在只有直接運行 `python scripts/multimodal_roi/config.py` 時才會顯示配置

---

### 2. 特徵維度不匹配 (22,272 vs 22,104)

**問題**: 顯示總特徵維度為 22,272 而不是預期的 22,104

**原因**: AAL atlas 可能包含 117 個區域（包括背景）而不是 116 個

**計算**:
- 預期: 116 ROIs × 3 modalities × 64 features = 22,104
- 實際: 117 ROIs × 3 modalities × 64 features = 22,464
- 或者: 116 ROIs × 3 modalities × 64 features = 22,272 (如果某處有額外的區域)

**檢查方法**:
```python
from scripts.multimodal_roi.patch_extractor import AAL116PatchExtractor

extractor = AAL116PatchExtractor()
print(f"實際 ROI 數量: {len(extractor.roi_labels)}")
```

**解決方案**:
1. 如果實際是 117 個 ROI，更新 `config.py`:
   ```python
   NUM_ROIS = 117  # 實際的 ROI 數量
   ```

2. 或者在 patch_extractor.py 中排除背景區域（已實現）

---

### 3. CUDA out of memory

**問題**: 
```
RuntimeError: CUDA out of memory
```

**解決方案**:

**方案 A: 減少 batch size**
```python
# config.py
BATCH_SIZE = 2  # 從 4 減少到 2
```

**方案 B: 減少 patch size**
```python
# config.py
PATCH_CONFIG = {
    "target_patch_size": (24, 24, 24),  # 從 32 減少到 24
}
```

**方案 C: 使用 CPU**
```python
# config.py
import torch
DEVICE = torch.device("cpu")
```

**方案 D: 清理 GPU 緩存**
```python
import torch
torch.cuda.empty_cache()
```

---

### 4. 找不到數據

**問題**:
```
[FAIL] 數據目錄不存在
```

**解決方案**:

1. 檢查數據路徑:
   ```python
   # config.py
   DATA_ROOT = Path("你的實際數據路徑")
   ```

2. 確保數據結構正確:
   ```
   DATA_ROOT/
   ├── NC/
   │   ├── sub_001_T1.nii.gz
   │   ├── sub_001_T2_FLAIR.nii.gz
   │   └── sub_001_DWI.nii.gz
   ├── MCI/
   └── AD/
   ```

3. 檢查文件命名:
   - 必須以 `_T1.nii.gz` 結尾
   - 必須以 `_T2_FLAIR.nii.gz` 結尾
   - 必須以 `_DWI.nii.gz` 結尾

---

### 5. AAL Atlas 下載失敗

**問題**:
```
[WARN] Could not load AAL atlas from nilearn
```

**原因**: 
- 沒有網路連接
- nilearn 無法訪問下載服務器

**解決方案**:

**方案 A: 手動下載**
1. 下載 AAL atlas: https://www.gin.cnrs.fr/en/tools/aal/
2. 放置到 `data/aal3/` 目錄
3. 修改 `patch_extractor.py` 使用本地文件

**方案 B: 使用代理**
```python
import os
os.environ['http_proxy'] = 'http://proxy:port'
os.environ['https_proxy'] = 'http://proxy:port'
```

**方案 C: 離線模式**
```python
# 使用已下載的 atlas
from nilearn import datasets
datasets.fetch_atlas_aal(data_dir='./data/nilearn_data')
```

---

### 6. 訓練太慢

**問題**: 訓練時間過長

**解決方案**:

**方案 A: 使用緩存**
```python
# dataset.py
use_cache = True  # 第一次運行後會快很多
```

**方案 B: 增加 workers**
```python
# config.py
NUM_WORKERS = 8  # 根據 CPU 核心數調整
```

**方案 C: 減少 epochs**
```python
# config.py
NUM_EPOCHS = 50  # 從 100 減少到 50
```

**方案 D: 使用更少的 ROI**
```python
# config.py
NUM_ROIS = 50  # 只使用最重要的 50 個 ROI
```

---

### 7. 準確率太低

**問題**: 驗證準確率低於 60%

**可能原因**:
1. 數據未正確配準
2. 樣本數量太少
3. 類別嚴重不平衡
4. 模型容量不足

**診斷步驟**:

1. **檢查數據質量**:
   ```python
   python scripts/multimodal_roi/patch_extractor.py
   ```

2. **檢查類別分布**:
   ```python
   from scripts.multimodal_roi.dataset import MultiModalROIDataset
   dataset = MultiModalROIDataset(data_root=DATA_ROOT)
   print(f"NC: {sum(1 for s in dataset.subjects if s['label'] == 0)}")
   print(f"MCI: {sum(1 for s in dataset.subjects if s['label'] == 1)}")
   print(f"AD: {sum(1 for s in dataset.subjects if s['label'] == 2)}")
   ```

3. **檢查過擬合**:
   - 如果訓練準確率 > 90% 但驗證準確率 < 70%，說明過擬合
   - 解決: 增加正則化、減少模型容量、使用數據增強

**解決方案**:

**方案 A: 增加模型容量**
```python
# config.py
RESNET_CONFIG = {
    "initial_filters": 64,  # 從 32 增加到 64
    "feature_dim": 128,     # 從 64 增加到 128
}
```

**方案 B: 使用類別權重**
```python
# 在訓練時自動使用
class_weights = dataset.get_class_weights()
```

**方案 C: 增加訓練時間**
```python
# config.py
NUM_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
```

---

### 8. Windows 編碼錯誤

**問題**:
```
UnicodeEncodeError: 'cp950' codec can't encode character
```

**解決**: 已修復。所有 emoji 已替換為 ASCII 字符

如果仍有問題:
```cmd
chcp 65001
```

---

### 9. Import 錯誤

**問題**:
```
ModuleNotFoundError: No module named 'scripts.multimodal_roi'
```

**解決方案**:

**方案 A: 添加到 Python path**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

**方案 B: 使用相對 import**
```python
from .config import *
from .resnet3d_mini import MultiModalFeatureExtractor
```

**方案 C: 從正確的目錄運行**
```bash
# 從專案根目錄運行
cd D:/projects/Cognivex
python scripts/multimodal_roi/train.py
```

---

### 10. 緩存問題

**問題**: 修改代碼後仍使用舊的緩存數據

**解決方案**:

**清理緩存**:
```bash
# Windows
rmdir /s /q cache\multimodal_roi

# Linux/Mac
rm -rf cache/multimodal_roi
```

**或者禁用緩存**:
```python
# dataset.py
dataset = MultiModalROIDataset(
    data_root=DATA_ROOT,
    use_cache=False  # 禁用緩存
)
```

---

## 診斷工具

### 系統健康檢查

```python
# 運行完整測試
python scripts/multimodal_roi/test_pipeline.py
```

### 檢查配置

```python
# 顯示當前配置
python scripts/multimodal_roi/config.py
```

### 檢查數據

```python
from scripts.multimodal_roi.dataset import MultiModalROIDataset
from scripts.multimodal_roi.config import DATA_ROOT

dataset = MultiModalROIDataset(data_root=DATA_ROOT, use_cache=False)
print(f"Total subjects: {len(dataset)}")

if len(dataset) > 0:
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Patches shape: {sample['patches']['T1'].shape}")
```

### 檢查模型

```python
from scripts.multimodal_roi.resnet3d_mini import MultiModalFeatureExtractor
import torch

model = MultiModalFeatureExtractor()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 測試前向傳播
t1 = torch.randn(1, 116, 1, 32, 32, 32)
t2 = torch.randn(1, 116, 1, 32, 32, 32)
dwi = torch.randn(1, 116, 1, 32, 32, 32)

features = model(t1, t2, dwi)
print(f"Output shape: {features.shape}")
```

---

## 獲取幫助

如果以上方法都無法解決問題:

1. 查看詳細日誌:
   - `output/multimodal_roi/logs/` - TensorBoard 日誌
   - 終端輸出 - 錯誤訊息

2. 運行診斷:
   ```bash
   python scripts/multimodal_roi/test_pipeline.py
   ```

3. 檢查文檔:
   - `README.md` - 完整使用指南
   - `IMPLEMENTATION_SUMMARY.md` - 技術細節
   - `QUICK_REFERENCE.md` - 快速參考

4. 查看範例:
   - 每個腳本都有 `if __name__ == "__main__"` 區塊
   - 可以直接運行查看範例用法
