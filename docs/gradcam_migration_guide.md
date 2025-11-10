# Grad-CAM Generator 遷移指南

## 概述

原本的 `app/core/cnn_3d/xai.py` 腳本已被重構為可重用的 `GradCAMGenerator` 類別。本指南說明如何從舊的腳本遷移到新的類別。

## 主要改進

### 舊版本 (xai.py 腳本)
- ❌ 單一腳本，難以測試和重用
- ❌ 硬編碼的參數
- ❌ 只能批次處理
- ❌ 缺乏靈活性

### 新版本 (GradCAMGenerator 類別)
- ✅ 模組化設計，易於測試
- ✅ 可配置的參數
- ✅ 支援單一和批次處理
- ✅ 提供多種聚合方法
- ✅ 完整的錯誤處理
- ✅ 統計資訊輸出

## 快速開始

### 1. 匯入類別

```python
from app.core.xai import GradCAMGenerator
```

### 2. 載入模型

```python
import torch
from app.core.cnn_3d.model import Simple3DCNN_InstanceNorm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 載入集成模型
models = []
for i in range(5):
    model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
    model.load_state_dict(torch.load(f"model/cnn_3d/cnn_3d_fold_{i+1}.pth"))
    model.to(device)
    model.eval()
    models.append(model)
```

### 3. 建立 Generator

```python
generator = GradCAMGenerator(
    models=models,
    device=device,
    target_layer_name="block4"
)
```

### 4. 生成 Grad-CAM

```python
# 準備輸入 (從 NIfTI 載入)
input_tensor = ...  # shape: (1, 1, 128, 128, 128)

# 生成集成 Grad-CAM
heatmap = generator.generate_ensemble(
    input_tensor=input_tensor,
    target_class=1,  # 0=NC, 1=AD
    threshold_percentile=95.0,
    aggregation_method="mean"
)
```

### 5. 儲存結果

```python
import numpy as np

# 準備 affine 矩陣 (從原始 NIfTI 取得)
affine = ...  # 4x4 numpy array

# 儲存為 NIfTI
output_path = generator.save_as_nifti(
    heatmap=heatmap,
    affine=affine,
    output_path="output/cnn_3d/xai_heatmaps/",
    subject_id="sub-01",
    target_class="AD"
)
```

## API 參考

### GradCAMGenerator 類別

#### 初始化

```python
GradCAMGenerator(
    models: List[nn.Module],
    device: torch.device,
    target_layer_name: str = "block4"
)
```

**參數:**
- `models`: 模型列表 (用於集成)
- `device`: PyTorch 裝置
- `target_layer_name`: 目標層名稱

#### 主要方法

##### generate_single_model()

為單一模型生成 Grad-CAM。

```python
heatmap = generator.generate_single_model(
    model=model,
    input_tensor=input_tensor,
    target_class=1
)
```

**返回:** 3D numpy array, shape (H, W, D)

##### generate_ensemble()

生成集成 Grad-CAM (平均多個模型)。

```python
heatmap = generator.generate_ensemble(
    input_tensor=input_tensor,
    target_class=1,
    threshold_percentile=95.0,
    aggregation_method="mean"  # "mean", "max", "weighted"
)
```

**參數:**
- `input_tensor`: 輸入張量, shape (1, 1, H, W, D)
- `target_class`: 目標類別索引 (0=NC, 1=AD)
- `threshold_percentile`: 閾值百分位數 (0-100)
- `aggregation_method`: 聚合方法 ("mean", "max", "weighted")

**返回:** 3D numpy array, 已標準化和閾值處理

##### save_as_nifti()

儲存熱圖為 NIfTI 格式。

```python
output_path = generator.save_as_nifti(
    heatmap=heatmap,
    affine=affine,
    output_path="output/dir/",
    subject_id="sub-01",
    target_class="AD"
)
```

**返回:** 儲存的檔案完整路徑

##### upsample_to_original()

將熱圖上採樣到原始解析度。

```python
upsampled = generator.upsample_to_original(
    heatmap=heatmap,
    target_shape=(256, 256, 256),
    order=1  # 0=最近鄰, 1=線性, 3=三次
)
```

##### get_statistics()

計算熱圖的統計資訊。

```python
stats = generator.get_statistics(heatmap)
# 返回: {
#     "min": float,
#     "max": float,
#     "mean": float,
#     "std": float,
#     "non_zero_count": int,
#     "total_voxels": int,
#     "non_zero_percentage": float,
#     "non_zero_mean": float,
#     "non_zero_std": float
# }
```

## 遷移範例

### 舊版本 (xai.py)

```python
# 舊的方式 - 只能透過腳本執行
# python app/core/cnn_3d/xai.py
# 需要設定 .env 檔案
```

### 新版本 (GradCAMGenerator)

```python
from app.core.xai import GradCAMGenerator
import torch

# 更靈活的使用方式
generator = GradCAMGenerator(models, device)

# 可以在程式中直接呼叫
heatmap = generator.generate_ensemble(
    input_tensor=input_tensor,
    target_class=1,
    threshold_percentile=95.0
)

# 可以取得統計資訊
stats = generator.get_statistics(heatmap)
print(f"非零體素: {stats['non_zero_count']}")

# 可以儲存到任意位置
generator.save_as_nifti(heatmap, affine, "custom/path/")
```

## 進階使用

### 自訂聚合方法

```python
# 使用最大值聚合
heatmap_max = generator.generate_ensemble(
    input_tensor=input_tensor,
    target_class=1,
    aggregation_method="max"
)

# 使用加權平均 (目前使用均等權重)
heatmap_weighted = generator.generate_ensemble(
    input_tensor=input_tensor,
    target_class=1,
    aggregation_method="weighted"
)
```

### 不同的閾值設定

```python
# 保留 top 10% 的訊號
heatmap_strict = generator.generate_ensemble(
    input_tensor=input_tensor,
    target_class=1,
    threshold_percentile=90.0
)

# 保留 top 1% 的訊號
heatmap_very_strict = generator.generate_ensemble(
    input_tensor=input_tensor,
    target_class=1,
    threshold_percentile=99.0
)
```

### 批次處理

```python
import glob
from tqdm import tqdm

# 掃描所有 NIfTI 檔案
nifti_files = glob.glob("data/**/*.nii.gz", recursive=True)

# 批次處理
for nifti_path in tqdm(nifti_files):
    # 載入和預處理 (使用你的資料載入函式)
    input_tensor, affine = load_and_preprocess(nifti_path)
    
    # 生成 Grad-CAM
    heatmap = generator.generate_ensemble(
        input_tensor=input_tensor,
        target_class=1,
        threshold_percentile=95.0
    )
    
    # 儲存
    subject_id = os.path.basename(nifti_path).split('.')[0]
    generator.save_as_nifti(
        heatmap=heatmap,
        affine=affine,
        output_path="output/heatmaps/",
        subject_id=subject_id,
        target_class="AD"
    )
```

## 常見問題

### Q: 舊的 xai.py 腳本還能用嗎？

A: 可以，舊腳本仍然保留在 `app/core/cnn_3d/xai.py`，但建議遷移到新的類別以獲得更好的靈活性和可維護性。

### Q: 如何選擇聚合方法？

A: 
- `mean`: 適合大多數情況，提供穩定的平均結果
- `max`: 強調最強的激活區域
- `weighted`: 可以根據模型準確度調整權重 (目前使用均等權重)

### Q: 閾值百分位數如何選擇？

A: 
- 95.0: 保留 top 5% 的訊號 (預設，適合大多數情況)
- 90.0: 保留 top 10% 的訊號 (顯示更多區域)
- 99.0: 保留 top 1% 的訊號 (只顯示最強的區域)

### Q: 如何處理記憶體不足的問題？

A: 
1. 使用 `torch.no_grad()` (已內建)
2. 一次只處理一個模型
3. 及時釋放不需要的張量
4. 降低批次大小

## 相關文件

- [設計文件](../design.md)
- [需求文件](../requirements.md)
- [範例程式](../../examples/gradcam_generator_example.py)

## 支援

如有問題，請參考:
1. 範例程式: `examples/gradcam_generator_example.py`
2. 單元測試: `tests/test_gradcam_generator.py` (待建立)
3. API 文件: 類別和方法的 docstrings
