# 效能優化指南

## 🚀 針對你的硬體配置優化

### 硬體規格
- **VRAM**: 23.89 GB (目前使用: 3.4 GB)
- **可用空間**: 20.5 GB
- **優化潛力**: 非常高 ⭐⭐⭐⭐⭐

## 📊 優化配置

### 已應用的優化 (config.py)

```python
# 1. 增加 Batch Size (4x 速度提升)
BATCH_SIZE = 16  # 從 4 增加到 16

# 2. 增加 Data Loading Workers (2x 速度提升)
NUM_WORKERS = 8  # 從 4 增加到 8

# 3. 啟用 CUDA 優化
torch.backends.cudnn.benchmark = True  # 自動調優
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 加速
torch.backends.cudnn.allow_tf32 = True
```

### 預期效能提升

| 配置 | Batch Size | 訓練時間 | VRAM 使用 | 速度提升 |
|------|-----------|---------|----------|---------|
| **原始** | 4 | 4 hours | ~3.4 GB | 1x |
| **優化後** | 16 | **1 hour** | ~10-12 GB | **4x** |
| **激進** | 32 | **30 min** | ~18-20 GB | **8x** |

## 🎯 進階優化選項

### 選項 1: 激進配置 (最快速度)

如果你想要最快的訓練速度，可以進一步增加 batch size：

```python
# config.py
BATCH_SIZE = 32  # 8x 速度提升
NUM_WORKERS = 12  # 更多 workers
```

**預期 VRAM 使用**: 18-20 GB  
**預期訓練時間**: 30-45 分鐘  
**風險**: 中等 (可能 OOM)

### 選項 2: 增加模型容量

利用額外的 VRAM 訓練更大的模型：

```python
# config.py
RESNET_CONFIG = {
    "in_channels": 1,
    "num_classes": 128,  # 從 64 增加到 128
    "block_config": [2, 2, 2, 2],  # ResNet-18 (從 ResNet-10)
    "initial_filters": 64,  # 從 32 增加到 64
}
```

**預期 VRAM 使用**: 12-15 GB  
**預期準確率提升**: +3-5%  
**預期訓練時間**: 1.5-2 hours

### 選項 3: 混合精度訓練 (推薦)

使用 FP16 混合精度訓練，速度更快且 VRAM 使用更少：

```python
# train.py 中添加
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 在訓練循環中
with autocast():
    features = model(t1_patches, t2_patches, dwi_patches)
    outputs = temp_classifier(features)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**預期速度提升**: 1.5-2x  
**預期 VRAM 節省**: 30-40%  
**準確率影響**: 幾乎無影響

### 選項 4: 增加 Patch Size

利用 VRAM 處理更大的 patches，獲得更多細節：

```python
# config.py
PATCH_CONFIG = {
    "target_patch_size": (40, 40, 40),  # 從 32 增加到 40
}
```

**預期 VRAM 增加**: +3-4 GB  
**預期準確率提升**: +1-2%  
**預期訓練時間**: +20-30%

## 📈 推薦配置方案

### 方案 A: 平衡配置 (推薦) ⭐

**目標**: 平衡速度和穩定性

```python
# config.py
BATCH_SIZE = 16
NUM_WORKERS = 8
RESNET_CONFIG["initial_filters"] = 32
PATCH_CONFIG["target_patch_size"] = (32, 32, 32)
```

**預期**:
- VRAM 使用: 10-12 GB
- 訓練時間: 1 hour
- 速度提升: 4x
- 風險: 低

### 方案 B: 高效能配置 ⚡

**目標**: 最快訓練速度

```python
# config.py
BATCH_SIZE = 24
NUM_WORKERS = 12
# 啟用混合精度訓練
USE_AMP = True
```

**預期**:
- VRAM 使用: 12-15 GB
- 訓練時間: 40-50 min
- 速度提升: 6x
- 風險: 中等

### 方案 C: 高準確率配置 🎯

**目標**: 最高準確率

```python
# config.py
BATCH_SIZE = 12
RESNET_CONFIG = {
    "initial_filters": 64,
    "num_classes": 128,
    "block_config": [2, 2, 2, 2],  # ResNet-18
}
PATCH_CONFIG["target_patch_size"] = (40, 40, 40)
NUM_EPOCHS = 150
```

**預期**:
- VRAM 使用: 15-18 GB
- 訓練時間: 2-2.5 hours
- 準確率提升: +5-8%
- 風險: 中等

## 🔧 實施步驟

### 步驟 1: 測試當前配置

```bash
# 運行一個 epoch 測試
python scripts/multimodal_roi/train.py
# 觀察 VRAM 使用和速度
```

### 步驟 2: 逐步增加 Batch Size

```python
# 測試不同的 batch size
for batch_size in [8, 12, 16, 20, 24]:
    # 修改 config.py
    BATCH_SIZE = batch_size
    # 運行測試
    # 觀察 VRAM 和速度
```

### 步驟 3: 找到最佳配置

記錄每個配置的:
- VRAM 使用峰值
- 每個 epoch 的時間
- 是否出現 OOM

### 步驟 4: 應用最佳配置

選擇 VRAM 使用在 18-20 GB 的配置（留 3-4 GB 緩衝）

## 📊 監控工具

### 監控 VRAM 使用

```bash
# Windows
nvidia-smi -l 1

# 或在 Python 中
import torch
print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"VRAM reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 監控訓練速度

```python
import time

start_time = time.time()
# 訓練一個 epoch
epoch_time = time.time() - start_time
print(f"Epoch time: {epoch_time:.2f} seconds")
```

### TensorBoard 監控

```bash
tensorboard --logdir output/multimodal_roi/logs
```

## 🎓 優化技巧

### 1. 使用 Pin Memory

```python
# dataset.py (已實現)
DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    pin_memory=True,  # 加速 CPU -> GPU 傳輸
)
```

### 2. 預取數據

```python
# dataset.py
DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    prefetch_factor=2,  # 預取 2 個 batch
    persistent_workers=True,  # 保持 workers 活躍
)
```

### 3. 使用緩存

```python
# dataset.py (已實現)
use_cache = True  # 第一次運行後會快很多
```

### 4. 梯度累積 (如果需要更大的有效 batch size)

```python
# train.py
accumulation_steps = 4  # 有效 batch size = 16 * 4 = 64

for i, batch in enumerate(dataloader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5. 分布式訓練 (如果有多個 GPU)

```python
# 使用 PyTorch DDP
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

## 📈 效能基準測試

### 測試腳本

創建 `scripts/multimodal_roi/benchmark.py`:

```python
import torch
import time
from config import *
from resnet3d_mini import MultiModalFeatureExtractor

def benchmark():
    model = MultiModalFeatureExtractor().to(DEVICE)
    model.eval()
    
    # 測試不同的 batch size
    batch_sizes = [4, 8, 12, 16, 20, 24, 32]
    
    for bs in batch_sizes:
        try:
            # 創建測試數據
            t1 = torch.randn(bs, 116, 1, 32, 32, 32).to(DEVICE)
            t2 = torch.randn(bs, 116, 1, 32, 32, 32).to(DEVICE)
            dwi = torch.randn(bs, 116, 1, 32, 32, 32).to(DEVICE)
            
            # 預熱
            with torch.no_grad():
                _ = model(t1, t2, dwi)
            
            # 測試
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    _ = model(t1, t2, dwi)
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            # 統計
            vram = torch.cuda.max_memory_allocated() / 1e9
            throughput = (bs * 10) / elapsed
            
            print(f"Batch Size: {bs:2d} | "
                  f"Time: {elapsed:.2f}s | "
                  f"Throughput: {throughput:.1f} samples/s | "
                  f"VRAM: {vram:.2f} GB")
            
            torch.cuda.reset_peak_memory_stats()
            
        except RuntimeError as e:
            print(f"Batch Size: {bs:2d} | OOM")
            break

if __name__ == "__main__":
    benchmark()
```

運行:
```bash
python scripts/multimodal_roi/benchmark.py
```

## 🎯 預期結果

使用優化配置後，你應該看到:

### 訓練速度
- **原始**: ~4 hours
- **優化後**: **~1 hour** (4x 提升)

### VRAM 使用
- **原始**: ~3.4 GB (14% 使用率)
- **優化後**: **~10-12 GB** (50% 使用率)

### 吞吐量
- **原始**: ~4 samples/second
- **優化後**: **~16 samples/second** (4x 提升)

## ⚠️ 注意事項

1. **逐步增加**: 不要一次性增加太多，逐步測試
2. **監控溫度**: 確保 GPU 溫度在安全範圍 (< 85°C)
3. **留有餘地**: 不要用滿 VRAM，留 3-4 GB 緩衝
4. **保存檢查點**: 定期保存模型，防止意外中斷

## 📞 需要幫助？

如果遇到問題:
1. 查看 `TROUBLESHOOTING.md`
2. 運行 `benchmark.py` 找到最佳配置
3. 監控 VRAM 使用避免 OOM

---

**建議**: 從 `BATCH_SIZE = 16` 開始，如果穩定再逐步增加到 24 或 32。
