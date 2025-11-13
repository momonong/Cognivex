# Windows 編碼修復說明

## 問題描述

在 Windows 系統上運行時，由於 CMD 使用 cp950 編碼，無法顯示 Unicode emoji 字符（如 ✅ ❌ ⚠️），導致 `UnicodeEncodeError`。

## 已修復

所有 emoji 字符已替換為 ASCII 兼容的標記：

| 原始 Emoji | 替換為 | 含義 |
|-----------|--------|------|
| ✅ | `[OK]` | 成功 |
| ❌ | `[FAIL]` | 失敗 |
| ⚠️ | `[WARN]` | 警告 |
| 🎉 | `[SUCCESS]` | 完成 |

## 修復的文件

- `test_pipeline.py` - 測試腳本
- `quickstart.py` - 快速啟動
- `patch_extractor.py` - Patch 提取器
- `dataset.py` - 數據集
- `train.py` - 訓練腳本
- `inference.py` - 推理腳本
- `resnet3d_mini.py` - 模型定義

## 現在可以運行

```bash
# 測試 Pipeline
python scripts/multimodal_roi/test_pipeline.py

# 快速啟動
python scripts/multimodal_roi/quickstart.py

# 訓練
python scripts/multimodal_roi/train.py
```

所有輸出現在都應該正常顯示，不會再出現編碼錯誤。

## 如果仍有問題

如果仍然遇到編碼問題，可以在腳本開頭添加：

```python
import sys
import io

# 設置 UTF-8 輸出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

或者在 Windows CMD 中設置：

```cmd
chcp 65001
```

這會將 CMD 編碼設置為 UTF-8。
