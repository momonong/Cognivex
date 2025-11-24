# HuggingFace 模型設定指南

## 概述

CDDA 系統現在支援直接使用 HuggingFace 模型，無需 Ollama。這讓你可以：

1. 使用本地下載的模型
2. 完全控制模型載入和推理
3. 使用 8-bit 量化節省記憶體

## 系統需求

### 必要套件

```bash
pip install transformers torch accelerate
```

### 硬體需求

- **GPU**: 建議使用 NVIDIA GPU (CUDA 支援)
- **記憶體**: 
  - 8-bit 量化: 至少 16GB RAM
  - 完整精度: 至少 32GB RAM
- **儲存空間**: 每個模型約 20-50GB

## 模型下載

### 方法 1: 使用 HuggingFace CLI

```bash
# 安裝 HuggingFace CLI
pip install huggingface-hub

# 登入 (如果模型需要授權)
huggingface-cli login

# 下載模型
huggingface-cli download google/gemma-2-27b-it --local-dir D:/hf_models/medgemma-27b
huggingface-cli download microsoft/phi-3-mini-4k-instruct --local-dir D:/hf_models/gpt-oss-20b
```

### 方法 2: 使用 Python

```python
from huggingface_hub import snapshot_download

# 下載模型
snapshot_download(
    repo_id="google/gemma-2-27b-it",
    local_dir="D:/hf_models/medgemma-27b",
    local_dir_use_symlinks=False
)
```

## 在 CDDA Web 介面中使用

### 1. 啟動 Streamlit 應用

```bash
streamlit run app_cdda.py
```

### 2. 設定模型路徑

在側邊欄中：

1. 選擇 "CDDA Framework (推薦)"
2. 勾選 "啟用 LLM 模式"
3. 輸入模型路徑：
   - **Agent A 模型路徑**: `D:/hf_models/gpt-oss-20b`
   - **Agent B 模型路徑**: `D:/hf_models/medgemma-27b`

### 3. 開始分析

選擇受試者並點擊 "開始分析"

## 推薦模型

### Agent A (Orchestrator)

適合用於工具調用和結構化推理：

- **microsoft/phi-3-mini-4k-instruct** (3.8B) - 輕量級，適合快速推理
- **meta-llama/Llama-3.1-8B-Instruct** (8B) - 平衡性能和速度
- **Qwen/Qwen2.5-14B-Instruct** (14B) - 更強的推理能力

### Agent B (Consultant)

適合用於醫學文本生成：

- **google/gemma-2-9b-it** (9B) - 通用對話模型
- **google/gemma-2-27b-it** (27B) - 更強的醫學推理
- **meta-llama/Llama-3.1-70B-Instruct** (70B) - 最佳性能（需要大量記憶體）

## 記憶體優化

### 使用 8-bit 量化

系統預設啟用 8-bit 量化，可以將記憶體使用量減少約 50%：

```python
agent = CDDAAgent(
    orchestrator_model_path="D:/hf_models/gpt-oss-20b",
    consultant_model_path="D:/hf_models/medgemma-27b",
    use_llm=True,
    load_in_8bit=True  # 啟用 8-bit 量化
)
```

### 使用 4-bit 量化

如果記憶體仍然不足，可以使用 4-bit 量化：

```python
# 修改 app/services/llm_providers/huggingface.py
# 將 load_in_8bit=True 改為 load_in_4bit=True
```

### 清除模型快取

如果需要釋放記憶體：

```python
from app.services.llm_providers import huggingface

huggingface.clear_cache()
```

## 故障排除

### 問題 1: 模型找不到

**錯誤**: `Model not found at: D:/hf_models/...`

**解決方案**:
1. 確認模型路徑正確
2. 檢查目錄中是否有 `config.json` 和 `.safetensors` 檔案
3. 重新下載模型

### 問題 2: CUDA 記憶體不足

**錯誤**: `CUDA out of memory`

**解決方案**:
1. 啟用 8-bit 或 4-bit 量化
2. 使用較小的模型
3. 減少 `max_new_tokens` 參數
4. 關閉其他使用 GPU 的程式

### 問題 3: 生成速度慢

**解決方案**:
1. 確認使用 GPU 而非 CPU
2. 使用較小的模型
3. 減少 `max_new_tokens` 參數

### 問題 4: 模型輸出品質不佳

**解決方案**:
1. 調整 `temperature` 參數 (0.1-0.7)
2. 使用更大的模型
3. 檢查 system prompt 是否適合該模型

## 效能比較

| 模型大小 | 記憶體使用 (8-bit) | 推理速度 | 輸出品質 |
|---------|-------------------|---------|---------|
| 3-4B    | ~4GB              | 快      | 中等    |
| 7-9B    | ~8GB              | 中等    | 良好    |
| 13-14B  | ~14GB             | 較慢    | 優秀    |
| 27B+    | ~27GB             | 慢      | 最佳    |

## 進階設定

### 自訂模型參數

```python
from app.agents.cdda_agent import CDDAAgent

agent = CDDAAgent(
    orchestrator_model="custom-model",
    orchestrator_model_path="/path/to/model",
    consultant_model="custom-model",
    consultant_model_path="/path/to/model",
    use_llm=True,
    load_in_8bit=True,
    verbose=True
)
```

### 使用不同的 dtype

修改 `app/services/llm_providers/huggingface.py`:

```python
response_text = huggingface.handle_text(
    prompt=user_prompt,
    model_path=self.config.model_path,
    torch_dtype="float16",  # 或 "bfloat16"
    ...
)
```

## 參考資源

- [HuggingFace Transformers 文件](https://huggingface.co/docs/transformers)
- [模型量化指南](https://huggingface.co/docs/transformers/main_classes/quantization)
- [CUDA 記憶體管理](https://pytorch.org/docs/stable/notes/cuda.html)
