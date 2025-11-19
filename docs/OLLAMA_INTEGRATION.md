# Ollama 本地端 LLM 整合文檔

## 概述

Ollama 已完全整合到 Cognivex 系統中，提供**本地端、隱私保護**的 LLM 推理能力。

### 為什麼使用 Ollama？

| 特性 | Ollama (本地) | AWS Bedrock (雲端) |
|------|---------------|-------------------|
| **資料隱私** | ✅ 完全本地，不上傳 | ❌ 數據傳送到雲端 |
| **成本** | ✅ 免費 | ❌ 按使用量計費 |
| **網路需求** | ✅ 離線可用 | ❌ 需要網路連接 |
| **速度** | ⚡ 取決於硬體 | 🌐 取決於網路 |
| **模型選擇** | 🔧 可自訂 | 📦 固定選項 |
| **醫療合規** | ✅ HIPAA 友好 | ⚠️ 需要額外配置 |

## 快速開始

### 1. 安裝 Ollama

```bash
# 訪問 https://ollama.ai 下載安裝

# 或使用命令行 (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# 驗證安裝
ollama --version
```

### 2. 安裝 Python 套件

```bash
pip install ollama
```

### 3. 運行設置腳本

```bash
python app/setup_ollama.py
```

這個腳本會：
- ✅ 檢查 Ollama 安裝狀態
- ✅ 列出可用模型
- ✅ 安裝推薦模型
- ✅ 測試推理功能
- ✅ 配置隱私模式

### 4. 下載推薦模型

```bash
# 小型模型 (4GB RAM, 快速)
ollama pull llama3.2:3b

# 中型模型 (8GB RAM, 平衡)
ollama pull llama3.1:8b

# 醫療專用模型 (8GB RAM)
ollama pull meditron:7b

# 視覺模型 (8GB RAM, 圖像分析)
ollama pull llava:7b
```

## 使用方法

### 方法 1: 使用默認提供者

```python
from app.services.llm_providers import llm_response

# 自動使用配置的默認提供者
response = llm_response("What is Alzheimer's disease?")
```

### 方法 2: 明確指定 Ollama

```python
from app.services.llm_providers import llm_response

response = llm_response(
    "What is Alzheimer's disease?",
    llm_provider="ollama",
    model="llama3.2:3b"
)
```

### 方法 3: 使用配置對象

```python
from app.services.llm_providers import llm_response
from app.services.llm_providers.config import get_config_by_name

# 使用預定義配置
config = get_config_by_name("ollama_large")
response = llm_response("What is Alzheimer's disease?", config=config)
```

### 方法 4: 啟用隱私模式

```python
import os

# 設置環境變量
os.environ['PRIVACY_MODE'] = 'true'

# 所有 LLM 請求將自動使用 Ollama
from app.services.llm_providers import llm_response
response = llm_response("What is Alzheimer's disease?")
```

### 方法 5: 運行時切換

```python
from app.services.llm_providers import use_local_llm, use_cloud_llm, llm_response

# 切換到本地
use_local_llm()
response = llm_response("What is Alzheimer's disease?")

# 切換回雲端
use_cloud_llm()
response = llm_response("What is Alzheimer's disease?")
```

## 推薦模型

### 文字生成模型

| 模型 | 大小 | RAM 需求 | 速度 | 適用場景 |
|------|------|----------|------|----------|
| **llama3.2:3b** | 2GB | 4GB | 快 | 測試、快速推理 |
| **llama3.1:8b** | 4.7GB | 8GB | 中 | 生產環境、平衡 |
| **llama3.1:70b** | 40GB | 64GB | 慢 | 最佳質量 |
| **meditron:7b** | 4GB | 8GB | 中 | 醫療專用 |

### 視覺模型

| 模型 | 大小 | RAM 需求 | 適用場景 |
|------|------|----------|----------|
| **llava:7b** | 4.5GB | 8GB | MRI 影像分析 |
| **bakllava:7b** | 4.5GB | 8GB | 醫學影像理解 |

## 配置選項

### 環境變量

```bash
# 啟用隱私模式 (強制使用 Ollama)
export PRIVACY_MODE=true

# 設置默認 LLM 提供者
export LLM_PROVIDER=ollama

# 設置 Ollama 服務器地址
export OLLAMA_BASE_URL=http://localhost:11434
```

### 配置文件

```python
# app/services/llm_providers/config.py

from app.services.llm_providers.config import LLMConfig

# 自定義配置
custom_config = LLMConfig(
    provider="ollama",
    model="llama3.1:8b",
    temperature=0.1,
    ollama_base_url="http://localhost:11434"
)
```

## 在 Workflow 中使用

### 更新報告生成器

```python
# app/agents/report_generator.py

from app.services.llm_providers import llm_response
from app.services.llm_providers.config import get_config_by_name

def generate_final_report(state: AgentState) -> dict:
    # 使用 Ollama 生成報告 (隱私保護)
    config = get_config_by_name("ollama_medical")
    
    report = llm_response(
        prompt=f"Generate medical report for: {state['classification_result']}",
        config=config
    )
    
    return {"generated_reports": {"medical": report}}
```

### CNN-RF 推理中使用

```python
# app/agents/cnn_rf_inference.py

# 在報告生成時使用本地 LLM
import os
os.environ['PRIVACY_MODE'] = 'true'

# 所有後續的 LLM 調用將使用 Ollama
```

## 性能優化

### 1. 選擇合適的模型

```python
# 快速推理 (測試、開發)
config = get_config_by_name("ollama")  # llama3.2:3b

# 生產環境 (平衡)
config = get_config_by_name("ollama_large")  # llama3.1:8b

# 醫療專用
config = get_config_by_name("ollama_medical")  # meditron:7b
```

### 2. 調整溫度參數

```python
# 更確定性的輸出 (醫療報告)
response = llm_response(
    prompt="Generate diagnosis",
    llm_provider="ollama",
    temperature=0.1  # 低溫度 = 更確定
)

# 更創造性的輸出 (解釋說明)
response = llm_response(
    prompt="Explain the condition",
    llm_provider="ollama",
    temperature=0.7  # 高溫度 = 更創造
)
```

### 3. 批量處理

```python
# 批量生成報告
prompts = [
    "Generate report for patient 1",
    "Generate report for patient 2",
    "Generate report for patient 3"
]

responses = [
    llm_response(prompt, llm_provider="ollama")
    for prompt in prompts
]
```

## 故障排除

### 問題 1: Ollama 服務器未運行

```
[ERROR] Ollama server is not running
```

**解決方案**:
```bash
# 啟動 Ollama 服務器
ollama serve

# 或在 Windows/Mac 上，Ollama 應該自動運行
# 檢查系統托盤圖標
```

### 問題 2: 模型未安裝

```
[ERROR] Model not found: llama3.2:3b
```

**解決方案**:
```bash
# 下載模型
ollama pull llama3.2:3b

# 列出已安裝的模型
ollama list
```

### 問題 3: 內存不足

```
[ERROR] Out of memory
```

**解決方案**:
- 使用更小的模型 (llama3.2:3b 而不是 llama3.1:70b)
- 關閉其他應用程序
- 增加系統 RAM

### 問題 4: Python 套件未安裝

```
[WARNING] ollama not installed
```

**解決方案**:
```bash
pip install ollama
```

## 隱私和安全

### HIPAA 合規性

使用 Ollama 的優勢：
- ✅ **數據不離開本地** - 所有處理在本地完成
- ✅ **無第三方訪問** - 不經過雲端服務
- ✅ **完全控制** - 你控制所有數據和模型
- ✅ **審計追蹤** - 本地日誌記錄

### 最佳實踐

1. **啟用隱私模式**
   ```bash
   export PRIVACY_MODE=true
   ```

2. **使用本地模型**
   ```python
   # 確保使用 Ollama
   config = get_config_by_name("ollama_medical")
   ```

3. **禁用雲端回退**
   ```python
   # 在 app/services/llm_providers/__init__.py 中
   # 移除 Bedrock 回退邏輯
   ```

4. **記錄所有 LLM 調用**
   ```python
   import logging
   logging.info(f"LLM call: provider={config.provider}, model={config.model}")
   ```

## 性能基準

### 推理速度 (tokens/秒)

| 模型 | CPU (16核) | GPU (RTX 3090) |
|------|-----------|----------------|
| llama3.2:3b | 20-30 | 80-100 |
| llama3.1:8b | 10-15 | 40-60 |
| llama3.1:70b | 2-5 | 15-25 |

### 內存使用

| 模型 | 最小 RAM | 推薦 RAM |
|------|----------|----------|
| llama3.2:3b | 4GB | 8GB |
| llama3.1:8b | 8GB | 16GB |
| llama3.1:70b | 64GB | 128GB |

## 與 AWS Bedrock 對比

### 何時使用 Ollama

- ✅ 處理敏感醫療數據
- ✅ 需要離線運行
- ✅ 希望降低成本
- ✅ 需要完全控制

### 何時使用 AWS Bedrock

- ✅ 需要最新的模型
- ✅ 不想管理基礎設施
- ✅ 需要全球擴展
- ✅ 數據隱私不是主要考量

## 下一步

1. **運行設置腳本**
   ```bash
   python app/setup_ollama.py
   ```

2. **測試推理**
   ```bash
   python -c "from app.services.llm_providers import llm_response; print(llm_response('Hello', llm_provider='ollama'))"
   ```

3. **更新應用配置**
   ```bash
   export PRIVACY_MODE=true
   ```

4. **運行完整測試**
   ```bash
   python app/test_cnn_rf_integration.py --mode single
   ```

## 參考資源

- [Ollama 官網](https://ollama.ai)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Ollama 模型庫](https://ollama.ai/library)
- [Meditron 模型](https://ollama.ai/library/meditron)
- [LLaVA 視覺模型](https://ollama.ai/library/llava)

## 支持

如有問題，請：
1. 檢查 Ollama 服務器狀態: `ollama list`
2. 查看日誌: `ollama logs`
3. 運行診斷: `python app/setup_ollama.py`
