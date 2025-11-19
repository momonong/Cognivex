## 🔒 隱私模式完整指南

## 概述

Cognivex 現在支持**完全本地化的 LLM 推理**，確保醫療數據不離開你的系統。

### 隱私保護架構

```
┌─────────────────────────────────────────────────────────────┐
│                    Cognivex 應用                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  隱私模式 OFF    │      │  隱私模式 ON     │            │
│  │  (默認)          │      │  (推薦)          │            │
│  └──────────────────┘      └──────────────────┘            │
│          │                          │                        │
│          ▼                          ▼                        │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  AWS Bedrock     │      │  Ollama (本地)   │            │
│  │  (雲端)          │      │  (本地)          │            │
│  └──────────────────┘      └──────────────────┘            │
│          │                          │                        │
│          ▼                          ▼                        │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  數據傳送到      │      │  數據保留在      │            │
│  │  AWS 服務器      │      │  本地機器        │            │
│  └──────────────────┘      └──────────────────┘            │
│          ❌                         ✅                       │
│     隱私風險                    隱私保護                    │
└─────────────────────────────────────────────────────────────┘
```

## 快速啟用隱私模式

### 方法 1: 環境變量 (推薦)

```bash
# Linux/Mac
export PRIVACY_MODE=true

# Windows (CMD)
set PRIVACY_MODE=true

# Windows (PowerShell)
$env:PRIVACY_MODE="true"
```

### 方法 2: 代碼中啟用

```python
import os
os.environ['PRIVACY_MODE'] = 'true'

# 之後所有 LLM 調用將使用本地 Ollama
from app.services.llm_providers import llm_response
response = llm_response("Generate medical report")
```

### 方法 3: 配置文件

```python
# app/services/llm_providers/config.py

# 修改默認提供者
DEFAULT_PROVIDER = "ollama"  # 從 "aws_bedrock" 改為 "ollama"
```

## 完整設置流程

### 步驟 1: 安裝 Ollama

```bash
# 訪問 https://ollama.ai 下載安裝程序

# 或使用命令行 (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# 驗證安裝
ollama --version
```

### 步驟 2: 安裝 Python 套件

```bash
pip install ollama
```

### 步驟 3: 下載模型

```bash
# 推薦：輕量級模型 (4GB RAM)
ollama pull llama3.2:3b

# 或：更好質量的模型 (8GB RAM)
ollama pull llama3.1:8b

# 或：醫療專用模型 (8GB RAM)
ollama pull meditron:7b
```

### 步驟 4: 運行設置腳本

```bash
python app/setup_ollama.py
```

### 步驟 5: 測試整合

```bash
python app/test_ollama_integration.py
```

### 步驟 6: 啟用隱私模式

```bash
export PRIVACY_MODE=true
```

### 步驟 7: 運行應用

```bash
python app/test_cnn_rf_integration.py --mode single
```

## 使用場景

### 場景 1: 醫療報告生成 (隱私保護)

```python
import os
os.environ['PRIVACY_MODE'] = 'true'

from app.graph.workflow import app

# 所有 LLM 調用將使用本地 Ollama
state = {
    "subject_id": "sub-0005",
    "analysis_mode": "structural",
    "model_type": "cnn_rf",
}

result = app.invoke(state)
# 報告生成完全在本地完成，數據不上傳
```

### 場景 2: 混合模式 (靈活切換)

```python
from app.services.llm_providers import llm_response, use_local_llm, use_cloud_llm

# 敏感數據使用本地
use_local_llm()
sensitive_report = llm_response("Generate patient diagnosis")

# 非敏感數據可以使用雲端 (更快)
use_cloud_llm()
general_info = llm_response("Explain Alzheimer's disease")
```

### 場景 3: 批量處理 (離線)

```python
import os
os.environ['PRIVACY_MODE'] = 'true'

from app.services.llm_providers import llm_response

# 處理多個患者，完全離線
patients = ["sub-0001", "sub-0002", "sub-0003"]

for patient_id in patients:
    report = llm_response(
        f"Generate report for {patient_id}",
        llm_provider="ollama"
    )
    # 保存報告...
```

## 性能對比

### 推理速度

| 場景 | AWS Bedrock | Ollama (本地) |
|------|-------------|---------------|
| 短文本 (100 tokens) | 1-2秒 | 2-5秒 |
| 長文本 (500 tokens) | 3-5秒 | 10-20秒 |
| 批量處理 (10個請求) | 10-20秒 | 20-100秒 |

### 成本對比

| 使用量 | AWS Bedrock | Ollama (本地) |
|--------|-------------|---------------|
| 1,000 請求/月 | $10-50 | $0 |
| 10,000 請求/月 | $100-500 | $0 |
| 100,000 請求/月 | $1,000-5,000 | $0 |

### 隱私對比

| 特性 | AWS Bedrock | Ollama (本地) |
|------|-------------|---------------|
| 數據傳輸 | ❌ 傳送到雲端 | ✅ 保留本地 |
| 數據存儲 | ❌ AWS 服務器 | ✅ 本地磁盤 |
| 第三方訪問 | ⚠️ 可能 | ✅ 無 |
| HIPAA 合規 | ⚠️ 需配置 | ✅ 天然合規 |
| 審計追蹤 | ⚠️ AWS 日誌 | ✅ 本地日誌 |

## 醫療合規性

### HIPAA 要求

使用 Ollama 滿足 HIPAA 要求：

1. ✅ **數據加密** - 數據不離開本地，無需傳輸加密
2. ✅ **訪問控制** - 完全由你控制
3. ✅ **審計日誌** - 本地日誌記錄
4. ✅ **數據最小化** - 數據不共享給第三方
5. ✅ **業務夥伴協議** - 不需要 (無第三方)

### GDPR 合規

1. ✅ **數據本地化** - 數據保留在歐盟境內
2. ✅ **數據處理透明** - 完全可見的處理過程
3. ✅ **數據刪除權** - 完全控制數據刪除
4. ✅ **數據可攜性** - 數據在本地，易於遷移

## 故障排除

### 問題 1: Ollama 未安裝

```
[WARNING] ollama not installed
```

**解決方案**:
```bash
pip install ollama
```

### 問題 2: Ollama 服務器未運行

```
[ERROR] Ollama server is not running
```

**解決方案**:
```bash
# 啟動服務器
ollama serve

# 或檢查是否已在後台運行
ps aux | grep ollama  # Linux/Mac
tasklist | findstr ollama  # Windows
```

### 問題 3: 模型未下載

```
[ERROR] Model not found: llama3.2:3b
```

**解決方案**:
```bash
# 下載模型
ollama pull llama3.2:3b

# 查看已安裝的模型
ollama list
```

### 問題 4: 內存不足

```
[ERROR] Out of memory
```

**解決方案**:
- 使用更小的模型: `ollama pull llama3.2:3b`
- 關閉其他應用程序
- 增加系統 RAM

### 問題 5: 推理速度慢

**解決方案**:
- 使用 GPU 加速 (如果有 NVIDIA GPU)
- 使用更小的模型
- 減少 batch size
- 升級硬件

## 最佳實踐

### 1. 選擇合適的模型

```python
# 開發/測試: 使用小模型
config = get_config_by_name("ollama")  # llama3.2:3b

# 生產環境: 使用平衡模型
config = get_config_by_name("ollama_large")  # llama3.1:8b

# 醫療專用: 使用醫療模型
config = get_config_by_name("ollama_medical")  # meditron:7b
```

### 2. 設置合理的超時

```python
from app.services.llm_providers import llm_response

response = llm_response(
    "Generate report",
    llm_provider="ollama",
    timeout=60  # 60秒超時
)
```

### 3. 實施錯誤處理

```python
from app.services.llm_providers import llm_response

try:
    response = llm_response(
        "Generate report",
        llm_provider="ollama"
    )
except Exception as e:
    print(f"Ollama failed: {e}")
    # 回退到雲端或使用緩存
    response = cached_response
```

### 4. 監控性能

```python
import time

start = time.time()
response = llm_response("Generate report", llm_provider="ollama")
duration = time.time() - start

print(f"Inference took {duration:.2f} seconds")

# 如果太慢，考慮切換模型或提供者
if duration > 30:
    print("Consider using a smaller model or cloud provider")
```

### 5. 定期更新模型

```bash
# 更新到最新版本
ollama pull llama3.2:3b

# 查看可用更新
ollama list
```

## 安全建議

### 1. 網絡隔離

```bash
# 確保 Ollama 只監聽本地
# 在 Ollama 配置中設置
OLLAMA_HOST=127.0.0.1:11434
```

### 2. 訪問控制

```bash
# 限制對 Ollama 的訪問
# 使用防火牆規則
sudo ufw deny 11434  # 阻止外部訪問
```

### 3. 日誌記錄

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 記錄所有 LLM 調用
logger.info(f"LLM call: provider=ollama, model=llama3.2:3b")
```

### 4. 數據清理

```python
# 定期清理臨時文件
import shutil
shutil.rmtree("cache/llm_responses", ignore_errors=True)
```

## 總結

### 何時使用隱私模式

✅ **應該使用**:
- 處理真實患者數據
- 需要 HIPAA/GDPR 合規
- 離線環境
- 成本敏感
- 需要完全控制

❌ **可以不用**:
- 處理公開數據
- 需要最新模型
- 需要最快速度
- 不關心成本
- 信任雲端提供商

### 推薦配置

**生產環境 (醫療數據)**:
```bash
export PRIVACY_MODE=true
export LLM_PROVIDER=ollama
ollama pull meditron:7b
```

**開發環境 (測試)**:
```bash
export PRIVACY_MODE=false
export LLM_PROVIDER=aws_bedrock
```

### 下一步

1. ✅ 運行設置腳本: `python app/setup_ollama.py`
2. ✅ 測試整合: `python app/test_ollama_integration.py`
3. ✅ 啟用隱私模式: `export PRIVACY_MODE=true`
4. ✅ 運行應用: `python app/test_cnn_rf_integration.py`
5. ✅ 監控性能和日誌

## 支持

如有問題:
1. 查看文檔: `docs/OLLAMA_INTEGRATION.md`
2. 運行診斷: `python app/setup_ollama.py`
3. 檢查日誌: `ollama logs`
4. 測試連接: `curl http://localhost:11434/api/tags`
