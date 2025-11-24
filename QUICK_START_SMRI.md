# 快速啟動指南 - sMRI 專用版本

**目標**: 5 分鐘內啟動 Cognivex CDDA sMRI 分析系統

---

## 前置檢查

### ✅ 必須存在的檔案

```bash
# 檢查資料目錄
dir data\MRI_processed\AD\sub-0005
dir data\MRI_processed\MCI\sub-0003
dir data\MRI_processed\NC\sub-0002

# 檢查模型檔案
dir model\cnn_rf\rf_model_NC_vs_AD_GM_only.joblib

# 檢查主程式
dir app_smri.py
```

### ✅ 資料結構確認

你的資料應該長這樣：
```
data/MRI_processed/
├── AD/
│   └── sub-0005/
│       ├── sub-0005_GM_to_MNI.nii.gz  ← 必須
│       ├── sub-0005_FA_to_MNI.nii.gz
│       └── sub-0005_MD_to_MNI.nii.gz
├── MCI/
└── NC/
```

---

## 啟動步驟

### 方案 A: 規則式模式 (推薦 - 不需要 LLM)

```bash
# 1. 直接啟動
streamlit run app_smri.py

# 2. 在 UI 中:
#    - 選擇受試者 (例如: sub-0005)
#    - 確保「啟用 LLM 模式」是 **未勾選**
#    - 點擊「開始分析」

# 3. 等待分析完成 (約 30-60 秒)
```

**優點**:
- ✅ 不需要下載 LLM 模型
- ✅ 速度快
- ✅ 穩定可靠
- ✅ 仍然有反事實分析和異常檢測

**缺點**:
- ❌ 臨床報告使用模板生成 (不是 LLM 生成)

---

### 方案 B: LLM 模式 (需要 HuggingFace 模型)

#### 步驟 1: 下載模型 (如果還沒有)

```bash
# 建立模型目錄
mkdir D:\hf_models

# 使用 HuggingFace CLI 下載
huggingface-cli download microsoft/Phi-4 --local-dir D:\hf_models\Phi-4-mini-instruct
huggingface-cli download medgemma-27b --local-dir D:\hf_models\medgemma-27b
```

**注意**: 這些模型很大 (20GB + 27GB)，下載需要時間。

#### 步驟 2: 啟動系統

```bash
# 1. 啟動 Streamlit
streamlit run app_smri.py

# 2. 在 UI 中:
#    - 選擇受試者
#    - ✅ 勾選「啟用 LLM 模式」
#    - 設定模型路徑:
#      - Agent A: D:/hf_models/Phi-4-mini-instruct
#      - Agent B: D:/hf_models/medgemma-27b
#    - 點擊「開始分析」

# 3. 等待分析完成 (約 2-5 分鐘，首次載入模型較慢)
```

**優點**:
- ✅ LLM 生成的臨床報告 (更自然、更詳細)
- ✅ 智能決策 (Agent A 使用 LLM 決定工具調用)

**缺點**:
- ❌ 需要大量 GPU 記憶體 (建議 24GB+)
- ❌ 首次載入慢
- ❌ 可能失敗 (會自動降級到規則式模式)

---

### 方案 C: Ollama 模式 (本地 LLM)

#### 步驟 1: 安裝 Ollama

```bash
# 下載並安裝 Ollama
# https://ollama.ai/

# 安裝模型
ollama pull llama3.1:8b
ollama pull mistral
```

#### 步驟 2: 修改程式碼

在 `app_smri.py` 中，修改 Agent 初始化：

```python
agent = CDDAAgent(
    orchestrator_model="llama3.1:8b",  # 改用 Ollama 模型
    orchestrator_model_path=None,       # Ollama 不需要路徑
    consultant_model="mistral",
    consultant_model_path=None,
    # ... 其他參數
)
```

並在 `agent_a_orchestrator.py` 和 `agent_b_consultant.py` 中設定 `provider="ollama"`。

---

## 測試檢查清單

### ✅ 基本功能測試

```bash
# 1. 啟動系統
streamlit run app_smri.py

# 2. 選擇一個 AD 患者 (例如: sub-0005)
# 3. 點擊「開始分析」
# 4. 確認以下內容:
```

- [ ] 系統成功載入受試者列表
- [ ] 顯示正確的真實標籤 (AD)
- [ ] 分析進度條正常運作
- [ ] 顯示診斷結果 (AD/MCI/NC)
- [ ] 顯示信心度和不確定性分數
- [ ] 顯示診斷驗證 (預測 vs 真實)
- [ ] 可以展開推理鏈
- [ ] 可以查看 MRI 檢視器

### ✅ 進階功能測試

- [ ] 反事實分析 (當 UQ > 0.8 時觸發)
- [ ] 異常檢測 (顯示異常腦區)
- [ ] 知識圖譜查詢 (如果 Neo4j 可用)
- [ ] 推理鏈顯示完整
- [ ] 元數據 JSON 正確

---

## 常見問題

### Q1: 找不到受試者資料

**錯誤**: "找不到任何受試者資料"

**解決**:
```bash
# 檢查資料目錄
dir data\MRI_processed\AD
dir data\MRI_processed\MCI
dir data\MRI_processed\NC

# 確認至少有一個受試者資料夾
```

### Q2: 找不到 MRI 檔案

**錯誤**: "找不到 MRI 檔案: data/MRI_processed/AD/sub-0005/sub-0005_GM_to_MNI.nii.gz"

**解決**:
```bash
# 檢查檔案是否存在
dir data\MRI_processed\AD\sub-0005\sub-0005_GM_to_MNI.nii.gz

# 如果不存在，檢查檔案命名是否正確
dir data\MRI_processed\AD\sub-0005
```

### Q3: 模型載入失敗

**錯誤**: "找不到模型檔案"

**解決**:
```bash
# 檢查模型檔案
dir model\cnn_rf\rf_model_NC_vs_AD_GM_only.joblib

# 如果不存在，需要訓練或下載模型
```

### Q4: LLM 模式失敗

**錯誤**: "LLM orchestration failed" 或 "LLM synthesis failed"

**解決**:
- ✅ 系統會自動降級到規則式模式
- ✅ 分析仍然會完成
- ✅ 檢查模型路徑是否正確
- ✅ 檢查 GPU 記憶體是否足夠

### Q5: Neo4j 連接失敗

**錯誤**: "Neo4j connection failed"

**解決**:
- ✅ 系統會自動降級到本地知識庫
- ✅ 知識圖譜功能仍然可用 (使用 JSON 資料)
- ✅ 如果需要 Neo4j，檢查 `.env` 設定

---

## 效能優化

### 記憶體不足？

如果遇到記憶體問題：

1. **使用規則式模式** (不啟用 LLM)
2. **使用 8-bit 量化** (已預設啟用)
3. **關閉其他程式** 釋放記憶體
4. **使用較小的模型** (Ollama llama3.1:8b)

### 速度太慢？

如果分析太慢：

1. **使用規則式模式** (30-60 秒)
2. **使用 GPU** (如果有 CUDA)
3. **預載入模型** (修改程式碼快取模型)

---

## 下一步

### 🎯 開始使用

```bash
# 最簡單的方式
streamlit run app_smri.py
```

### 📚 深入了解

- 閱讀 `SYSTEM_ARCHITECTURE_SMRI.md` 了解系統架構
- 閱讀 `DATA_ARCHITECTURE_ISSUES.md` 了解資料結構
- 查看 `app/agents/cdda_agent.py` 了解 CDDA 實作

### 🔧 客製化

- 修改 `config/prompts/` 調整 LLM 提示
- 修改 `app/core/ml_processing/cdda_tools.py` 調整分析邏輯
- 修改 `app_smri.py` 調整 UI 介面

---

## 支援

如果遇到問題：

1. 檢查本文件的「常見問題」
2. 查看 `DATA_ARCHITECTURE_ISSUES.md`
3. 檢查終端機的錯誤訊息
4. 查看 Streamlit UI 的錯誤提示

---

**祝你使用愉快！** 🎉
