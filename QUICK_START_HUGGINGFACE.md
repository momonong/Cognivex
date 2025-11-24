# CDDA 系統快速開始指南 (HuggingFace 版本)

## 🚀 快速開始

### 1. 安裝依賴

```bash
# 基本依賴
pip install streamlit nilearn plotly

# HuggingFace 依賴
pip install transformers torch accelerate

# 可選：加速推理
pip install bitsandbytes  # 用於 8-bit/4-bit 量化
```

### 2. 下載模型 (可選)

如果你想使用 LLM 模式，需要下載模型：

```bash
# 安裝 HuggingFace CLI
pip install huggingface-hub

# 下載較小的測試模型 (推薦新手)
huggingface-cli download microsoft/phi-3-mini-4k-instruct --local-dir D:/hf_models/phi-3-mini

# 或下載更大的模型 (需要更多記憶體)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir D:/hf_models/llama-3.1-8b
```

### 3. 啟動應用

```bash
streamlit run app_cdda.py
```

### 4. 使用介面

#### 選項 A: 規則模式 (不需要 LLM，推薦開始)

1. 在側邊欄選擇 "CDDA Framework (推薦)"
2. **不要勾選** "啟用 LLM 模式"
3. 選擇受試者 (例如: sub-0005)
4. 點擊 "開始分析"

#### 選項 B: LLM 模式 (需要下載模型)

1. 在側邊欄選擇 "CDDA Framework (推薦)"
2. **勾選** "啟用 LLM 模式"
3. 輸入模型路徑：
   - Agent A: `D:/hf_models/phi-3-mini`
   - Agent B: `D:/hf_models/phi-3-mini`
4. 選擇受試者
5. 點擊 "開始分析"

## 📊 理解結果

### 診斷結果

- **AD**: 阿茲海默症 (Alzheimer's Disease)
- **MCI**: 輕度認知障礙 (Mild Cognitive Impairment)
- **NC**: 正常認知 (Normal Cognition)

### 關鍵指標

1. **診斷信心度**: 模型對診斷的信心程度
   - 高 (>80%): 模型非常確定
   - 中 (60-80%): 模型有一定信心
   - 低 (<60%): 模型不確定，需要更多檢查

2. **不確定性評分**: 模型的不確定性
   - 低 (<0.5): 診斷可靠
   - 中 (0.5-0.8): 需要臨床驗證
   - 高 (>0.8): 強烈建議進一步檢查

3. **異常腦區數量**: 檢測到統計異常的腦區
   - 0: 無異常
   - 1-3: 少量異常，可能是正常變異
   - >3: 多個異常，可能提示混合病理

4. **反事實分析**: 是否執行了模擬
   - 是: 系統識別了關鍵診斷驅動因子
   - 否: 標準診斷流程

### 臨床報告

報告包含以下部分：

1. **診斷摘要**: 整體診斷結果
2. **關鍵發現**: 最重要的腦區變化
3. **異常分析** (如有): 統計異常的腦區及其臨床意義
4. **反事實分析** (如有): 哪些腦區對診斷影響最大
5. **臨床解釋**: 整合所有證據的解釋
6. **建議事項**: 後續檢查和臨床建議

## 🔍 互動式 MRI 檢視器

展開 "探索原始 MRI 掃描" 區塊可以：

- 查看原始 MRI 影像
- 在三個平面 (軸向、冠狀、矢狀) 切換
- 調整切片位置
- 查看影像細節

## ⚙️ 進階設定

### 使用不同的模型

你可以為 Agent A 和 Agent B 使用不同的模型：

- **Agent A** (工具調用): 較小的模型即可 (3-8B)
  - microsoft/phi-3-mini-4k-instruct (3.8B)
  - meta-llama/Llama-3.1-8B-Instruct (8B)

- **Agent B** (臨床報告): 建議使用較大的模型 (7B+)
  - google/gemma-2-9b-it (9B)
  - meta-llama/Llama-3.1-8B-Instruct (8B)
  - google/gemma-2-27b-it (27B) - 需要大量記憶體

### 記憶體不足？

如果遇到記憶體問題：

1. 使用較小的模型 (3-8B)
2. 確保啟用了 8-bit 量化 (預設已啟用)
3. 關閉其他使用 GPU 的程式
4. 考慮使用規則模式 (不需要 LLM)

## 🐛 常見問題

### Q: 找不到模型

**A**: 確認模型路徑正確，並且目錄中包含以下檔案：
- `config.json`
- `*.safetensors` 檔案
- `tokenizer.json` 或 `tokenizer_config.json`

### Q: CUDA 記憶體不足

**A**: 
1. 使用較小的模型
2. 確保啟用 8-bit 量化
3. 重啟應用以清除快取

### Q: 生成速度很慢

**A**:
1. 確認使用 GPU 而非 CPU
2. 使用較小的模型
3. 考慮使用規則模式 (速度快很多)

### Q: 找不到 NIfTI 檔案

**A**: 
1. 確認資料位於 `data/MRI_processed/` 目錄
2. 檢查目錄結構: `data/MRI_processed/{AD|MCI|NC}/sub-XXXX/`
3. 確認檔案名稱格式正確

### Q: 輸出不是繁體中文

**A**: 
1. 某些模型可能不支援繁體中文
2. 嘗試使用其他模型
3. 或使用規則模式 (輸出為繁體中文)

## 📚 更多資源

- **完整設定指南**: `docs/HUGGINGFACE_SETUP.md`
- **修復摘要**: `FIXES_APPLIED_20241121.md`
- **測試腳本**: `test_huggingface_integration.py`

## 💡 提示

1. **第一次使用**: 建議先使用規則模式熟悉系統
2. **測試 LLM**: 先用較小的模型 (phi-3-mini) 測試
3. **生產環境**: 使用較大的模型 (8B+) 以獲得更好的報告品質
4. **批次分析**: 可以連續分析多個受試者，系統會快取模型

## 🎯 下一步

1. 嘗試分析不同的受試者
2. 比較 AD、MCI、NC 的診斷結果
3. 查看推理鏈了解系統決策過程
4. 探索互動式 MRI 檢視器
5. 匯出報告用於臨床討論

---

**需要幫助？** 查看 `docs/` 目錄中的詳細文件或執行測試腳本 `python test_huggingface_integration.py`
