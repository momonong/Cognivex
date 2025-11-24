# 修復摘要 - 2024/11/21

## 問題描述

1. **Ollama 依賴問題**: 系統依賴 Ollama，但使用者希望直接使用 HuggingFace 模型
2. **互動式切片找不到檔案**: 前端的 MRI 檢視器無法找到原始 NIfTI 檔案
3. **前端資訊不實用**: 顯示的資訊對臨床使用者沒有幫助，需要更清晰的臨床解釋

## 已應用的修復

### 1. HuggingFace 整合 (移除 Ollama 依賴)

#### 修改檔案: `app_cdda.py`

**變更內容**:
- 新增 HuggingFace 模型路徑設定介面
- 允許使用者在側邊欄輸入 Agent A 和 Agent B 的模型路徑
- 更新 `initialize_cdda_agent()` 函數以接受模型路徑參數
- 傳遞模型路徑到 `CDDAAgent` 初始化

**使用方式**:
```python
# 在 Streamlit 側邊欄中
1. 勾選 "啟用 LLM 模式"
2. 輸入 Agent A 模型路徑: D:/hf_models/gpt-oss-20b
3. 輸入 Agent B 模型路徑: D:/hf_models/medgemma-27b
4. 開始分析
```

#### 修改檔案: `app/agents/cdda_agent.py`

**變更內容**:
- 更新初始化邏輯以顯示使用的 provider (HuggingFace 或 Ollama)
- 當提供 `model_path` 時自動使用 HuggingFace provider
- 改善日誌輸出以顯示模型路徑和 provider 資訊

#### 修改檔案: `app/agents/agent_b_consultant.py`

**變更內容**:
- 更新 `_call_llm()` 方法以更好地處理 HuggingFace 模型
- 新增模型存在性檢查
- 改善錯誤處理和日誌輸出
- 將 `max_new_tokens` 增加到 2048 以支援更長的臨床報告
- 新增繁體中文提示以生成中文報告

**關鍵改進**:
```python
# 自動檢測 provider
if self.config.provider == "huggingface":
    if not self.config.model_path:
        raise ValueError("model_path required for HuggingFace provider")
    
    # 檢查模型是否存在
    model_info = huggingface.get_model_info(self.config.model_path)
    if not model_info['exists']:
        raise FileNotFoundError(f"Model not found at: {self.config.model_path}")
    
    # 使用 HuggingFace
    response_text = huggingface.handle_text(
        prompt=user_prompt,
        model_path=self.config.model_path,
        system_instruction=self.system_prompt,
        temperature=self.config.temperature,
        max_new_tokens=2048,
        load_in_8bit=self.config.load_in_8bit
    )
```

### 2. 修復互動式切片找不到檔案

#### 修改檔案: `app_cdda.py`

**變更內容**:
- 新增智慧檔案搜尋邏輯
- 當 `nii_path` 不存在時，自動搜尋多個可能的路徑
- 支援多種目錄結構 (MRI_processed, sMRI, fMRI)
- 顯示實際載入的檔案路徑

**搜尋路徑**:
```python
possible_paths = [
    f"data/MRI_processed/{ground_truth_label}/{selected_subject}/anat/{selected_subject}_T1w.nii.gz",
    f"data/MRI_processed/{ground_truth_label}/{selected_subject}/{selected_subject}_T1w.nii.gz",
    f"data/sMRI/{ground_truth_label}/{selected_subject}/anat/{selected_subject}_T1w.nii.gz",
    f"data/fMRI/{ground_truth_label}/{selected_subject}/func/{selected_subject}_task-rest_bold.nii.gz",
]
```

**改進**:
- 自動偵測並載入第一個找到的檔案
- 顯示檔案路徑給使用者確認
- 提供清晰的錯誤訊息和建議

### 3. 改善前端臨床資訊呈現

#### 修改檔案: `app_cdda.py`

**變更 1: 重新設計 `format_cdda_report()` 函數**

**改進內容**:
- 新增診斷結果的中文翻譯 (AD → 阿茲海默症)
- 新增代理決策的中文說明
- 使用顏色編碼表示信心度和不確定性等級
- 重新設計 HTML 佈局，更清晰易讀
- 顯示 Agent B 生成的完整臨床報告
- 改善反事實分析的呈現，強調臨床意義
- 改善異常分析的呈現，包含知識圖譜摘要

**關鍵改進**:
```python
# 診斷結果翻譯
diagnosis_map = {
    'AD': '阿茲海默症 (Alzheimer\'s Disease)',
    'MCI': '輕度認知障礙 (Mild Cognitive Impairment)',
    'NC': '正常認知 (Normal Cognition)'
}

# 信心度等級
if result.confidence > 0.8:
    confidence_level = "高信心度"
    confidence_color = "#388e3c"
elif result.confidence > 0.6:
    confidence_level = "中等信心度"
    confidence_color = "#f57c00"
else:
    confidence_level = "低信心度"
    confidence_color = "#d32f2f"

# 不確定性等級
if result.uq_score > 0.8:
    uq_level = "高不確定性 - 建議進一步檢查"
    uq_color = "#d32f2f"
```

**變更 2: 重新設計結果顯示區塊**

**改進內容**:
- 新增 "診斷驗證" 區塊，清楚比較真實診斷與 AI 預測
- 新增 "關鍵診斷指標" 區塊，顯示 4 個關鍵指標：
  1. 診斷信心度 (帶等級標示)
  2. 不確定性評分 (帶等級標示)
  3. 異常腦區數量
  4. 是否執行反事實分析
- 使用 Streamlit 的 `metric` 組件提供視覺化指標
- 新增 delta 指示器顯示指標等級
- 改善錯誤訊息的顯示

**視覺化改進**:
```python
# 使用 metric 組件顯示關鍵指標
st.metric(
    "診斷信心度", 
    f"{result.confidence:.1%}",
    delta=confidence_delta,  # "高", "中", "低"
    delta_color=confidence_color,  # "normal", "off", "inverse"
    help="模型對診斷結果的信心程度"
)
```

### 4. 新增文件

#### 新檔案: `docs/HUGGINGFACE_SETUP.md`

**內容**:
- HuggingFace 模型設定完整指南
- 系統需求和硬體建議
- 模型下載方法 (CLI 和 Python)
- 在 CDDA Web 介面中的使用步驟
- 推薦模型列表 (Agent A 和 Agent B)
- 記憶體優化技巧 (8-bit/4-bit 量化)
- 常見問題故障排除
- 效能比較表
- 進階設定範例

## 測試建議

### 1. 測試 HuggingFace 整合

```bash
# 確保已安裝必要套件
pip install transformers torch accelerate

# 下載測試模型 (較小的模型)
huggingface-cli download microsoft/phi-3-mini-4k-instruct --local-dir D:/hf_models/phi-3-mini

# 啟動應用
streamlit run app_cdda.py

# 在介面中:
# 1. 勾選 "啟用 LLM 模式"
# 2. 輸入模型路徑
# 3. 選擇受試者
# 4. 開始分析
```

### 2. 測試檔案搜尋

```bash
# 確認資料目錄結構
ls data/MRI_processed/AD/sub-0001/

# 啟動應用並選擇受試者
streamlit run app_cdda.py

# 展開 "探索原始 MRI 掃描" 區塊
# 應該能看到檔案路徑和互動式檢視器
```

### 3. 測試前端顯示

```bash
# 執行完整分析
streamlit run app_cdda.py

# 檢查以下內容:
# 1. 診斷結果是否顯示中文翻譯
# 2. 信心度和不確定性是否有顏色編碼
# 3. 關鍵指標是否正確顯示
# 4. 臨床報告是否清晰易讀
# 5. 反事實分析是否有臨床意義說明
```

## 已知限制

1. **模型大小**: 大型模型 (27B+) 需要大量記憶體，建議使用 8-bit 量化
2. **推理速度**: HuggingFace 本地推理比 Ollama 慢，特別是在 CPU 上
3. **模型下載**: 需要手動下載模型，無法自動下載
4. **繁體中文支援**: 某些模型可能不支援繁體中文輸出，需要測試

## 後續改進建議

1. **自動模型下載**: 新增自動下載模型的功能
2. **模型快取管理**: 新增清除快取和管理多個模型的介面
3. **效能監控**: 新增推理時間和記憶體使用監控
4. **批次處理**: 支援批次分析多個受試者
5. **報告匯出**: 新增匯出 PDF 或 Word 格式報告的功能
6. **多語言支援**: 支援英文和繁體中文切換

## 相關檔案

- `app_cdda.py` - 主要 Web 介面
- `app/agents/cdda_agent.py` - CDDA Agent 主類別
- `app/agents/agent_b_consultant.py` - Agent B (臨床顧問)
- `app/services/llm_providers/huggingface.py` - HuggingFace provider
- `docs/HUGGINGFACE_SETUP.md` - 設定指南

## 總結

這次修復解決了三個主要問題：

1. ✅ **移除 Ollama 依賴**: 現在可以直接使用 HuggingFace 模型，無需 Ollama
2. ✅ **修復檔案搜尋**: 互動式切片器現在可以自動找到 NIfTI 檔案
3. ✅ **改善臨床資訊**: 前端顯示更清晰、更有臨床價值的資訊

系統現在更靈活、更易用，並且提供更好的臨床決策支援。
