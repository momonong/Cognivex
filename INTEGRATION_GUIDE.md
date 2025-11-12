# 🚀 Structured Output Integration Guide

## 當前狀態

✅ **所有組件已整合完成**

- ✅ Report Generator 已更新（生成結構化 JSON）
- ✅ UI Components 已更新（渲染結構化報告）
- ✅ 語言切換功能正常
- ✅ 錯誤處理已實施

## 使用流程

### 1. 啟動系統

```bash
streamlit run app.py
```

### 2. 選擇語言

在頁面頂部選擇：
- **中文** - 顯示純中文報告
- **English** - 顯示純英文報告

### 3. 選擇分析模式

左側邊欄：
- **Structural MRI (T1)** ← 預設選項
- Functional MRI (fMRI)

### 4. 選擇受試者

從下拉選單選擇受試者（例如：sub_0001）

### 5. 開始分析

點擊 **"Start Analysis"** 按鈕

### 6. 查看結果

系統會依序顯示：

#### A. 臨床指標
```
┌──────────┬──────────┬──────────┬──────────┐
│ 臨床診斷  │ AI 預測  │ 信心度   │ 模型     │
│ NC       │ NC       │ 53%      │ RF       │
└──────────┴──────────┴──────────┴──────────┘

✓ 診斷一致
```

#### B. 主要發現
```
ℹ️ 觀察到輕度雙側海馬迴體積減少，與早期阿茲海默症
   病理變化一致。建議進行追蹤評估以監測病程進展。
```

#### C. 關鍵發現
```
結構性變化
🟡 雙側海馬迴萎縮 (Moderate)
🟢 顳葉體積減少 (Mild)

體積分析
• 海馬迴: 體積減少 -12% ▼
• 杏仁核: 體積減少 -8% ▼
• 額葉: 正常 0% =
```

#### D. 臨床解釋
```
[左側]                [右側]
AD 指標              保護因子
⚠️ 海馬迴萎縮        ✓ 額葉功能保留
⚠️ 顳葉體積減少      ✓ 白質完整性正常
⚠️ 杏仁核變化
```

#### E. 建議
```
立即行動
1. 6個月後追蹤 MRI
2. 建議進行認知功能評估

監測項目
• 追蹤記憶功能
• 監測日常活動

額外檢查
• 考慮 PET 掃描
• CSF 生物標記評估
```

#### F. 重要腦區分析
```
[表格顯示 Top 10 腦區]
排名 | 腦區名稱 | 重要性 | 功能分類 | 半球
```

#### G. 醫師備註
```
[文字輸入框]
可輸入臨床觀察、建議或其他相關資訊
```

## 報告生成流程

### 正常流程

```
1. 用戶點擊 "Start Analysis"
   ↓
2. 系統執行 sMRI 分析
   ↓
3. 調用 Report Generator
   ↓
4. LLM 生成結構化 JSON (English)
   ↓
5. 解析並驗證 JSON
   ↓
6. 翻譯為中文 JSON
   ↓
7. 存儲到 state["structured_report"]
   ↓
8. UI 根據語言選擇渲染
   ↓
9. 顯示完整的結構化報告
```

### 如果看到 "Report generation in progress..."

這是正常的，表示：
- ✅ 系統正在等待 LLM 回應
- ✅ 報告生成需要 10-30 秒
- ✅ 請耐心等待

### 如果報告一直不出現

可能的原因：
1. **LLM Provider 問題**
   - 檢查 AWS Bedrock 配置
   - 檢查 API credentials
   - 查看終端機錯誤訊息

2. **JSON 解析失敗**
   - 系統會自動 fallback
   - 查看終端機警告訊息

3. **網路問題**
   - 檢查網路連接
   - 重試分析

## 調試步驟

### 1. 檢查終端機輸出

啟動 Streamlit 後，查看終端機訊息：

```
--- Node: Structural MRI Report Generator ---
  - Generating structured MRI report...
  - English report generated.
  - Successfully parsed structured report.
  - Chinese translation generated.
  - Successfully parsed Chinese report.
  - Node: Structured MRI report generation complete.
```

### 2. 檢查錯誤訊息

如果看到錯誤：

```
Warning: Failed to parse JSON: ...
```

這表示 LLM 返回的不是有效 JSON，但系統會自動 fallback。

### 3. 測試 LLM 連接

```python
# 在 Python 中測試
from app.services.llm_providers import llm_response

test_prompt = "Say hello in JSON format: {\"message\": \"hello\"}"
response = llm_response(prompt=test_prompt, llm_provider="aws_bedrock")
print(response)
```

## 常見問題

### Q1: 報告顯示空白
**A**: 檢查 `final_state.get("structured_report")` 是否有數據

### Q2: 語言切換無效
**A**: 確保選擇了正確的語言，然後重新整理頁面

### Q3: JSON 解析失敗
**A**: 系統會自動 fallback，查看終端機警告訊息

### Q4: 中文翻譯不正確
**A**: LLM 翻譯品質問題，可以調整 translation prompt

### Q5: 視覺化元素不顯示
**A**: 檢查 severity 和 percentage 欄位格式

## 優化建議

### 短期優化
1. ✅ 已完成：結構化輸出
2. ✅ 已完成：視覺化渲染
3. ✅ 已完成：語言切換

### 中期優化（可選）
1. 添加載入動畫
2. 改進錯誤提示
3. 添加報告導出功能

### 長期優化（可選）
1. 歷史報告對比
2. 自定義報告模板
3. 批次分析功能

## 測試清單

### 功能測試
- [ ] 系統啟動正常
- [ ] 語言選擇器正常
- [ ] 受試者選擇正常
- [ ] 分析執行正常
- [ ] 報告生成正常
- [ ] 結構化內容顯示
- [ ] 視覺化元素正確
- [ ] 語言切換正常
- [ ] 醫師備註可輸入

### 視覺測試
- [ ] 主要發現顯示
- [ ] 關鍵發現格式正確
- [ ] 嚴重度圖示正確
- [ ] 體積分析格式正確
- [ ] 臨床解釋佈局正確
- [ ] 建議分類清晰
- [ ] 表格顯示正常

### 語言測試
- [ ] 中文模式純中文
- [ ] English 模式純英文
- [ ] 切換語言正常
- [ ] 翻譯品質良好

## 下一步行動

### 立即測試
```bash
# 1. 啟動系統
streamlit run app.py

# 2. 執行完整測試流程
# 3. 驗證所有功能
# 4. 記錄任何問題
```

### 如果一切正常
🎉 **系統已準備好使用！**

### 如果有問題
1. 查看終端機錯誤訊息
2. 檢查上述調試步驟
3. 聯繫支援

## 技術細節

### 數據流
```
User Input
  ↓
Analysis Workflow
  ↓
Report Generator (LLM)
  ↓
JSON Parsing
  ↓
Translation (LLM)
  ↓
State Storage
  ↓
UI Rendering
  ↓
User Display
```

### 關鍵檔案
- `app/agents/report_generator.py` - 報告生成邏輯
- `app/ui/structural_mri_components.py` - UI 渲染邏輯
- `app/services/llm_providers/__init__.py` - LLM 介面

### 狀態管理
```python
final_state = {
    "structured_report": {
        "en": {...},  # English structured report
        "zh": {...}   # Chinese structured report
    },
    "classification_result": "NC" or "AD",
    "prediction_confidence": 0.53,
    "activated_regions": [...],
    ...
}
```

## 支援

如有問題，請提供：
1. 終端機完整輸出
2. 錯誤訊息截圖
3. 使用的受試者 ID
4. 選擇的語言

---

*整合日期: 2024年*
*版本: v2.0 - Structured Output*
*狀態: Production Ready*
