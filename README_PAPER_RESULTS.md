# CDDA Paper Results Generator

## 概述

`test_cdda_paper_results.py` 是一個專門為學術論文生成詳細實驗結果的腳本。

## 功能特點

### 1. 詳細的實驗結果輸出

- **診斷性能指標**
  - 準確率、信心度、不確定性
  - Ground truth vs AI prediction 比較
  
- **特徵重要性分析**
  - Top 10 診斷驅動因子
  - SHAP 值 + Z-score 分析
  - 統計摘要（均值、標準差）
  
- **Agent 決策分析**
  - MCP 動作日誌
  - 工具調用結果（反事實、知識圖譜）
  - 推理鏈統計
  
- **性能指標**
  - 初始化時間
  - 分析時間
  - 吞吐量（subjects/hour）
  - 記憶體使用

### 2. 多種輸出格式

#### 文本報告 (`.txt`)
- 完整的實驗結果
- 臨床報告全文
- 完整推理鏈

#### JSON 數據 (`.json`)
- 結構化數據
- 便於進一步分析
- 可用於繪圖和統計

#### LaTeX 表格 (`.tex`)
- 即用型 LaTeX 代碼
- Table 1: 診斷性能
- Table 2: 特徵重要性
- Table 3: 系統性能
- 可直接複製到論文中

## 使用方式

```bash
python test_cdda_paper_results.py
```

## 輸出文件

所有結果保存在 `output/paper_results/` 目錄：

```
output/paper_results/
├── paper_results_sub-XXXX_YYYYMMDD_HHMMSS.txt    # 完整文本報告
├── paper_results_sub-XXXX_YYYYMMDD_HHMMSS.json   # JSON 數據
└── paper_tables_sub-XXXX_YYYYMMDD_HHMMSS.tex     # LaTeX 表格
```

## 輸出示例

### 控制台輸出

```
====================================================================================================
  CDDA Framework - Experimental Results
====================================================================================================
Experiment Date: 2025-11-26 15:00:00
System: Cognitive Discrepancy-Driven Agent (CDDA)
Architecture: Dual-LLM A2A Pattern
  - Agent A (Orchestrator): Phi-4-mini
  - Agent B (Consultant): Llama3.1-Aloe-Beta-8B

----------------------------------------------------------------------------------------------------
  4.1 Diagnostic Performance
----------------------------------------------------------------------------------------------------

Table 1: Diagnostic Results
Metric                         | Value                | Interpretation                          
------------------------------------------------------------------------------------------------
Ground Truth                   | Alzheimer's Disease  | Clinical diagnosis                      
AI Prediction                  | Alzheimer's Disease  | [OK] Correct                            
Confidence Score               | 0.8734               | High                                    
Uncertainty (UQ) Score         | 0.2156               | Low                                     
Agent Decision Mode            | STANDARD_REPORT      | Adaptive decision-making                
```

### LaTeX 表格示例

```latex
\begin{table}[htbp]
\centering
\caption{Diagnostic Performance Metrics}
\label{tab:diagnostic_performance}
\begin{tabular}{lll}
\hline
Metric & Value & Interpretation \\
\hline
Ground Truth & Alzheimer's Disease & Clinical diagnosis \\
AI Prediction & Alzheimer's Disease & Correct \\
Confidence & 0.8734 & High \\
Uncertainty & 0.2156 & Low \\
\hline
\end{tabular}
\end{table}
```

## 修復的問題

1. **Feature 對象存取** - 使用 `safe_get_feature_attr()` 統一處理
2. **Unicode 編碼** - 所有特殊字符替換為 ASCII 兼容字符
3. **Windows 兼容性** - 確保在 Windows (cp950) 環境下正常運行

## 適用場景

- Conference paper 實驗結果章節
- 系統性能評估
- 診斷準確率報告
- 特徵重要性分析
- 系統架構說明

## 注意事項

1. 腳本會自動選擇第一個有效受試者進行測試
2. 使用 rule-based 模式以確保可重現性
3. 所有輸出文件都包含時間戳以避免覆蓋
4. LaTeX 表格可直接複製到論文中使用

## 相關文件

- `test_full_cdda_analysis.py` - 完整的功能測試腳本
- `test_agent_a_debug.py` - Agent A 調試腳本
- `BUGFIX_MISSING_MRI_FILES.md` - 數據驗證問題修復
- `API_COMPATIBILITY_FIX.md` - API 兼容性修復
