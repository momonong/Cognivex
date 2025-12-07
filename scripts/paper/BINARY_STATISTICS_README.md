# Binary Statistics Script 使用說明

## 概述

`binary_statistics.py` 是專為 **NC vs AD 二分類系統**設計的統計分析腳本，提供完整的 Paper-ready 報告。

## 主要功能

### 1. LOOCV 完整性驗證 ✓
- 自動驗證每個受試者是否使用專屬的 LOOCV 模型
- 確保 100% 嚴格的訓練-測試分離
- 識別使用通用模型的 MCI/OOD 案例

### 2. 二分類性能指標
- **Accuracy**: 整體準確率
- **Precision**: AD 預測精確度
- **Recall/Sensitivity**: AD 檢測靈敏度
- **Specificity**: NC 檢測特異性
- **F1-Score**: 精確度與召回率的調和平均
- **Balanced Accuracy**: 平衡準確率
- **Confusion Matrix**: 完整混淆矩陣

### 3. 不確定性量化分析
- UQ Score 分布統計
- 高不確定性案例識別
- 信心度與不確定性的關聯分析

### 4. Agent 決策路徑統計
- Standard Pathway: 標準診斷流程
- Counterfactual Simulation: 反事實模擬觸發
- Knowledge Graph Query: 知識圖譜查詢觸發

### 5. 特徵重要性分析
- Top 20 最重要的腦區特徵
- SHAP values 平均值
- Z-scores 統計分布

### 6. 異常檢測統計
- 異常檢測率
- 最常見的異常腦區
- 異常與預測準確性的關聯

## 使用方法

### 基本用法
```bash
python scripts/paper/binary_statistics.py
```

### 指定輸出目錄
```bash
python scripts/paper/binary_statistics.py --output output/my_stats
```

### 測試模式 (限制受試者數量)
```bash
python scripts/paper/binary_statistics.py --limit 10
```

### 禁用 LLM (使用規則決策)
```bash
python scripts/paper/binary_statistics.py --no-llm
```

### 完整參數
```bash
python scripts/paper/binary_statistics.py \
    --output output/binary_stats \
    --orchestrator-path D:/hf_models/Phi-4-mini-instruct \
    --consultant-path D:/hf_models/Llama3.1-Aloe-Beta-8B \
    --use-4bit \
    --binary-only \
    --limit 50
```

## 輸出文件

腳本會生成以下文件：

### 1. `binary_statistics_report.txt`
完整的文字報告，包含：
- LOOCV 完整性驗證結果
- 二分類性能指標
- 詳細的統計分析
- 關鍵發現總結

### 2. `binary_statistics.json`
結構化的 JSON 數據，包含：
- 所有統計指標
- 每個受試者的詳細結果
- 可用於進一步分析或可視化

### 3. `binary_statistics.csv`
CSV 格式的結果表格，包含：
- Subject ID
- Prediction
- Confidence
- UQ Score
- Ground Truth
- Model Used
- LOOCV Verified

### 4. `binary_performance_table.tex`
LaTeX 格式的性能表格，可直接用於 Paper：
```latex
\begin{table}[htbp]
\centering
\caption{Binary Classification Performance (NC vs AD)}
\label{tab:binary_performance}
\begin{tabular}{lc}
\toprule
Metric & Value \\
\midrule
Accuracy & 0.9500 \\
Precision (AD) & 0.9400 \\
...
\end{tabular}
\end{table}
```

## Paper 撰寫建議

### Methods Section
使用以下信息描述評估方法：
- LOOCV 策略 (從 Section 0)
- 性能指標定義 (從 Section 2)
- 不確定性量化方法 (從 Section 5)

### Results Section
使用以下數據：
- 二分類性能表格 (LaTeX table)
- 混淆矩陣 (Section 2)
- 信心度與 UQ 分布 (Section 4-5)
- Agent 決策路徑統計 (Section 6)

### Discussion Section
參考以下發現：
- LOOCV 完整性驗證結果 (Section 0)
- 高不確定性案例分析 (Section 5)
- 異常檢測與臨床意義 (Section 7)
- 特徵重要性與神經解剖學關聯 (Section 8)

### Supplementary Materials
可包含：
- 完整的統計報告 (TXT)
- 詳細的受試者結果 (CSV)
- 推理鏈分析 (Section 10)

## 關鍵指標解讀

### LOOCV Coverage
- **100%**: ✓ 完美的訓練-測試分離
- **95-99%**: ⚠ 高覆蓋率但需檢查
- **< 95%**: ✗ 需要重新檢查模型配置

### Accuracy vs Balanced Accuracy
- **Accuracy**: 整體準確率，適用於平衡數據集
- **Balanced Accuracy**: 考慮類別不平衡，更適合不平衡數據集

### Sensitivity vs Specificity
- **Sensitivity (Recall)**: AD 檢測能力 (避免漏診)
- **Specificity**: NC 識別能力 (避免誤診)
- 臨床上通常優先考慮 Sensitivity

### UQ Score 解讀
- **< 0.3**: 低不確定性，模型有信心
- **0.3-0.5**: 中等不確定性
- **0.5-0.8**: 高不確定性，建議臨床複核
- **> 0.8**: 極高不確定性，可能觸發反事實模擬

## 常見問題

### Q: 為什麼有些受試者使用 Global Model？
A: 這些通常是 MCI 受試者或數據集外的新病人，因為 LOOCV 模型只針對 NC/AD 訓練。

### Q: 如何提高 LOOCV Coverage？
A: 確保 `model/loocv_models_binary_opt/` 目錄包含所有受試者的專屬模型。

### Q: 報告中的 "Low Confidence + High UQ" 代表什麼？
A: 這些案例模型既不確定又缺乏信心，建議優先進行臨床複核。

### Q: 如何解讀 Agent Decision Pathways？
A: 
- **Standard**: 常規診斷流程
- **Counterfactual**: 高不確定性觸發反事實分析
- **Knowledge Query**: 異常檢測觸發知識圖譜查詢

## 技術細節

### 模型驗證邏輯
腳本通過解析 reasoning chain 中的 log 來驗證模型使用：
```python
# 匹配: "using rf_model_sub-001.joblib"
match = re.search(r"using ([\w\-\.]+\.joblib)", full_log)
if subject_id in model_name:
    return "loocv_verified"
```

### 二分類指標計算
```python
# Confusion Matrix
TP = AD correctly identified as AD
TN = NC correctly identified as NC
FP = NC incorrectly identified as AD
FN = AD incorrectly identified as NC

# Metrics
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
Specificity = TN / (TN + FP)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

## 與舊版 comprehensive_statistics.py 的差異

| 特性 | 舊版 | 新版 (binary_statistics.py) |
|------|------|----------------------------|
| 分類類型 | 3-class (NC/MCI/AD) | 2-class (NC/AD) |
| LOOCV 驗證 | 部分支持 | 完整驗證 + 覆蓋率計算 |
| 性能指標 | 基本準確率 | 完整二分類指標 (Precision, Recall, F1, etc.) |
| LaTeX 輸出 | 無 | 自動生成 Paper-ready 表格 |
| 模型追蹤 | 無 | 每個受試者的模型使用記錄 |
| 報告格式 | 通用 | 針對二分類優化 |

## 聯繫與支持

如有問題或建議，請參考：
- 主要文檔: `README.md`
- 系統架構: `docs/CDDA_Architecture_Spec.md`
- Agent 實現: `app/agents/cdda_agent.py`
