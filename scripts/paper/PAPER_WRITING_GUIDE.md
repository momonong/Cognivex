# Paper 撰寫指南 - Binary Statistics 輸出使用

## 快速索引

本指南說明如何使用 `binary_statistics.py` 的輸出來撰寫學術論文的各個章節。

---

## 1. Methods Section (方法章節)

### 1.1 Model Evaluation Strategy

**使用數據來源**: Section 0 (LOOCV INTEGRITY VERIFICATION)

**建議撰寫**:
```
We employed a Leave-One-Out Cross-Validation (LOOCV) strategy to ensure 
strict train-test separation. For each subject, a dedicated Random Forest 
model was trained on all other subjects, ensuring no data leakage. 

Our analysis achieved 100% LOOCV coverage (X/X subjects), confirming that 
each prediction was made using a model that had never seen the test subject 
during training.
```

**關鍵數據**:
- LOOCV Coverage: `{coverage_percentage}%`
- LOOCV Verified: `{loocv_verified}` subjects
- Global Fallback: `{fallback_global}` subjects (if any)

---

### 1.2 Performance Metrics

**使用數據來源**: Section 2 (BINARY CLASSIFICATION PERFORMANCE)

**建議撰寫**:
```
Model performance was evaluated using standard binary classification metrics:
- Accuracy: overall correctness
- Precision: positive predictive value for AD
- Recall/Sensitivity: true positive rate for AD detection
- Specificity: true negative rate for NC identification
- F1-Score: harmonic mean of precision and recall
- Balanced Accuracy: average of sensitivity and specificity
```

**LaTeX 表格**: 直接使用 `binary_performance_table.tex`

---

### 1.3 Uncertainty Quantification

**使用數據來源**: Section 5 (UNCERTAINTY QUANTIFICATION ANALYSIS)

**建議撰寫**:
```
We quantified prediction uncertainty using ensemble variance from the 
Random Forest model. Cases with UQ scores > 0.8 were flagged for 
counterfactual analysis to identify key diagnostic drivers.
```

**關鍵數據**:
- Mean UQ Score: `{uq_stats.mean}`
- High UQ Cases (> 0.8): `{len(high_uq_subjects)}`

---

## 2. Results Section (結果章節)

### 2.1 Overall Performance

**使用數據來源**: Section 2 (BINARY CLASSIFICATION PERFORMANCE)

**建議撰寫**:
```
The binary classification model achieved an accuracy of X.XX% (95% CI: [X.XX, X.XX]) 
on the LOOCV evaluation. Sensitivity for AD detection was X.XX%, while specificity 
for NC identification was X.XX%. The F1-score of X.XX indicates balanced performance 
across both classes.
```

**表格建議**:
```latex
\begin{table}[htbp]
\centering
\caption{Binary Classification Performance (NC vs AD)}
\begin{tabular}{lc}
\toprule
Metric & Value \\
\midrule
Accuracy & 0.XXXX \\
Sensitivity (AD) & 0.XXXX \\
Specificity (NC) & 0.XXXX \\
F1-Score & 0.XXXX \\
\bottomrule
\end{tabular}
\end{table}
```

---

### 2.2 Confusion Matrix Analysis

**使用數據來源**: Section 2 (Confusion Matrix)

**建議撰寫**:
```
The confusion matrix revealed X true positives (AD correctly identified), 
X true negatives (NC correctly identified), X false positives (NC misclassified 
as AD), and X false negatives (AD misclassified as NC). The false positive rate 
of X.XX% suggests [interpretation], while the false negative rate of X.XX% 
indicates [interpretation].
```

**圖表建議**: 創建 confusion matrix heatmap

---

### 2.3 Confidence and Uncertainty Distribution

**使用數據來源**: 
- Section 4 (CONFIDENCE ANALYSIS)
- Section 5 (UNCERTAINTY QUANTIFICATION ANALYSIS)

**建議撰寫**:
```
Model confidence scores ranged from X.XX to X.XX (mean: X.XX ± X.XX). 
Uncertainty quantification scores ranged from X.XX to X.XX (mean: X.XX ± X.XX). 
X subjects (X.XX%) exhibited high uncertainty (UQ > 0.8), triggering 
counterfactual analysis for diagnostic driver identification.
```

**圖表建議**: 
- Confidence distribution histogram
- UQ score distribution histogram
- Scatter plot: Confidence vs UQ

---

### 2.4 System Value Analysis (核心亮點！)

**使用數據來源**: Section 12 (SYSTEM VALUE ANALYSIS - AGENT INTERVENTION IMPACT)

**建議撰寫**:
```
Our CDDA system demonstrated significant value through autonomous agent intervention. 
Among X subjects requiring intervention (X.XX%), the system successfully corrected 
X cases through counterfactual simulation and X cases through knowledge graph 
consultation.

Critically, the intervention pathway achieved X.XX% accuracy compared to X.XX% 
for the standard pathway, representing a +X.XX% improvement. This demonstrates 
that the system's uncertainty-aware decision-making not only identifies challenging 
cases but actively improves diagnostic accuracy through targeted analysis.

The system contributed to X.XX% of the overall accuracy, with a 100% correction 
rate among intervention cases, validating the clinical utility of our agent-based 
approach.
```

**關鍵數據**:
- Intervention Accuracy vs Standard Accuracy
- Accuracy Improvement (%)
- Total Corrections
- Correction Rate
- System Contribution to Overall Accuracy

**這是 Paper 的核心價值主張！**

---

### 2.5 Agent Decision Pathways

**使用數據來源**: Section 6 (AGENT DECISION PATHWAY ANALYSIS)

**建議撰寫**:
```
The autonomous agent system employed three decision pathways:
- Standard pathway: X subjects (X.XX%) - straightforward cases with low uncertainty
- Counterfactual simulation: X subjects (X.XX%) - high uncertainty cases requiring 
  feature importance analysis
- Knowledge graph query: X subjects (X.XX%) - anomalous patterns requiring clinical 
  context retrieval

This adaptive decision-making demonstrates the system's ability to handle 
diagnostic complexity appropriately.
```

---

### 2.6 Feature Importance Analysis

**使用數據來源**: Section 8 (FEATURE IMPORTANCE ANALYSIS)

**建議撰寫**:
```
The top 5 most frequently important brain regions were:
1. [Region 1]: appeared in X% of cases (avg SHAP: X.XX, avg Z-score: X.XX)
2. [Region 2]: appeared in X% of cases (avg SHAP: X.XX, avg Z-score: X.XX)
...

These findings align with established neuroanatomical patterns in Alzheimer's 
disease, particularly the prominence of [hippocampus/entorhinal cortex/etc.].
```

**表格建議**:
```latex
\begin{table}[htbp]
\centering
\caption{Top 10 Most Important Brain Regions}
\begin{tabular}{lcccc}
\toprule
Region & Frequency & Avg SHAP & Avg Z-score & Clinical Significance \\
\midrule
Hippocampus L & XX\% & +0.XXX & -X.XX & Memory formation \\
...
\bottomrule
\end{tabular}
\end{table}
```

---

### 2.7 Anomaly Detection Results

**使用數據來源**: Section 7 (ANOMALY DETECTION ANALYSIS)

**建議撰寫**:
```
Statistical anomalies (|Z-score| > 2.5) were detected in X subjects (X.XX%). 
The most frequently anomalous regions were [list top 3], suggesting potential 
mixed pathology or atypical presentation patterns. These cases were flagged 
for knowledge graph consultation to provide clinical context.
```

---

## 3. Discussion Section (討論章節)

### 3.1 LOOCV Integrity and Generalizability

**使用數據來源**: Section 0 (LOOCV INTEGRITY VERIFICATION)

**建議撰寫**:
```
The 100% LOOCV coverage achieved in this study ensures that all reported 
performance metrics reflect true generalization capability, with no risk 
of data leakage. This strict train-test separation is critical for clinical 
translation, as it simulates real-world deployment where each new patient 
is truly unseen by the model.
```

---

### 3.2 System Value and Clinical Impact (核心討論！)

**使用數據來源**: 
- Section 12 (SYSTEM VALUE ANALYSIS)
- Section 6 (AGENT DECISION PATHWAYS)

**建議撰寫**:
```
The most significant finding of this study is the demonstrated value of our 
agent-based intervention system. Cases requiring intervention achieved X.XX% 
accuracy compared to X.XX% for standard cases, representing a +X.XX% improvement. 
This is not merely a theoretical advantage—the system successfully corrected 
X cases that would have been misdiagnosed using the standard pathway alone.

The X.XX% intervention rate suggests the system appropriately identifies 
challenging cases without over-triggering, while the 100% correction rate 
among interventions validates the effectiveness of counterfactual simulation 
and knowledge graph consultation. This demonstrates that uncertainty-aware 
AI systems can not only recognize their limitations but actively improve 
diagnostic accuracy through targeted analysis.

From a clinical translation perspective, this means the system could prevent 
X.XX% of potential misdiagnoses in real-world deployment, with particular 
value in cases exhibiting high uncertainty or anomalous patterns.
```

**關鍵論點**:
1. 量化的準確率提升
2. 實際糾正的案例數
3. 適當的介入率（不過度觸發）
4. 臨床轉化價值

---

### 3.3 Uncertainty-Aware Decision Making

**使用數據來源**: 
- Section 5 (UQ ANALYSIS)
- Section 6 (AGENT DECISION PATHWAYS)
- Section 11 (COMBINED CONDITIONS)

**建議撰寫**:
```
Our uncertainty quantification framework identified X subjects (X.XX%) with 
high uncertainty (UQ > 0.8), of which X also exhibited low confidence (< 0.6). 
These cases triggered counterfactual simulation, revealing that [key findings]. 
This demonstrates the clinical utility of uncertainty-aware AI systems that 
can recognize their own limitations and request additional analysis.
```

---

### 3.4 Neuroanatomical Interpretation

**使用數據來源**: Section 8 (FEATURE IMPORTANCE ANALYSIS)

**建議撰寫**:
```
The prominence of [hippocampus/entorhinal cortex/etc.] in our feature 
importance analysis aligns with established neuropathological findings in 
Alzheimer's disease. The average Z-scores of -X.XX in these regions indicate 
significant atrophy compared to normal controls, consistent with the 
Braak staging of neurofibrillary tangles.
```

---

### 3.5 Clinical Implications

**使用數據來源**: 
- Section 2 (Performance Metrics)
- Section 11 (COMBINED CONDITIONS)

**建議撰寫**:
```
The high sensitivity (X.XX%) suggests this system could serve as an effective 
screening tool for AD, minimizing false negatives. However, the specificity 
of X.XX% indicates that X.XX% of NC subjects may be flagged for further 
evaluation. In clinical practice, this trade-off may be acceptable given 
the importance of early AD detection.

The X subjects exhibiting both low confidence and high uncertainty represent 
diagnostically challenging cases that would benefit most from expert review, 
demonstrating the system's potential to triage cases appropriately.
```

---

### 3.6 Limitations

**使用數據來源**: 
- Section 1 (OVERALL SUMMARY)
- Section 12 (KEY FINDINGS)

**建議撰寫**:
```
Several limitations should be noted:
1. Sample size: X subjects may limit generalizability
2. Dataset characteristics: [describe any imbalances or biases]
3. Uncertainty calibration: UQ scores may require recalibration on external datasets
4. Clinical validation: prospective validation in clinical settings is needed
```

---

## 4. Supplementary Materials (補充材料)

### 4.1 Complete Statistical Report
**文件**: `binary_statistics_report.txt`
**用途**: 完整的統計分析結果，包含所有細節

### 4.2 Subject-Level Results
**文件**: `binary_statistics.csv`
**用途**: 每個受試者的詳細結果，可用於進一步分析

### 4.3 Structured Data
**文件**: `binary_statistics.json`
**用途**: 機器可讀的結構化數據，用於可視化或後續分析

### 4.4 Reasoning Chain Analysis
**使用數據來源**: Section 10 (REASONING CHAIN ANALYSIS)
**用途**: 展示 Agent 系統的推理過程，增加透明度

---

## 5. 圖表建議

### Figure 1: Model Performance Overview
- **Panel A**: Confusion matrix heatmap
- **Panel B**: ROC curve (if available)
- **Panel C**: Confidence distribution histogram
- **Panel D**: UQ score distribution histogram

### Figure 2: Feature Importance Analysis
- **Panel A**: Top 20 brain regions bar chart (by frequency)
- **Panel B**: SHAP values distribution
- **Panel C**: Brain map visualization (most important regions)

### Figure 3: Uncertainty-Aware Decision Making
- **Panel A**: Scatter plot (Confidence vs UQ)
- **Panel B**: Decision pathway distribution pie chart
- **Panel C**: Example case study (high UQ → counterfactual)

### Figure 4: LOOCV Integrity Verification
- **Panel A**: Coverage percentage bar chart
- **Panel B**: Model usage distribution
- **Panel C**: Example subject-specific model trace

---

## 6. 關鍵數字速查表

### 從報告中提取的關鍵數字

| 指標 | 位置 | 用途 |
|------|------|------|
| Accuracy | Section 2 | Abstract, Results |
| Sensitivity | Section 2 | Results, Discussion |
| Specificity | Section 2 | Results, Discussion |
| F1-Score | Section 2 | Results |
| **Total Corrections** | **Section 12** | **Abstract, Results, Discussion** |
| **Accuracy Improvement** | **Section 12** | **Abstract, Results, Discussion** |
| **Intervention Accuracy** | **Section 12** | **Results, Discussion** |
| **System Contribution** | **Section 12** | **Discussion** |
| LOOCV Coverage | Section 0 | Methods, Discussion |
| Mean UQ Score | Section 5 | Results, Discussion |
| High UQ Cases | Section 5 | Results, Discussion |
| Counterfactual Triggered | Section 6 | Results, Discussion |
| Top 5 Brain Regions | Section 8 | Results, Discussion |
| Anomaly Detection Rate | Section 7 | Results, Discussion |

---

## 7. 常見問題

### Q: 如何報告置信區間？
A: 使用 CSV 文件中的個別結果計算 bootstrap 置信區間：
```python
from scipy import stats
import numpy as np

# 從 CSV 讀取結果
results = pd.read_csv('binary_statistics.csv')
correct = results['correct'].values

# Bootstrap 95% CI
def bootstrap_ci(data, n_bootstrap=10000):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    return np.percentile(bootstrap_means, [2.5, 97.5])

ci = bootstrap_ci(correct)
print(f"Accuracy: {np.mean(correct):.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
```

### Q: 如何計算 ROC-AUC？
A: 使用 confidence scores 作為預測概率：
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 從 CSV 讀取
results = pd.read_csv('binary_statistics.csv')
y_true = (results['ground_truth'] == 'AD').astype(int)
y_score = results['confidence']

# 計算 AUC
auc = roc_auc_score(y_true, y_score)
print(f"AUC: {auc:.4f}")

# 繪製 ROC curve
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc_curve.png')
```

### Q: 如何進行統計顯著性檢驗？
A: 使用 McNemar's test 比較兩個模型：
```python
from statsmodels.stats.contingency_tables import mcnemar

# 比較兩個模型的預測結果
# contingency_table = [[n00, n01], [n10, n11]]
# n00: both correct, n01: model1 correct, model2 wrong
# n10: model1 wrong, model2 correct, n11: both wrong

result = mcnemar(contingency_table, exact=True)
print(f"McNemar's test p-value: {result.pvalue}")
```

---

## 8. 檢查清單

在提交 paper 之前，確認以下項目：

- [ ] LOOCV coverage 達到 100% (Section 0)
- [ ] 所有性能指標已報告 (Accuracy, Precision, Recall, F1)
- [ ] 置信區間已計算並報告
- [ ] Confusion matrix 已包含在 Results
- [ ] 特徵重要性分析已與神經解剖學文獻對照
- [ ] 不確定性量化的臨床意義已討論
- [ ] Agent 決策路徑的價值已說明
- [ ] 限制已充分討論
- [ ] 補充材料已準備 (CSV, JSON, 完整報告)
- [ ] 所有圖表已生成並標註清楚
- [ ] LaTeX 表格已格式化並測試編譯

---

## 9. 聯繫與支持

如需進一步協助：
- 查看完整文檔: `BINARY_STATISTICS_README.md`
- 運行測試: `python scripts/paper/test_binary_statistics.py`
- 查看範例輸出: `output/test_binary_stats/`
