# Binary Statistics - 快速開始指南

## 🚀 5 分鐘快速開始

### Step 1: 測試腳本 (30 秒)
```bash
# 運行測試確保一切正常
python scripts/paper/test_binary_statistics.py
```

**預期輸出**: `✓ ALL TESTS PASSED!`

---

### Step 2: 小規模測試 (2 分鐘)
```bash
# 分析 10 個受試者 (測試模式)
python scripts/paper/binary_statistics.py --limit 10 --no-llm
```

**預期輸出**: 
- `output/binary_statistics/binary_statistics_report.txt`
- `output/binary_statistics/binary_statistics.json`
- `output/binary_statistics/binary_statistics.csv`
- `output/binary_statistics/binary_performance_table.tex`

---

### Step 3: 查看報告 (1 分鐘)
```bash
# Windows
type output\binary_statistics\binary_statistics_report.txt

# Linux/Mac
cat output/binary_statistics/binary_statistics_report.txt
```

**關鍵檢查點**:
- Section 0: LOOCV Coverage 應該是 100%
- Section 2: 查看 Accuracy, Precision, Recall
- Section 12: 查看關鍵發現總結

---

### Step 4: 完整分析 (視數據集大小而定)
```bash
# 分析所有 NC/AD 受試者
python scripts/paper/binary_statistics.py
```

---

## 📊 輸出文件說明

### 1. 文字報告 (`binary_statistics_report.txt`)
- **用途**: 完整的統計分析結果
- **章節**: 0-12 個主要章節
- **適用於**: 快速查看、補充材料

### 2. JSON 數據 (`binary_statistics.json`)
- **用途**: 結構化數據
- **內容**: 所有統計指標 + 每個受試者的詳細結果
- **適用於**: 進一步分析、可視化

### 3. CSV 表格 (`binary_statistics.csv`)
- **用途**: 受試者級別的結果
- **欄位**: Subject ID, Prediction, Confidence, UQ Score, Ground Truth, Model Used, etc.
- **適用於**: Excel 分析、統計檢驗

### 4. LaTeX 表格 (`binary_performance_table.tex`)
- **用途**: Paper-ready 性能表格
- **格式**: 標準 LaTeX table 環境
- **適用於**: 直接插入 Paper

---

## 📝 Paper 撰寫流程

### 1. Methods Section
**使用**: Section 0 (LOOCV 驗證)

```latex
We employed Leave-One-Out Cross-Validation (LOOCV) with strict 
train-test separation. Our analysis achieved 100% LOOCV coverage 
(X/X subjects), ensuring no data leakage.
```

### 2. Results Section
**使用**: Section 2 (性能指標) + LaTeX 表格

```latex
The binary classification model achieved an accuracy of X.XX% 
(Table 1). Sensitivity for AD detection was X.XX%, while 
specificity for NC identification was X.XX%.

\input{binary_performance_table.tex}
```

### 3. Discussion Section
**使用**: Section 5 (UQ 分析) + Section 6 (Agent 決策)

```latex
Our uncertainty quantification framework identified X subjects 
(X.XX%) with high uncertainty (UQ > 0.8), triggering counterfactual 
analysis. This demonstrates the clinical utility of uncertainty-aware 
AI systems.
```

---

## 🔍 關鍵指標速查

從報告中提取這些數字用於 Paper：

| 指標 | 位置 | 用於 |
|------|------|------|
| **Accuracy** | Section 2 | Abstract, Results |
| **Sensitivity** | Section 2 | Results, Discussion |
| **Specificity** | Section 2 | Results, Discussion |
| **F1-Score** | Section 2 | Results |
| **🌟 Total Corrections** | **Section 12** | **Abstract, Results** |
| **🌟 Accuracy Improvement** | **Section 12** | **Abstract, Results** |
| **🌟 System Contribution** | **Section 12** | **Discussion** |
| **LOOCV Coverage** | Section 0 | Methods, Discussion |
| **Mean UQ Score** | Section 5 | Results, Discussion |
| **High UQ Cases** | Section 5 | Results, Discussion |
| **Top 5 Brain Regions** | Section 8 | Results, Discussion |

---

## ⚠️ 常見問題

### Q: LOOCV Coverage 不是 100%？
**A**: 檢查 `model/loocv_models_binary_opt/` 是否包含所有受試者的模型。

### Q: 如何只分析 NC 和 AD？
**A**: 使用 `--binary-only` 參數（默認已啟用）。

### Q: 如何加速分析？
**A**: 使用 `--no-llm` 參數禁用 LLM，使用規則決策。

### Q: 如何生成可視化？
**A**: 使用 CSV 文件配合 Python/R 繪圖庫：
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 讀取結果
df = pd.read_csv('output/binary_statistics/binary_statistics.csv')

# 繪製混淆矩陣
y_true = df['ground_truth']
y_pred = df['prediction']
cm = confusion_matrix(y_true, y_pred, labels=['AD', 'NC'])
disp = ConfusionMatrixDisplay(cm, display_labels=['AD', 'NC'])
disp.plot()
plt.savefig('confusion_matrix.png')
```

---

## 📚 進階使用

### 計算置信區間
```python
import numpy as np
import pandas as pd

df = pd.read_csv('output/binary_statistics/binary_statistics.csv')
correct = df['correct'].values

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

### 計算 ROC-AUC
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv('output/binary_statistics/binary_statistics.csv')
y_true = (df['ground_truth'] == 'AD').astype(int)
y_score = df['confidence']

auc = roc_auc_score(y_true, y_score)
fpr, tpr, _ = roc_curve(y_true, y_score)

plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc_curve.png')
```

---

## 📖 完整文檔

- **使用說明**: `BINARY_STATISTICS_README.md`
- **Paper 指南**: `PAPER_WRITING_GUIDE.md`
- **測試腳本**: `test_binary_statistics.py`
- **範例輸出**: `output/test_binary_stats/`

---

## ✅ 檢查清單

在提交 Paper 之前：

- [ ] 運行完整分析 (`python scripts/paper/binary_statistics.py`)
- [ ] LOOCV Coverage = 100%
- [ ] 所有輸出文件已生成
- [ ] LaTeX 表格可正常編譯
- [ ] 已計算置信區間
- [ ] 已生成 ROC curve
- [ ] 已生成 confusion matrix
- [ ] 已生成 feature importance 圖表
- [ ] 已與文獻對照特徵重要性
- [ ] 已準備補充材料

---

## 🎯 下一步

1. ✅ 運行測試: `python scripts/paper/test_binary_statistics.py`
2. ✅ 小規模測試: `python scripts/paper/binary_statistics.py --limit 10`
3. ⏳ 完整分析: `python scripts/paper/binary_statistics.py`
4. ⏳ 生成可視化
5. ⏳ 撰寫 Paper

---

**祝你順利完成 Paper！** 🎉
