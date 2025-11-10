# Quick Start: 24 ROIs vs 116 ROIs Comparison

## 為什麼要做這個比較？

你的模型目前使用 **24 個精選腦區**，表現非常好（100% 準確率）。但你想知道：

❓ **使用全部 116 個 AAL 腦區會不會更好？**

這個實驗會告訴你答案！

---

## 快速開始（3 步驟）

### Step 1: 提取所有 116 個 ROI 特徵

```bash
python scripts/ml/extract_all_roi_features.py
```

**這會做什麼？**
- 從所有 MRI 影像中提取 116 個 AAL 腦區的平均強度
- 儲存到 `data/processed/all_116_roi_features.csv`
- 大約需要 5-10 分鐘（取決於影像數量）

**預期輸出：**
```
Loading AAL atlas...
✓ Loaded AAL atlas from nilearn

Processing 42 NC subjects...
  Progress: 10/42
  Progress: 20/42
  ...

Processing 23 AD subjects...
  Progress: 10/23
  ...

Feature Extraction Complete!
Extracted features: 116 ROIs
Total subjects: 65
  NC: 42
  AD: 23

Saved to: data/processed/all_116_roi_features.csv
```

---

### Step 2: 比較不同特徵集

```bash
python scripts/ml/compare_real_features.py
```

**這會做什麼？**
- 訓練 5 種不同的模型：
  1. 24 個精選 ROIs（你目前的方法）
  2. 全部 116 個 ROIs（Random Forest）
  3. 全部 116 個 ROIs（L1 正則化）
  4. Top 30 ROIs（單變量選擇）
  5. Top 30 ROIs（互信息選擇）

- 使用 5-fold 交叉驗證評估每個模型
- 比較準確率、精確度、召回率、F1 分數
- 分析過擬合風險

**預期輸出：**
```
Comparing Different Approaches
================================================================================

1. Using 24 Selected ROIs (from original 24)
Training Selected_24_ROIs...
  CV Accuracy: 0.923 ± 0.045
  CV Precision: 0.920
  CV Recall: 0.915
  CV F1: 0.917
  Train Accuracy: 0.985
  Overfitting Gap: 0.062

2. Using All 116 ROIs (Random Forest)
Training All_116_ROIs_RF...
  CV Accuracy: 0.877 ± 0.067
  CV Precision: 0.865
  CV Recall: 0.870
  CV F1: 0.867
  Train Accuracy: 1.000
  Overfitting Gap: 0.123

3. Using All 116 ROIs (L1 Regularization)
Training All_116_ROIs_L1...
  CV Accuracy: 0.892 ± 0.052
  CV Precision: 0.885
  CV Recall: 0.880
  CV F1: 0.882
  Train Accuracy: 0.908
  Overfitting Gap: 0.016
  Features Selected: 28/116

...
```

---

### Step 3: 查看結果

結果會儲存在 `output/ml/real_feature_comparison/`：

1. **`real_feature_comparison.png`** - 視覺化比較圖表
   - CV 準確率比較
   - 訓練 vs 測試準確率
   - 過擬合分析
   - 特徵數量 vs 效能

2. **`real_feature_comparison_report.txt`** - 詳細報告
   - 效能摘要表
   - 過擬合分析
   - 關鍵發現
   - 建議

3. **`real_feature_comparison_summary.csv`** - 數據表格
   - 所有指標的 CSV 格式
   - 方便進一步分析

---

## 如何解讀結果？

### 情境 1: 24 ROIs 表現更好

```
Quick Summary:
--------------------------------------------------------------------------------
Selected_24_ROIs        : CV=0.923±0.045, F1=0.917, Gap=0.062
All_116_ROIs_RF         : CV=0.877±0.067, F1=0.867, Gap=0.123
```

**結論：保持使用 24 個精選 ROIs** ✅

**原因：**
- ✅ 更高的 CV 準確率
- ✅ 更低的過擬合風險（Gap < 0.1）
- ✅ 更好的臨床可解釋性
- ✅ 更適合小樣本（n=65）

---

### 情境 2: 116 ROIs 表現更好

```
Quick Summary:
--------------------------------------------------------------------------------
Selected_24_ROIs        : CV=0.846±0.078, F1=0.840, Gap=0.089
All_116_ROIs_L1         : CV=0.923±0.045, F1=0.920, Gap=0.032
```

**結論：考慮使用更多 ROIs，但要用正則化** ⚠️

**建議：**
- ⚠️ 使用 L1 正則化（自動特徵選擇）
- ⚠️ 或使用 Top 30-50 ROIs
- ⚠️ 驗證選出的 ROIs 是否有臨床意義
- ⚠️ 需要更多數據來確認（目標：200+ 樣本）

---

### 情境 3: 兩者差不多

```
Quick Summary:
--------------------------------------------------------------------------------
Selected_24_ROIs        : CV=0.908±0.052, F1=0.905, Gap=0.077
All_116_ROIs_RF         : CV=0.900±0.058, F1=0.897, Gap=0.100
```

**結論：保持 24 ROIs，但可以考慮集成** 💡

**建議：**
- 💡 保持 24 ROIs 作為主要模型（更簡單）
- 💡 可以訓練 116 ROIs 模型作為輔助
- 💡 使用集成方法結合兩者的預測
- 💡 在關鍵案例中參考兩個模型的結果

---

## 關鍵指標說明

### 1. CV Accuracy（交叉驗證準確率）
- **最重要的指標**
- 代表模型在未見過的數據上的表現
- 越高越好（但要注意過擬合）

### 2. Overfitting Gap（過擬合差距）
- Train Accuracy - CV Accuracy
- **越小越好**
- < 0.1: ✅ 良好
- 0.1-0.2: ⚠️ 中等
- > 0.2: ❌ 過擬合嚴重

### 3. F1 Score
- 精確度和召回率的調和平均
- 對於不平衡數據集很重要（NC:42, AD:23）
- 越高越好

### 4. Standard Deviation（標準差）
- CV 準確率的變異程度
- **越小越好**（表示模型穩定）
- < 0.05: ✅ 非常穩定
- 0.05-0.10: ⚠️ 中等穩定
- > 0.10: ❌ 不穩定

---

## 常見問題

### Q1: 為什麼 116 ROIs 的訓練準確率是 100% 但 CV 準確率只有 87%？

**A:** 這是**過擬合**的典型症狀！

- 模型在訓練數據上記住了所有樣本
- 但在新數據上表現不好
- 116 個特徵對 65 個樣本來說太多了

### Q2: L1 正則化選出了 28 個特徵，這些是哪些？

**A:** 查看報告中的 "Top Features" 部分，或：

```python
import pandas as pd
results = pd.read_csv('output/ml/real_feature_comparison/real_feature_comparison_summary.csv')
print(results)
```

### Q3: 如果 116 ROIs 表現更好，我應該重新訓練模型嗎？

**A:** 取決於：

1. **CV 準確率提升 > 5%**：值得考慮
2. **過擬合差距 < 0.1**：安全
3. **選出的特徵有臨床意義**：可信
4. **有外部驗證數據**：必須

**建議：**
- 先在當前數據上驗證
- 收集更多數據（目標：200+ 樣本）
- 在獨立測試集上驗證
- 然後再決定是否更換

### Q4: 我可以同時使用兩個模型嗎？

**A:** 可以！這叫做**模型集成**：

```python
# 簡單平均
pred_24 = model_24.predict_proba(X_24)
pred_116 = model_116.predict_proba(X_116)
pred_ensemble = (pred_24 + pred_116) / 2

# 加權平均（根據 CV 準確率）
weight_24 = 0.923 / (0.923 + 0.877)
weight_116 = 0.877 / (0.923 + 0.877)
pred_ensemble = weight_24 * pred_24 + weight_116 * pred_116
```

---

## 下一步

### 如果 24 ROIs 更好：

✅ **繼續使用當前模型**
- 已經驗證是最佳選擇
- 專注於收集更多數據
- 考慮加入其他模態（T2, DWI, fMRI）

### 如果 116 ROIs 更好：

⚠️ **謹慎採用**
1. 使用 L1 正則化或特徵選擇
2. 驗證選出的特徵有臨床意義
3. 在獨立數據集上測試
4. 收集更多數據來確認

### 如果差不多：

💡 **保持簡單**
- 使用 24 ROIs（更容易解釋）
- 可以訓練 116 ROIs 作為備用
- 在關鍵案例中參考兩個模型

---

## 總結

這個實驗幫助你：

1. ✅ **驗證特徵選擇**：24 個 ROIs 是否足夠？
2. ✅ **評估過擬合風險**：更多特徵是否導致過擬合？
3. ✅ **發現最佳方法**：哪種方法在你的數據上表現最好？
4. ✅ **做出明智決策**：基於數據而非猜測

**記住：更多特徵 ≠ 更好的模型**

在小樣本（n=65）的情況下，**簡單且有臨床意義的特徵集**通常比複雜的模型表現更好！

---

## 需要幫助？

如果遇到問題：

1. 檢查 `data/processed/all_116_roi_features.csv` 是否存在
2. 確認 AAL atlas 已正確載入
3. 查看錯誤訊息和 traceback
4. 參考 `docs/MODEL.md` 了解更多細節

Good luck! 🚀
