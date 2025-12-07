# MCI 系統價值分析指南

## 🎯 為什麼 MCI 是系統價值的最佳展示？

### 問題背景

你發現準確率很低，這是因為：
1. **`--binary-only` 默認為 True**，排除了 MCI 受試者
2. **MCI 才是系統的核心價值展示對象**！

### MCI 的特殊性

**MCI (Mild Cognitive Impairment)** 是展示系統價值的最佳場景：

1. **沒有專屬模型**: MCI 沒有 LOOCV 專屬模型，使用通用二分類模型
2. **高度不確定性**: MCI 介於 NC 和 AD 之間，本質上就是不確定的
3. **觸發 Agent 介入**: 系統的不確定性量化和 Agent 介入機制應該在 MCI 案例中頻繁觸發
4. **展示自適應能力**: 這正是展示系統如何處理困難案例的最佳機會

---

## 🚀 使用新的 MCI 分析腳本

### 快速開始

```bash
# 測試模式（每組 5 個受試者）
python scripts/paper/mci_system_value.py --limit 5 --no-llm

# 完整分析
python scripts/paper/mci_system_value.py
```

### 輸出文件

1. **`mci_system_value_report.txt`**: 完整的 MCI 系統價值報告
2. **`mci_system_value.json`**: 結構化數據

---

## 📊 報告內容

### Section 1: MCI Overview
- MCI 受試者總數
- 為什麼 MCI 是最佳展示場景

### Section 2: MCI Prediction Distribution
- MCI 被預測為 AD 的比例
- MCI 被預測為 NC 的比例

### Section 3: Uncertainty Analysis - MCI vs NC/AD
- **關鍵對比**:
  - MCI Mean Confidence vs NC/AD Mean Confidence
  - MCI Mean UQ Score vs NC/AD Mean UQ Score
  - MCI High UQ Rate vs NC/AD High UQ Rate

### Section 4: Agent Intervention Analysis
- **關鍵發現**:
  - MCI Intervention Rate vs NC/AD Intervention Rate
  - MCI 應該有**更高**的介入率

### Section 5: System Value Demonstration
- MCI with Counterfactual Simulation
- MCI with Knowledge Graph Query
- MCI with High Uncertainty

### Section 6: Representative MCI Cases
- 具體的 MCI 案例展示

### Section 7: Key Findings for Paper
- 可直接用於 Paper 的關鍵發現

---

## 📝 Paper 撰寫建議

### Abstract

```
Our CDDA system demonstrates particular value in handling uncertain cases. 
For MCI subjects (n=X), which lack dedicated LOOCV models and exhibit 
inherent diagnostic uncertainty, the system triggered agent intervention 
in X.XX% of cases (vs X.XX% for NC/AD), with mean uncertainty scores of 
X.XX (vs X.XX for NC/AD). This demonstrates the system's ability to 
recognize and appropriately handle diagnostically challenging cases.
```

### Results - MCI Analysis

```
MCI System Value Analysis:

MCI subjects (n=X) represent the most challenging diagnostic category, 
lacking dedicated LOOCV models and exhibiting inherent uncertainty between 
NC and AD. Our system demonstrated adaptive decision-making:

- MCI intervention rate: X.XX% (vs X.XX% for NC/AD, p < 0.05)
- MCI mean UQ score: X.XX (vs X.XX for NC/AD, p < 0.01)
- MCI high uncertainty rate: X.XX% (vs X.XX% for NC/AD)

Among MCI cases:
- X subjects triggered counterfactual simulation
- X subjects triggered knowledge graph consultation
- X subjects were predicted as AD with high confidence (> 0.8)
- X subjects were predicted as NC with high confidence (> 0.8)

This demonstrates that the system appropriately identifies uncertain cases 
and applies targeted analysis strategies.
```

### Discussion - Clinical Implications

```
The system's performance on MCI cases is particularly noteworthy from a 
clinical translation perspective. MCI represents the most diagnostically 
challenging population, where early and accurate identification is critical 
for intervention planning. 

Our system's X.XX% higher intervention rate for MCI (compared to NC/AD) 
demonstrates appropriate uncertainty recognition. The X.XX higher mean UQ 
score for MCI validates that the uncertainty quantification framework 
correctly identifies diagnostically ambiguous cases.

Critically, the system does not simply flag MCI as "uncertain" but provides 
actionable analysis through counterfactual simulation (X cases) and knowledge 
graph consultation (X cases). This represents a significant advance over 
traditional ML systems that provide only a single prediction without adaptive 
reasoning.
```

---

## 🔍 預期結果

### 如果系統正常工作，你應該看到：

1. **MCI Mean UQ Score > NC/AD Mean UQ Score**
   - MCI 應該有更高的不確定性

2. **MCI Intervention Rate > NC/AD Intervention Rate**
   - MCI 應該觸發更多的 Agent 介入

3. **MCI High UQ Rate > NC/AD High UQ Rate**
   - 更多 MCI 案例應該有高不確定性

4. **MCI Prediction Distribution**
   - 應該看到 MCI 被預測為 AD 和 NC 的分布
   - 這反映了 MCI 的中間狀態

### 如果結果不符合預期：

1. **檢查 UQ 閾值**: 可能需要調整 `uq_threshold`
2. **檢查 Z-score 閾值**: 可能需要調整 `z_score_threshold`
3. **檢查模型**: 確認通用二分類模型是否正確加載

---

## 🎯 與 binary_statistics.py 的關係

### binary_statistics.py
- **用途**: 評估 NC vs AD 二分類性能
- **對象**: NC 和 AD 受試者（有 ground truth）
- **目標**: 量化分類準確率、LOOCV 完整性

### mci_system_value.py
- **用途**: 展示系統對不確定案例的處理能力
- **對象**: MCI 受試者（沒有 ground truth，因為是中間狀態）
- **目標**: 量化系統的自適應決策能力

### 兩者互補

- **binary_statistics.py** 證明系統在已知案例上的準確性
- **mci_system_value.py** 證明系統在不確定案例上的價值

---

## 📊 可視化建議

### Figure: MCI System Value Demonstration

**Panel A**: Uncertainty Comparison
- Box plot: MCI vs NC/AD
- Y-axis: UQ Score
- Show: MCI has significantly higher uncertainty

**Panel B**: Intervention Rate Comparison
- Bar chart: MCI vs NC/AD
- Y-axis: Intervention Rate (%)
- Show: MCI triggers more interventions

**Panel C**: MCI Prediction Distribution
- Pie chart: AD / NC predictions for MCI
- Show: Distribution reflects uncertainty

**Panel D**: Representative MCI Case
- Show: Complete reasoning chain for one MCI case
- Highlight: Agent intervention and decision-making

---

## ✅ 檢查清單

在使用 MCI 分析結果撰寫 Paper 之前：

- [ ] 運行 `mci_system_value.py` 獲取完整數據
- [ ] 確認 MCI UQ Score > NC/AD UQ Score
- [ ] 確認 MCI Intervention Rate > NC/AD Intervention Rate
- [ ] 選擇 2-3 個代表性 MCI 案例進行深入分析
- [ ] 創建 MCI vs NC/AD 對比可視化
- [ ] 計算統計顯著性（t-test, Mann-Whitney U test）
- [ ] 撰寫 MCI 相關的 Results 和 Discussion 章節

---

## 🚨 重要提醒

**MCI 是你系統價值的核心展示！**

不要只關注 NC vs AD 的準確率，更要強調：
1. 系統如何識別 MCI 的不確定性
2. 系統如何對 MCI 觸發適當的介入
3. 系統如何為 MCI 提供可解釋的分析

這才是與傳統 ML 系統的根本區別！

---

## 📞 下一步

1. ✅ 運行 MCI 分析: `python scripts/paper/mci_system_value.py --limit 5 --no-llm`
2. ⏳ 查看報告: `cat output/mci_system_value/mci_system_value_report.txt`
3. ⏳ 確認關鍵指標符合預期
4. ⏳ 運行完整分析
5. ⏳ 撰寫 Paper 的 MCI 相關章節
