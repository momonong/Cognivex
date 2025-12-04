# CDDA Comprehensive Statistics Script 使用指南

這個腳本會自動掃描並分析所有可用的受試者，生成詳細的統計報告。

## 🎯 功能特色

### 完整的統計分析

#### 1. 基本統計
- ✅ 總受試者數
- ✅ 成功/失敗分析數
- ✅ 成功率

#### 2. 預測統計
- ✅ 預測分布 (AD/MCI/NC)
- ✅ 真實標籤分布
- ✅ 準確率 (總體和各類別)

#### 3. 信心度分析
- ✅ 信心度分布 (Very High/High/Medium/Low/Very Low)
- ✅ 低信心度受試者列表 (< 0.6)
- ✅ 低信心度百分比

#### 4. 不確定性分析
- ✅ UQ 分數分布
- ✅ 高不確定性受試者列表 (> 0.8)
- ✅ 高不確定性百分比

#### 5. Agent 決策分析
- ✅ 決策類型分布
- ✅ 反事實模擬觸發次數和百分比
- ✅ 知識圖譜查詢觸發次數和百分比
- ✅ 標準路徑次數

#### 6. 異常檢測分析
- ✅ 異常檢測率
- ✅ 異常受試者列表
- ✅ 最常見的異常區域
- ✅ 異常區域頻率統計

#### 7. 組合條件分析
- ✅ 低信心度 + 高不確定性
- ✅ 高信心度 + 高不確定性
- ✅ 低信心度 + 異常
- ✅ 高不確定性 + 異常

#### 8. 特徵重要性分析
- ✅ 最常出現的重要特徵 (Top 20)
- ✅ 平均 SHAP 值
- ✅ 平均 Z-score

#### 9. 性能分析
- ✅ 平均初始化時間
- ✅ 平均分析時間
- ✅ 平均總時間
- ✅ 吞吐量 (subjects/hour)
- ✅ 時間統計 (最小/最大)

#### 10. 推理鏈分析
- ✅ 平均推理步驟數
- ✅ 平均 Agent A 步驟數
- ✅ 平均 Agent B 步驟數
- ✅ 平均 MCP 動作數

#### 11. 錯誤與回退分析
- ✅ 錯誤列表
- ✅ 回退機制使用統計

#### 12. 關鍵發現總結
- ✅ 自動生成關鍵發現

## 🚀 使用方法

### 1. 基本用法 (分析所有受試者)

```bash
python scripts/comprehensive_statistics.py
```

這會：
- 自動掃描 `data/MRI_processed/` 目錄
- 分析所有找到的受試者
- 生成完整統計報告

### 2. 指定輸出目錄

```bash
python scripts/comprehensive_statistics.py --output output/my_statistics
```

### 3. 測試模式 (限制受試者數量)

```bash
# 只分析前 5 個受試者
python scripts/comprehensive_statistics.py --limit 5
```

### 4. 使用自定義模型路徑

```bash
python scripts/comprehensive_statistics.py \
    --orchestrator-path /path/to/phi-4-mini \
    --consultant-path /path/to/llama3.1-aloe
```

### 5. 使用規則模式 (不使用 LLM)

```bash
python scripts/comprehensive_statistics.py --no-llm
```

## 📂 輸出文件

運行後會在輸出目錄生成以下文件：

```
output/comprehensive_statistics/
├── comprehensive_statistics_report.txt    # 詳細文本報告
├── comprehensive_statistics.json          # JSON 格式數據
└── comprehensive_statistics.csv           # CSV 格式結果表
```

### 1. 文本報告 (comprehensive_statistics_report.txt)

完整的統計報告，包含：
- 總體概況
- 預測分布
- 準確性分析
- 信心度分析
- 不確定性分析
- Agent 決策分析
- 異常檢測分析
- 組合條件分析
- 特徵重要性分析
- 性能分析
- 推理鏈分析
- 錯誤與回退分析
- 關鍵發現總結

### 2. JSON 數據 (comprehensive_statistics.json)

結構化的統計數據，包含：
```json
{
  "statistics": {
    "total_subjects": 100,
    "successful_analyses": 98,
    "predictions": {"AD": 45, "MCI": 30, "NC": 23},
    "confidence_ranges": {...},
    "uq_ranges": {...},
    ...
  },
  "results": [
    {
      "subject_id": "sub-0001",
      "prediction": "AD",
      "confidence": 0.8523,
      "uq_score": 0.7234,
      ...
    },
    ...
  ]
}
```

### 3. CSV 結果表 (comprehensive_statistics.csv)

可用於 Excel 或 Python 分析：
```csv
subject_id,prediction,confidence,uq_score,agent_decision,ground_truth,correct,...
sub-0001,AD,0.8523,0.7234,STANDARD_REPORT,AD,True,...
sub-0002,MCI,0.7456,0.8123,SIMULATION_TRIGGERED,MCI,True,...
...
```

## 📊 報告示例

### 總體概況
```
================================================================================
1. OVERALL SUMMARY
================================================================================
Total Subjects Scanned: 100
Successful Analyses: 98
Failed Analyses: 2
Success Rate: 98.00%
```

### 預測分布
```
================================================================================
2. PREDICTION DISTRIBUTION
================================================================================
AD: 45 (45.92%)
MCI: 30 (30.61%)
NC: 23 (23.47%)
```

### 信心度分析
```
================================================================================
5. CONFIDENCE ANALYSIS
================================================================================
Confidence Distribution:
  Very High: 35 (35.71%)
  High: 28 (28.57%)
  Medium: 25 (25.51%)
  Low: 8 (8.16%)
  Very Low: 2 (2.04%)

Low Confidence Subjects (< 0.6): 10 (10.20%)

Low Confidence Details:
  - sub-0023: Confidence=0.5234, Prediction=MCI, GT=MCI
  - sub-0045: Confidence=0.4567, Prediction=AD, GT=NC
  ...
```

### Agent 決策分析
```
================================================================================
7. AGENT DECISION ANALYSIS
================================================================================
Decision Distribution:
  SIMULATION_TRIGGERED: 25 (25.51%)
  ANOMALY_INVESTIGATION: 18 (18.37%)
  STANDARD_REPORT: 55 (56.12%)

Counterfactual Simulation Triggered: 25 (25.51%)
Knowledge Graph Query Triggered: 18 (18.37%)
Standard Pathway: 55 (56.12%)
```

### 特徵重要性分析
```
================================================================================
10. FEATURE IMPORTANCE ANALYSIS
================================================================================
Most Frequently Important Features (Top 20):
   1. Hippocampus_L: 85 times (86.73%)
      Avg SHAP: +0.145623, Avg Z-score: -2.3456
   2. Hippocampus_R: 82 times (83.67%)
      Avg SHAP: +0.132456, Avg Z-score: -2.1234
   3. Entorhinal_L: 78 times (79.59%)
      Avg SHAP: +0.098765, Avg Z-score: -1.9876
  ...
```

## 📝 論文撰寫建議

### 1. 實驗設置章節

使用報告中的：
- 總體概況 (受試者數量)
- 真實標籤分布

```markdown
### Experimental Setup

We evaluated the CDDA framework on a dataset of N subjects, 
comprising X AD cases, Y MCI cases, and Z NC cases.
```

### 2. 結果章節

使用報告中的：
- 預測分布
- 準確性分析
- Agent 決策分析

```markdown
### Overall Performance

The system achieved an overall accuracy of X%, with class-specific 
accuracies of Y% for AD, Z% for MCI, and W% for NC.

The adaptive decision mechanism triggered counterfactual simulation 
in X% of cases and knowledge graph queries in Y% of cases.
```

### 3. 不確定性分析章節

使用報告中的：
- 不確定性分析
- 組合條件分析

```markdown
### Uncertainty Quantification

High uncertainty (UQ > 0.8) was detected in X% of cases. 
Among these, Y cases also showed low confidence (< 0.6), 
indicating challenging diagnostic scenarios.
```

### 4. 可解釋性章節

使用報告中的：
- 特徵重要性分析
- 異常檢測分析

```markdown
### Feature Importance and Explainability

The most frequently important features were:
1. Hippocampus (bilateral): appeared in X% of cases
2. Entorhinal cortex: appeared in Y% of cases
3. Amygdala: appeared in Z% of cases

Anomalies were detected in W% of subjects, with the most 
frequently anomalous regions being...
```

### 5. 性能分析章節

使用報告中的：
- 性能分析

```markdown
### Computational Performance

The system demonstrated efficient processing with:
- Average analysis time: X seconds per subject
- Throughput: Y subjects per hour
- Suitable for clinical deployment
```

## 🔍 進階分析

### 使用 Python 分析 JSON 數據

```python
import json
import pandas as pd

# 讀取 JSON 數據
with open('output/comprehensive_statistics/comprehensive_statistics.json', 'r') as f:
    data = json.load(f)

# 轉換為 DataFrame
df = pd.DataFrame(data['results'])

# 分析低信心度案例
low_conf = df[df['confidence'] < 0.6]
print(f"Low confidence cases: {len(low_conf)}")
print(low_conf[['subject_id', 'prediction', 'confidence', 'ground_truth']])

# 分析高不確定性案例
high_uq = df[df['uq_score'] > 0.8]
print(f"High uncertainty cases: {len(high_uq)}")

# 分析錯誤預測
if 'correct' in df.columns:
    incorrect = df[df['correct'] == False]
    print(f"Incorrect predictions: {len(incorrect)}")
    print(incorrect[['subject_id', 'prediction', 'ground_truth', 'confidence']])
```

### 使用 CSV 數據生成圖表

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV
df = pd.read_csv('output/comprehensive_statistics/comprehensive_statistics.csv')

# 繪製信心度分布
plt.figure(figsize=(10, 6))
plt.hist(df['confidence'], bins=20, edgecolor='black')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.title('Confidence Distribution')
plt.savefig('confidence_distribution.png')

# 繪製 UQ 分布
plt.figure(figsize=(10, 6))
plt.hist(df['uq_score'], bins=20, edgecolor='black')
plt.xlabel('UQ Score')
plt.ylabel('Frequency')
plt.title('Uncertainty Distribution')
plt.savefig('uq_distribution.png')

# 繪製信心度 vs UQ 散點圖
plt.figure(figsize=(10, 6))
plt.scatter(df['confidence'], df['uq_score'], alpha=0.5)
plt.xlabel('Confidence')
plt.ylabel('UQ Score')
plt.title('Confidence vs Uncertainty')
plt.axhline(y=0.8, color='r', linestyle='--', label='UQ Threshold')
plt.axvline(x=0.6, color='orange', linestyle='--', label='Low Confidence')
plt.legend()
plt.savefig('confidence_vs_uq.png')
```

## 🎯 常見問題

### Q1: 分析時間太長怎麼辦？

**A**: 使用 `--limit` 參數先測試少量受試者：
```bash
python scripts/comprehensive_statistics.py --limit 10
```

### Q2: 如何只分析特定類別的受試者？

**A**: 修改腳本中的 `scan_all_subjects()` 函數，添加過濾邏輯。

### Q3: 如何導出特定統計數據？

**A**: 使用 JSON 文件，用 Python 提取需要的數據：
```python
import json

with open('comprehensive_statistics.json', 'r') as f:
    data = json.load(f)

# 提取低信心度受試者
low_conf = data['statistics']['low_confidence_subjects']
print(json.dumps(low_conf, indent=2))
```

### Q4: 如何比較不同配置的統計結果？

**A**: 使用不同輸出目錄運行多次：
```bash
# 配置 1: LLM 模式
python scripts/comprehensive_statistics.py --output output/stats_llm

# 配置 2: 規則模式
python scripts/comprehensive_statistics.py --output output/stats_rules --no-llm

# 比較結果
diff output/stats_llm/comprehensive_statistics_report.txt \
     output/stats_rules/comprehensive_statistics_report.txt
```

### Q5: 記憶體不足怎麼辦？

**A**: 
1. 確保使用 4-bit 量化 (默認啟用)
2. 使用 `--limit` 分批處理
3. 關閉其他程序釋放記憶體

## 📧 支持

如果遇到問題，請檢查：
1. 數據目錄結構是否正確
2. 模型是否正確加載
3. GPU 記憶體是否充足

---

**祝統計分析順利！** 📊✨
