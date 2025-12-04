# CDDA Paper Analysis - 完整使用範例

這份文件提供完整的使用範例，從安裝到生成論文圖表。

## 📋 目錄

1. [環境準備](#環境準備)
2. [快速測試](#快速測試)
3. [單個受試者分析](#單個受試者分析)
4. [批量分析](#批量分析)
5. [生成可視化圖表](#生成可視化圖表)
6. [論文撰寫建議](#論文撰寫建議)

---

## 1. 環境準備

### 1.1 安裝依賴

```bash
# 基本依賴
pip install -r requirements.txt

# 可視化依賴 (可選)
pip install matplotlib pandas scikit-learn
```

### 1.2 下載模型

```bash
# 下載 Phi-4-mini (Agent A)
python scripts/download_models.py --model phi-4-mini --output D:/hf_models/Phi-4-mini-instruct

# 下載 Llama3.1-Aloe-Beta-8B (Agent B)
python scripts/download_models.py --model llama3.1-aloe-beta-8b --output D:/hf_models/Llama3.1-Aloe-Beta-8B
```

### 1.3 準備數據

確保數據結構如下：

```
data/MRI_processed/
├── AD/
│   ├── sub-0001/
│   │   ├── sub-0001_T1w.nii.gz
│   │   ├── sub-0001_GM.nii.gz
│   │   └── sub-0001_WM.nii.gz
│   └── sub-0002/
├── MCI/
│   └── sub-0003/
└── NC/
    └── sub-0005/
```

---

## 2. 快速測試

在進行完整分析前，先運行快速測試確保系統正常：

```bash
python scripts/quick_paper_test.py
```

這會分析 2 個受試者 (sub-0005, sub-0015)，結果保存在 `output/quick_test/`。

**預期輸出**:
```
================================================================================
CDDA Quick Paper Test
================================================================================

Test subjects: sub-0005, sub-0015

Running command:
python scripts/paper_analysis.py --subjects sub-0005 sub-0015 --output output/quick_test --use-4bit

================================================================================

[系統信息]
[初始化過程]
[分析過程]
...

================================================================================
Quick test completed successfully!
================================================================================

Check the results in: output/quick_test/
```

---

## 3. 單個受試者分析

### 3.1 基本用法

```bash
python scripts/paper_analysis.py --subject sub-0005
```

### 3.2 指定輸出目錄

```bash
python scripts/paper_analysis.py \
    --subject sub-0005 \
    --output output/case_study_sub0005
```

### 3.3 提供真實標籤

創建 `ground_truth.json`:
```json
{
  "sub-0005": "AD"
}
```

運行：
```bash
python scripts/paper_analysis.py \
    --subject sub-0005 \
    --ground-truth-file ground_truth.json
```

### 3.4 查看結果

分析完成後，檢查輸出目錄：

```bash
# Windows
explorer output\paper_results

# Linux/Mac
open output/paper_results
```

**生成的文件**:
- `result_sub-0005_*.json` - 完整結果
- `reports/report_sub-0005_*.md` - 臨床報告
- `reasoning_chains/reasoning_sub-0005_*.txt` - 推理鏈
- `metrics/features_sub-0005_*.csv` - 特徵重要性
- `logs/analysis_log_*.txt` - 執行日誌

---

## 4. 批量分析

### 4.1 分析多個指定受試者

```bash
python scripts/paper_analysis.py \
    --subjects sub-0001 sub-0002 sub-0003 sub-0005 sub-0010 \
    --output output/batch_analysis_5subjects
```

### 4.2 分析所有受試者

```bash
# 不指定 --subject 或 --subjects，會自動掃描所有受試者
python scripts/paper_analysis.py \
    --output output/full_dataset_analysis
```

### 4.3 使用真實標籤文件

創建完整的 `ground_truth.json`:
```json
{
  "sub-0001": "AD",
  "sub-0002": "AD",
  "sub-0003": "MCI",
  "sub-0005": "AD",
  "sub-0010": "NC",
  "sub-0015": "NC"
}
```

運行：
```bash
python scripts/paper_analysis.py \
    --subjects sub-0001 sub-0002 sub-0003 sub-0005 sub-0010 sub-0015 \
    --ground-truth-file ground_truth.json \
    --output output/batch_with_gt
```

### 4.4 查看總結報告

批量分析會自動生成總結報告：

```bash
# 查看 Markdown 總結
cat output/batch_analysis_5subjects/analysis_summary_*.md

# 或在瀏覽器中打開
# Windows: start output/batch_analysis_5subjects/analysis_summary_*.md
```

**總結報告包含**:
- 預測分布表格
- Agent 決策分布
- 平均信心度和不確定性
- 準確率 (如果有真實標籤)
- 性能統計
- 個別結果表格

---

## 5. 生成可視化圖表

### 5.1 基本用法

```bash
python scripts/visualize_results.py --input output/batch_analysis_5subjects
```

### 5.2 指定輸出目錄

```bash
python scripts/visualize_results.py \
    --input output/batch_analysis_5subjects \
    --output output/paper_figures
```

### 5.3 生成的圖表

運行後會生成以下圖表：

1. **prediction_distribution.png**
   - 預測類別分布條形圖
   - 用於展示系統在不同類別的預測分布

2. **confidence_vs_uncertainty.png**
   - 信心度 vs 不確定性散點圖
   - 按預測類別著色
   - 顯示 UQ 閾值線

3. **agent_decision_distribution.png**
   - Agent 決策分布條形圖
   - 展示觸發不同決策路徑的頻率

4. **performance_metrics.png**
   - 性能指標圖 (箱型圖 + 條形圖)
   - 展示初始化時間、分析時間分布

5. **top_features_shap.png**
   - 前 10 個最重要特徵的 SHAP 值
   - 水平條形圖

6. **confusion_matrix.png** (如果有真實標籤)
   - 混淆矩陣熱圖
   - 展示預測準確性

---

## 6. 論文撰寫建議

### 6.1 方法論章節

**使用文件**:
- `logs/analysis_log_*.txt` - 系統配置和初始化過程
- `reasoning_chains/reasoning_*.txt` - Agent 推理過程

**建議內容**:
```markdown
### 3.1 System Architecture

The CDDA framework consists of:
- Agent A (Orchestrator): Phi-4-mini for decision making
- Agent B (Consultant): Llama3.1-Aloe-Beta-8B for clinical synthesis
- MCP Server: Resource and tool provider
- ToolKit: CNN-RF model + SHAP + UQ + Anomaly Detection

[插入系統架構圖]

### 3.2 Agent Reasoning Process

Agent A evaluates diagnostic signals and decides which tools to invoke:
- If UQ > 0.8 → Trigger counterfactual simulation
- If anomaly detected → Query knowledge graph
- Otherwise → Standard report

[插入推理鏈範例]
```

### 6.2 實驗結果章節

**使用文件**:
- `analysis_summary_*.md` - 統計總結
- `visualizations/prediction_distribution.png` - 預測分布圖
- `visualizations/confusion_matrix.png` - 混淆矩陣

**建議內容**:
```markdown
### 4.1 Overall Performance

We analyzed N subjects from the dataset. The prediction distribution is shown in Figure X.

[插入 prediction_distribution.png]

The system achieved an overall accuracy of X% (Table 1).

| Metric | Value |
|--------|-------|
| Accuracy | X% |
| Average Confidence | X.XXX |
| Average Uncertainty | X.XXX |

[插入 confusion_matrix.png]

### 4.2 Agent Decision Analysis

The agent decision distribution (Figure Y) shows that:
- X% cases triggered counterfactual simulation (high uncertainty)
- Y% cases triggered knowledge graph query (anomaly detected)
- Z% cases followed standard pathway

[插入 agent_decision_distribution.png]
```

### 6.3 案例研究章節

**使用文件**:
- `reports/report_sub-XXXX_*.md` - 臨床報告
- `metrics/features_sub-XXXX_*.csv` - 特徵重要性
- `reasoning_chains/reasoning_sub-XXXX_*.txt` - 推理鏈

**建議內容**:
```markdown
### 4.3 Case Study: Subject sub-0005

#### Diagnostic Results
- Prediction: AD
- Confidence: 0.8523
- Uncertainty: 0.8234
- Decision: SIMULATION_TRIGGERED

#### Feature Importance
The top contributing features were:
1. Hippocampus_L (SHAP: 0.1523, Z-score: -2.85)
2. Hippocampus_R (SHAP: 0.1234, Z-score: -2.67)
3. Entorhinal_L (SHAP: 0.0987, Z-score: -2.34)

[插入特徵重要性表格或圖表]

#### Counterfactual Analysis
When masking the top 3 features, the prediction changed from AD (85.2%) to NC (45.3%), 
indicating these regions are key diagnostic drivers.

#### Clinical Report
[插入部分臨床報告內容]
```

### 6.4 可解釋性章節

**使用文件**:
- `visualizations/top_features_shap.png` - SHAP 特徵重要性
- `reasoning_chains/reasoning_*.txt` - 完整推理鏈

**建議內容**:
```markdown
### 4.4 Explainability Analysis

#### Feature Importance
Figure Z shows the average SHAP values for the top 10 most important features across all subjects.

[插入 top_features_shap.png]

The most influential regions are:
- Hippocampus (bilateral): Memory formation
- Entorhinal cortex: Early AD marker
- Amygdala: Emotional processing

#### Reasoning Transparency
The complete reasoning chain provides full transparency:
1. Agent A reads diagnostic report
2. Agent A evaluates UQ score (0.82 > 0.8)
3. Agent A triggers counterfactual simulation
4. Agent A compiles ContextObject
5. Agent B receives context
6. Agent B generates clinical report

[插入推理鏈片段]
```

### 6.5 性能分析章節

**使用文件**:
- `visualizations/performance_metrics.png` - 性能圖表
- `analysis_summary_*.md` - 性能統計

**建議內容**:
```markdown
### 4.5 Performance Analysis

Figure W shows the time distribution for initialization and analysis phases.

[插入 performance_metrics.png]

Performance metrics:
- Average initialization time: X.XX seconds
- Average analysis time: Y.YY seconds
- Throughput: ZZ subjects/hour

The system demonstrates efficient processing suitable for clinical deployment.
```

---

## 7. 完整工作流程範例

以下是一個完整的論文準備工作流程：

```bash
# Step 1: 快速測試
python scripts/quick_paper_test.py

# Step 2: 準備真實標籤文件
cat > ground_truth.json << EOF
{
  "sub-0001": "AD",
  "sub-0002": "AD",
  "sub-0003": "MCI",
  "sub-0005": "AD",
  "sub-0010": "NC",
  "sub-0015": "NC"
}
EOF

# Step 3: 批量分析
python scripts/paper_analysis.py \
    --subjects sub-0001 sub-0002 sub-0003 sub-0005 sub-0010 sub-0015 \
    --ground-truth-file ground_truth.json \
    --output output/paper_final_results

# Step 4: 生成可視化圖表
python scripts/visualize_results.py \
    --input output/paper_final_results \
    --output output/paper_figures

# Step 5: 查看結果
# Windows
explorer output\paper_final_results
explorer output\paper_figures

# Linux/Mac
open output/paper_final_results
open output/paper_figures

# Step 6: 查看總結報告
cat output/paper_final_results/analysis_summary_*.md
```

---

## 8. 常見問題

### Q1: 分析時間太長怎麼辦？

**A**: 使用 4-bit 量化和減少受試者數量：
```bash
python scripts/paper_analysis.py \
    --subjects sub-0001 sub-0002 \
    --use-4bit
```

### Q2: GPU 記憶體不足？

**A**: 確保使用 4-bit 量化，並關閉其他 GPU 程序：
```bash
# 檢查 GPU 使用情況
nvidia-smi

# 使用 4-bit 量化
python scripts/paper_analysis.py --subject sub-0005 --use-4bit
```

### Q3: 如何只生成特定圖表？

**A**: 修改 `visualize_results.py`，註釋掉不需要的圖表函數。

### Q4: 如何導出 LaTeX 表格？

**A**: 使用 pandas 讀取 CSV 並轉換：
```python
import pandas as pd

df = pd.read_csv('output/paper_results/metrics/features_sub-0005_*.csv')
print(df.head(10).to_latex(index=False))
```

### Q5: 如何比較不同配置？

**A**: 使用不同輸出目錄：
```bash
# 配置 1: LLM 模式
python scripts/paper_analysis.py --subjects sub-0001 sub-0002 \
    --output output/results_llm

# 配置 2: 規則模式
python scripts/paper_analysis.py --subjects sub-0001 sub-0002 \
    --output output/results_rules --no-llm

# 比較結果
diff output/results_llm/analysis_summary_*.md \
     output/results_rules/analysis_summary_*.md
```

---

## 9. 進階技巧

### 9.1 自動化批量處理

創建 `run_all_analyses.sh`:
```bash
#!/bin/bash

# 定義受試者組
AD_SUBJECTS="sub-0001 sub-0002 sub-0005"
MCI_SUBJECTS="sub-0003 sub-0007"
NC_SUBJECTS="sub-0010 sub-0015"

# 分析 AD 組
python scripts/paper_analysis.py \
    --subjects $AD_SUBJECTS \
    --output output/analysis_AD_group

# 分析 MCI 組
python scripts/paper_analysis.py \
    --subjects $MCI_SUBJECTS \
    --output output/analysis_MCI_group

# 分析 NC 組
python scripts/paper_analysis.py \
    --subjects $NC_SUBJECTS \
    --output output/analysis_NC_group

# 生成所有圖表
for dir in output/analysis_*_group; do
    python scripts/visualize_results.py --input $dir
done

echo "All analyses complete!"
```

運行：
```bash
chmod +x run_all_analyses.sh
./run_all_analyses.sh
```

### 9.2 生成 LaTeX 表格

創建 `export_latex_tables.py`:
```python
import pandas as pd
import json
from pathlib import Path

# 讀取總結數據
with open('output/paper_results/analysis_summary_*.md', 'r') as f:
    # 解析 Markdown 表格並轉換為 LaTeX
    pass

# 或直接從 metrics 生成
metrics_dir = Path('output/paper_results/metrics')
all_metrics = []

for file in metrics_dir.glob('metrics_*.json'):
    with open(file) as f:
        all_metrics.append(json.load(f))

# 創建 DataFrame
df = pd.DataFrame([
    {
        'Subject': m['subject_id'],
        'Prediction': m['prediction'],
        'Confidence': f"{m['confidence']:.4f}",
        'UQ': f"{m['uq_score']:.4f}",
        'Decision': m['agent_decision']
    }
    for m in all_metrics
])

# 導出 LaTeX
print(df.to_latex(index=False, caption='CDDA Analysis Results', label='tab:results'))
```

---

**祝論文撰寫順利！** 📚✨

如有問題，請參考主 README.md 或聯繫開發團隊。
