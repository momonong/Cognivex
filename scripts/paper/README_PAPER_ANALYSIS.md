# CDDA Paper Analysis Script 使用指南

這個腳本專門為論文撰寫設計，會詳細輸出所有分析過程和結果。

## 🎯 功能特色

### 完整的分析輸出
- ✅ 系統信息 (Python 版本、GPU 信息等)
- ✅ 初始化過程詳細記錄
- ✅ 診斷結果 (預測、信心度、不確定性)
- ✅ 特徵重要性分析 (SHAP + Z-score)
- ✅ 異常檢測結果
- ✅ 工具調用結果 (反事實模擬、知識圖譜查詢)
- ✅ 執行摘要
- ✅ 完整推理鏈 (Agent A + Agent B)
- ✅ MCP 動作詳細記錄
- ✅ 臨床報告全文
- ✅ 性能指標 (時間、吞吐量)

### 多種輸出格式
- 📄 **JSON**: 完整結構化數據
- 📝 **Markdown**: 臨床報告和總結
- 📊 **CSV**: 特徵重要性數據
- 📋 **TXT**: 推理鏈純文本
- 📈 **日誌**: 完整執行日誌

### 批量處理支持
- 單個受試者分析
- 多個受試者批量分析
- 自動生成總結報告
- 統計分析 (準確率、平均信心度等)

## 📦 安裝

確保已安裝所有依賴：

```bash
pip install -r requirements.txt
```

## 🚀 使用方法

### 1. 分析單個受試者

```bash
python scripts/paper_analysis.py --subject sub-0005
```

### 2. 分析多個受試者

```bash
python scripts/paper_analysis.py --subjects sub-0001 sub-0002 sub-0003 sub-0005
```

### 3. 分析所有受試者

```bash
# 不指定 subject，會自動掃描 data/MRI_processed/ 目錄
python scripts/paper_analysis.py
```

### 4. 指定輸出目錄

```bash
python scripts/paper_analysis.py --subject sub-0005 --output output/my_paper_results
```

### 5. 使用自定義模型路徑

```bash
python scripts/paper_analysis.py \
    --subject sub-0005 \
    --orchestrator-path /path/to/phi-4-mini \
    --consultant-path /path/to/llama3.1-aloe
```

### 6. 提供真實標籤文件

創建一個 JSON 文件 `ground_truth.json`:
```json
{
  "sub-0001": "AD",
  "sub-0002": "NC",
  "sub-0003": "MCI",
  "sub-0005": "AD"
}
```

然後運行：
```bash
python scripts/paper_analysis.py \
    --subjects sub-0001 sub-0002 sub-0003 sub-0005 \
    --ground-truth-file ground_truth.json
```

### 7. 使用規則模式 (不使用 LLM)

```bash
python scripts/paper_analysis.py --subject sub-0005 --no-llm
```

## 📂 輸出結構

運行後會在輸出目錄生成以下結構：

```
output/paper_results/
├── logs/
│   └── analysis_log_20251128_143022.txt          # 完整執行日誌
├── reports/
│   ├── report_sub-0001_20251128_143022.md        # 臨床報告 (Markdown)
│   ├── report_sub-0002_20251128_143045.md
│   └── ...
├── reasoning_chains/
│   ├── reasoning_sub-0001_20251128_143022.json   # 推理鏈 (JSON)
│   ├── reasoning_sub-0001_20251128_143022.txt    # 推理鏈 (純文本)
│   └── ...
├── metrics/
│   ├── metrics_sub-0001_20251128_143022.json     # 性能指標
│   ├── features_sub-0001_20251128_143022.csv     # 特徵重要性
│   └── ...
├── result_sub-0001_20251128_143022.json          # 完整結果
├── result_sub-0002_20251128_143045.json
└── analysis_summary_20251128_143022.md           # 總結報告
```

## 📊 輸出文件說明

### 1. 完整結果 (result_*.json)

包含所有分析數據的完整 JSON 文件：
- 診斷結果
- ContextObject
- 推理鏈
- 臨床報告
- 元數據

### 2. 推理鏈 (reasoning_*.json / reasoning_*.txt)

完整的 Agent A 和 Agent B 推理過程：
- Agent A 編排步驟
- MCP 動作記錄
- Agent B 合成步驟
- 時間戳

### 3. 臨床報告 (report_*.md)

Markdown 格式的臨床報告：
- 診斷摘要
- 預測結果
- 執行摘要
- 完整報告內容

### 4. 特徵重要性 (features_*.csv)

CSV 格式的特徵分析：
```csv
Rank,ROI_Name,SHAP_Value,Z_Score,Feature_Value,Significance
1,Hippocampus_L,0.152341,-2.8456,2500.0,Atrophy
2,Hippocampus_R,0.123456,-2.6789,2450.0,Atrophy
...
```

### 5. 性能指標 (metrics_*.json)

性能和統計數據：
- 初始化時間
- 分析時間
- 吞吐量
- 推理步驟統計
- 前 5 個重要特徵

### 6. 總結報告 (analysis_summary_*.md)

所有受試者的統計總結：
- 預測分布
- 決策分布
- 平均信心度
- 平均不確定性
- 準確率 (如果有真實標籤)
- 性能統計
- 個別結果表格

### 7. 執行日誌 (analysis_log_*.txt)

完整的執行過程記錄：
- 系統信息
- 初始化過程
- 每個階段的詳細輸出
- 錯誤信息 (如果有)

## 📝 論文撰寫建議

### 1. 系統架構圖

使用 `analysis_log_*.txt` 中的系統信息部分：
- Python 版本
- GPU 型號和記憶體
- 模型配置

### 2. 方法論

使用推理鏈文件展示：
- Agent A 的決策邏輯
- MCP 動作序列
- Agent B 的合成過程

### 3. 實驗結果

使用總結報告 (`analysis_summary_*.md`)：
- 預測分布表格
- 準確率統計
- 信心度和不確定性分析

### 4. 案例研究

選擇代表性案例，使用：
- 臨床報告 (`report_*.md`)
- 特徵重要性 (`features_*.csv`)
- 推理鏈 (`reasoning_*.txt`)

### 5. 性能分析

使用性能指標文件：
- 平均分析時間
- 吞吐量
- 各階段時間分解

### 6. 可解釋性展示

使用特徵重要性和推理鏈：
- SHAP 值排序
- Z-score 異常檢測
- 反事實分析結果
- 知識圖譜查詢結果

## 🔍 常見問題

### Q1: 如何只分析特定類別的受試者？

創建一個包含特定受試者的列表：
```bash
python scripts/paper_analysis.py --subjects sub-0001 sub-0003 sub-0005
```

### Q2: 如何獲得更詳細的日誌？

日誌已經非常詳細。如果需要更多調試信息，可以修改腳本中的 `verbose=False` 為 `verbose=True`。

### Q3: 分析失敗怎麼辦？

檢查日誌文件中的錯誤信息。常見問題：
- 模型路徑不正確
- GPU 記憶體不足 (使用 `--use-4bit`)
- 數據文件缺失

### Q4: 如何比較不同配置的結果？

使用不同的輸出目錄：
```bash
# 配置 1: LLM 模式
python scripts/paper_analysis.py --subjects sub-0001 sub-0002 \
    --output output/results_llm

# 配置 2: 規則模式
python scripts/paper_analysis.py --subjects sub-0001 sub-0002 \
    --output output/results_rules --no-llm
```

### Q5: 如何生成可視化圖表？

特徵重要性 CSV 文件可以用 Python 或 Excel 生成圖表：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取特徵數據
df = pd.read_csv('output/paper_results/metrics/features_sub-0005_*.csv')

# 繪製 SHAP 值
plt.figure(figsize=(10, 6))
plt.barh(df['ROI_Name'][:10], df['SHAP_Value'][:10])
plt.xlabel('SHAP Value')
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

## 📧 支持

如果遇到問題，請檢查：
1. 日誌文件中的錯誤信息
2. 模型是否正確加載
3. 數據文件是否完整

## 🎓 引用

如果在論文中使用此腳本，請引用：

```bibtex
@software{cdda_paper_analysis,
  title={CDDA Paper Analysis Script},
  author={Your Name},
  year={2025},
  url={https://github.com/your-org/cdda-framework}
}
```

---

**祝論文撰寫順利！** 📚✨
