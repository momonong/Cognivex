# CDDA Scripts 總覽

這個目錄包含所有用於 CDDA 系統分析和測試的腳本。

## 📋 腳本列表

### 1. 論文分析腳本

#### `paper_analysis.py` - 詳細分析腳本
**用途**: 分析單個或多個受試者，生成完整的論文用數據

**特色**:
- ✅ 完整的分析過程記錄
- ✅ 多種格式輸出 (JSON, Markdown, CSV, TXT)
- ✅ 詳細的推理鏈記錄
- ✅ 臨床報告生成
- ✅ 性能指標統計

**使用方法**:
```bash
# 分析單個受試者
python scripts/paper_analysis.py --subject sub-0005

# 分析多個受試者
python scripts/paper_analysis.py --subjects sub-0001 sub-0002 sub-0003

# 分析所有受試者
python scripts/paper_analysis.py
```

**文檔**: [README_PAPER_ANALYSIS.md](README_PAPER_ANALYSIS.md)

---

#### `comprehensive_statistics.py` - 綜合統計腳本 ⭐ 推薦
**用途**: 自動掃描並分析所有受試者，生成詳細統計報告

**特色**:
- ✅ 自動掃描所有可用受試者
- ✅ 14 個類別的詳細統計
- ✅ 信心度、不確定性、異常檢測分析
- ✅ Agent 決策統計
- ✅ 特徵重要性統計
- ✅ 組合條件分析
- ✅ 性能和推理鏈統計
- ✅ 關鍵發現自動總結

**使用方法**:
```bash
# 分析所有受試者
python scripts/comprehensive_statistics.py

# 測試模式 (只分析 5 個)
python scripts/comprehensive_statistics.py --limit 5

# 指定輸出目錄
python scripts/comprehensive_statistics.py --output output/my_stats
```

**文檔**: [README_COMPREHENSIVE_STATISTICS.md](README_COMPREHENSIVE_STATISTICS.md)

---

#### `visualize_results.py` - 可視化腳本
**用途**: 從分析結果生成論文用圖表

**特色**:
- ✅ 預測分布圖
- ✅ 信心度 vs 不確定性散點圖
- ✅ Agent 決策分布圖
- ✅ 性能指標圖
- ✅ 特徵重要性圖
- ✅ 混淆矩陣

**使用方法**:
```bash
# 從 paper_analysis.py 的結果生成圖表
python scripts/visualize_results.py --input output/paper_results

# 指定輸出目錄
python scripts/visualize_results.py \
    --input output/paper_results \
    --output output/figures
```

---

### 2. 快速測試腳本

#### `quick_paper_test.py` - 快速論文測試
**用途**: 快速測試 paper_analysis.py 是否正常運作

**使用方法**:
```bash
python scripts/quick_paper_test.py
```

分析 2 個受試者，結果保存在 `output/quick_test/`

---

#### `test_statistics.py` - 快速統計測試
**用途**: 快速測試 comprehensive_statistics.py 是否正常運作

**使用方法**:
```bash
python scripts/test_statistics.py
```

分析 3 個受試者，結果保存在 `output/test_statistics/`

---

### 3. 其他腳本

#### `download_models.py` (如果存在)
**用途**: 下載 LLM 模型

#### `import_knowledge_graph.py` (如果存在)
**用途**: 導入知識圖譜到 Neo4j

#### `batch_analysis.py` (如果存在)
**用途**: 批量處理腳本

---

## 🚀 推薦工作流程

### 論文撰寫工作流程

```bash
# Step 1: 快速測試系統
python scripts/test_statistics.py

# Step 2: 運行完整統計分析
python scripts/comprehensive_statistics.py

# Step 3: 查看統計報告
cat output/comprehensive_statistics/comprehensive_statistics_report.txt

# Step 4: 如需詳細的個別案例分析
python scripts/paper_analysis.py --subjects sub-0001 sub-0005 sub-0010

# Step 5: 生成可視化圖表
python scripts/visualize_results.py --input output/paper_results
```

### 快速驗證工作流程

```bash
# 快速測試
python scripts/quick_paper_test.py

# 如果成功，運行完整分析
python scripts/comprehensive_statistics.py
```

---

## 📊 輸出對比

### paper_analysis.py 輸出
```
output/paper_results/
├── logs/                           # 執行日誌
├── reports/                        # 臨床報告 (Markdown)
├── reasoning_chains/               # 推理鏈 (JSON + TXT)
├── metrics/                        # 性能指標和特徵
├── result_*.json                   # 完整結果
└── analysis_summary_*.md           # 總結報告
```

**適合**: 詳細案例研究、推理過程分析

### comprehensive_statistics.py 輸出
```
output/comprehensive_statistics/
├── comprehensive_statistics_report.txt    # 詳細統計報告
├── comprehensive_statistics.json          # JSON 數據
└── comprehensive_statistics.csv           # CSV 結果表
```

**適合**: 整體統計分析、論文結果章節

---

## 🎯 使用場景

### 場景 1: 需要整體統計數據
**使用**: `comprehensive_statistics.py`

這個腳本會給你：
- 所有受試者的統計總結
- 信心度、不確定性分布
- Agent 決策統計
- 特徵重要性統計
- 準確率分析

### 場景 2: 需要詳細案例分析
**使用**: `paper_analysis.py`

這個腳本會給你：
- 每個受試者的完整推理鏈
- 詳細的臨床報告
- MCP 動作記錄
- 執行摘要

### 場景 3: 需要可視化圖表
**使用**: `visualize_results.py`

這個腳本會給你：
- 6 種論文用圖表
- 高解析度 PNG 文件
- 可直接用於論文

### 場景 4: 快速驗證系統
**使用**: `quick_paper_test.py` 或 `test_statistics.py`

快速測試系統是否正常運作

---

## 📝 論文撰寫建議

### 方法論章節
使用 `paper_analysis.py` 的：
- 推理鏈文件 (展示 Agent 決策過程)
- 執行日誌 (展示系統配置)

### 實驗結果章節
使用 `comprehensive_statistics.py` 的：
- 統計報告 (整體性能)
- CSV 數據 (生成表格)

### 案例研究章節
使用 `paper_analysis.py` 的：
- 臨床報告 (詳細案例)
- 特徵重要性 CSV

### 可視化章節
使用 `visualize_results.py` 的：
- 所有生成的圖表

---

## 🔧 常見問題

### Q: 應該先運行哪個腳本？

**A**: 建議順序：
1. `test_statistics.py` (快速測試)
2. `comprehensive_statistics.py` (完整統計)
3. `paper_analysis.py` (詳細案例，如需要)
4. `visualize_results.py` (生成圖表)

### Q: 兩個分析腳本有什麼區別？

**A**: 
- `paper_analysis.py`: 詳細記錄每個受試者的完整過程
- `comprehensive_statistics.py`: 統計所有受試者的整體表現

### Q: 如何選擇使用哪個腳本？

**A**:
- 需要整體統計 → `comprehensive_statistics.py`
- 需要詳細案例 → `paper_analysis.py`
- 兩者都需要 → 都運行

### Q: 輸出文件太多怎麼辦？

**A**: 
- `comprehensive_statistics.py` 只生成 3 個文件
- `paper_analysis.py` 可以只分析需要的受試者

### Q: 如何加快分析速度？

**A**:
1. 使用 4-bit 量化 (默認啟用)
2. 使用 `--limit` 參數限制數量
3. 只分析需要的受試者

---

## 📚 詳細文檔

- [paper_analysis.py 使用指南](README_PAPER_ANALYSIS.md)
- [comprehensive_statistics.py 使用指南](README_COMPREHENSIVE_STATISTICS.md)
- [完整使用範例](EXAMPLE_USAGE.md)

---

## 💡 提示

1. **首次使用**: 先運行快速測試腳本
2. **論文撰寫**: 優先使用 `comprehensive_statistics.py`
3. **案例研究**: 使用 `paper_analysis.py` 分析特定受試者
4. **圖表生成**: 最後運行 `visualize_results.py`

---

**祝研究順利！** 📊✨
