# Cognivex CDDA - 快速參考卡片

## 🚀 一分鐘快速開始

```bash
# 1. 測試系統（30 秒）
python test_all_systems.py

# 2. 啟動 Web 介面（5 秒）
run_cdda_app.bat
# 或
streamlit run app_cdda.py

# 3. 瀏覽器開啟
http://localhost:8501
```

---

## 📋 核心命令

### 系統測試
```bash
# 快速測試所有模組
python test_all_systems.py

# CDDA 詳細測試
python tests/test_cdda_tools.py      # Phase 1
python tests/test_cdda_agent.py      # Phase 2
python scripts/neo4j/test_multihop_queries.py  # Phase 3
python tests/test_agent_b_consultant.py  # Phase 4
python tests/test_a2a_integration.py     # Phase 4
```

### CDDA 分析
```bash
# 命令列分析
python -c "
from app.agents.cdda_agent import CDDAAgent
agent = CDDAAgent(use_llm=False)
result = agent.run_analysis('sub_0005')
agent.print_report(result)
"

# 完整演示
python scripts/demo_phase4_complete.py
python scripts/demo_a2a_agents.py
python scripts/demo_mcp_server.py
```

### Web 介面
```bash
# CDDA 整合介面（推薦）
streamlit run app_cdda.py

# 傳統介面
streamlit run app.py

# 指定埠號
streamlit run app_cdda.py --server.port 8502
```

---

## 🎯 CDDA 決策邏輯

| 條件 | 決策 | 行動 |
|------|------|------|
| UQ > 0.8 | 反事實模擬 | Tool 2: 識別關鍵驅動因素 |
| \|Z\| > 2.5 | 異常調查 | Tool 4: 查詢知識圖譜 |
| 標準情況 | 基礎報告 | 生成標準診斷報告 |

---

## 📊 系統指標

| 項目 | 數值 |
|------|------|
| **測試覆蓋率** | 24/24 (100%) ✅ |
| **Phase 完成度** | 5/5 (100%) ✅ |
| **知識圖譜** | 360 關係, 163 實體 |
| **執行時間（規則式）** | 10-30 秒 |
| **執行時間（LLM）** | 2-5 分鐘 |

---

## 🔧 常見問題快速解決

### Neo4j 連接失敗
```bash
docker restart neo4j-fmri
# 或
sudo systemctl restart neo4j
```

### CUDA 不可用
```bash
nvidia-smi
poetry run poe autoinstall-torch-cuda
```

### LLM 錯誤
```bash
# 檢查 Ollama
curl http://localhost:11434/api/tags

# 或使用規則式模式（關閉 LLM）
```

---

## 📚 文檔快速導航

| 文檔 | 用途 |
|------|------|
| **README.md** | 完整系統文檔 |
| **快速開始.md** | 5 分鐘快速開始 |
| **TESTING_GUIDE.md** | 詳細測試指南 |
| **CDDA_WEB_INTERFACE_GUIDE.md** | Web 介面使用 |
| **CDDA_IMPLEMENTATION_STATUS.md** | 實作狀態 |
| **PROJECT_COMPLETION_SUMMARY.md** | 專案總結 |

---

## 🎓 使用場景

### 研究者
```python
# 啟用 LLM 模式 + 顯示推理鏈
use_llm = True
show_reasoning = True

# 匯出推理日誌
agent.save_reasoning_log(result, "output/logs/sub_0005.json")
```

### 臨床醫師
```python
# 快速模式（規則式）
use_llm = False
show_reasoning = False

# 執行時間：~10-30 秒
```

### 開發者
```bash
# 執行所有測試
python test_all_systems.py

# 查看架構
cat CDDA_IMPLEMENTATION_STATUS.md
```

---

## 🌟 核心功能

- ✅ **自主決策** - 三路決策邏輯
- ✅ **反事實分析** - What-if 模擬
- ✅ **異常檢測** - Z-score 評估
- ✅ **混合病理** - 多重疾病識別
- ✅ **透明推理** - 完整推理鏈
- ✅ **知識增強** - GraphRAG 整合
- ✅ **雙 LLM** - Agent A + Agent B
- ✅ **Web 介面** - 友善 UI

---

## 📞 獲取幫助

1. **快速測試**: `python test_all_systems.py`
2. **查看文檔**: `README.md`
3. **測試指南**: `TESTING_GUIDE.md`
4. **Web 指南**: `CDDA_WEB_INTERFACE_GUIDE.md`

---

**Cognivex CDDA** - Making neuroimaging AI explainable and trustworthy

**版本:** 1.0 | **狀態:** ✅ 生產就緒 | **測試:** 100% 通過
