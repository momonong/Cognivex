# Cognivex 測試指南

## 快速測試（5 分鐘）

### 1. 一鍵系統測試
```bash
python test_all_systems.py
```

**這個腳本會測試：**
- ✅ 數據結構完整性
- ✅ CDDA Phase 1: 核心工具（RF + SHAP + UQ + 反事實）
- ✅ CDDA Phase 2: 自主代理（三路決策邏輯）
- ✅ CDDA Phase 3: 知識整合（GraphRAG + Neo4j）
- ✅ CDDA Phase 4: 雙 LLM A2A（MCP + Agent A + Agent B）
- ✅ LLM 提供者（Bedrock + Ollama + HuggingFace）
- ✅ LangGraph 管線
- ✅ Neo4j 連接

**預期結果：**
```
總計: 8/8 測試通過 (100.0%)
🎉 所有測試通過！系統運行正常。
```

---

## 詳細測試（15-30 分鐘）

### Phase 1: 核心工具
```bash
python tests/test_cdda_tools.py
```
**測試內容：**
- Tool 1: 診斷報告生成（RF 預測 + SHAP 解釋 + UQ 評分 + 異常檢測）
- Tool 2: 反事實模擬（特徵遮罩 + 影響分析 + 自然語言解釋）

**預期結果：** 4/4 tests passed

---

### Phase 2: 自主代理
```bash
python tests/test_cdda_agent.py
```
**測試內容：**
- 代理初始化
- 三路決策邏輯（UQ / Anomaly / Standard）
- 工具自動編排
- 決策優先級
- 推理鏈生成

**預期結果：** 7/7 tests passed

---

### Phase 3: 知識圖譜
```bash
python scripts/neo4j/test_multihop_queries.py
```
**測試內容：**
- 多區域查詢（3 個區域）
- 相關區域查找（10 個相關區域）
- 疾病關聯查詢（32 個 AD 區域）
- GraphRAG 整合測試

**預期結果：** 4/4 tests passed

---

### Phase 4: 雙 LLM A2A
```bash
# Agent B 單元測試
python tests/test_agent_b_consultant.py

# A2A 整合測試
python tests/test_a2a_integration.py
```
**測試內容：**
- Agent B 臨床報告合成
- 異常感知分析
- 反事實解釋
- A2A 交接協議
- 上下文隔離
- 推理鏈聚合

**預期結果：** 
- Agent B: 5/5 tests passed
- A2A Integration: 4/4 tests passed

---

## 系統演示（10-15 分鐘）

### 完整 Phase 4 演示
```bash
python scripts/demo_phase4_complete.py
```
**展示內容：**
- MCP 資源讀取
- Agent A 決策過程
- 工具調用（如需要）
- Agent B 臨床合成
- 完整推理鏈

---

### A2A 代理協作演示
```bash
python scripts/demo_a2a_agents.py
```
**展示內容：**
- Agent A 的 MCP 操作
- ContextObject 編譯
- A2A 交接
- Agent B 的臨床報告

---

### MCP 伺服器演示
```bash
python scripts/demo_mcp_server.py
```
**展示內容：**
- MCP 資源列表
- 資源讀取
- 工具調用
- URI 路由

---

## 單一受試者分析

### 使用 CDDA Agent
```bash
python -c "
from app.agents.cdda_agent import CDDAAgent
agent = CDDAAgent(use_llm=False)
result = agent.run_analysis('sub-0005')
agent.print_report(result)
"
```

**輸出包含：**
- 診斷預測（AD/NC）
- 信心度評分
- UQ 不確定性評分
- 異常區域檢測
- 反事實分析（如觸發）
- 知識圖譜洞察（如觸發）
- 完整推理鏈

---

## Web 介面測試

### 啟動 Streamlit 應用
```bash
streamlit run app.py
```

**測試流程：**
1. 選擇受試者（sub-01 到 sub-20）
2. 選擇模型（CapsNet / MCADNNet）
3. 開始分析
4. 查看進度更新
5. 檢視結果：
   - 腦激活圖
   - 預測驗證
   - 互動式 fMRI 檢視器
   - 雙語臨床報告

---

## 故障排除

### 問題 1: Neo4j 連接失敗
```bash
# 檢查 Neo4j 狀態
docker ps | grep neo4j

# 重啟 Neo4j
docker restart neo4j-fmri
```

### 問題 2: CUDA 不可用
```bash
# 檢查 CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 重新安裝 PyTorch with CUDA
poetry run poe autoinstall-torch-cuda
```

### 問題 3: LLM 提供者錯誤
```bash
# 測試 Ollama
curl http://localhost:11434/api/tags

# 測試 AWS Bedrock
python -c "from app.services.llm_providers.bedrock import handle_text; print(handle_text('test'))"
```

---

## 測試結果驗證

### CDDA Framework 預期結果
```
Phase 1 (Tools):        4/4 tests passed ✅
Phase 2 (Agent):        7/7 tests passed ✅
Phase 3 (GraphRAG):     4/4 tests passed ✅
Phase 4 (A2A):          9/9 tests passed ✅
Total:                 24/24 tests passed (100%)
```

### 系統指標
- **執行時間**: 3-7 秒/分析
- **記憶體使用**: ~350 MB
- **GPU 記憶體**: ~2 GB（推論時）
- **Neo4j 關係**: 360 個活躍關係
- **知識實體**: 163 個腦區實體

---

## 測試優先級

### 必須通過（核心功能）
1. ✅ `test_all_systems.py` - 快速系統驗證
2. ✅ `test_cdda_tools.py` - 核心工具
3. ✅ `test_cdda_agent.py` - 自主代理

### 建議通過（完整功能）
4. ✅ `test_multihop_queries.py` - 知識圖譜
5. ✅ `test_agent_b_consultant.py` - Agent B
6. ✅ `test_a2a_integration.py` - A2A 整合

### 可選（演示和 UI）
7. ⭕ `demo_phase4_complete.py` - 完整演示
8. ⭕ `streamlit run app.py` - Web 介面

---

## 快速檢查清單

- [ ] 執行 `python test_all_systems.py` → 8/8 通過
- [ ] 執行 `python tests/test_cdda_tools.py` → 4/4 通過
- [ ] 執行 `python tests/test_cdda_agent.py` → 7/7 通過
- [ ] 執行 `python scripts/neo4j/test_multihop_queries.py` → 4/4 通過
- [ ] 執行 `python tests/test_agent_b_consultant.py` → 5/5 通過
- [ ] 執行 `python tests/test_a2a_integration.py` → 4/4 通過
- [ ] 執行 `python scripts/demo_phase4_complete.py` → 成功演示
- [ ] 執行 `streamlit run app.py` → Web 介面正常

**如果所有項目都打勾，系統完全正常運行！** 🎉

---

## 文檔參考

- 📄 **README.md** - 完整系統文檔
- 📄 **CDDA_IMPLEMENTATION_STATUS.md** - 實作狀態
- 📄 **docs/CDDA_Phase4_Complete.md** - Phase 4 詳細文檔
- 📄 **GRAPHRAG_MULTIHOP_COMPLETE.md** - GraphRAG 文檔
- 📄 **TESTING_GUIDE.md** - 本文件

---

**最後更新：** 2025年11月20日  
**測試覆蓋率：** 24/24 tests (100%)  
**系統狀態：** ✅ 完全運行正常
