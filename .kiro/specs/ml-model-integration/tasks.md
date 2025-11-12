# Implementation Plan

## Task Overview

本實作計畫將 ML 模型整合分為 6 個主要階段，每個階段包含具體的編碼任務。所有任務都專注於可執行的程式碼實作，並確保每一步都能增量式地建構功能。

---

## Phase 1: 核心模組建立

- [x] 1. 建立 ML 處理模組基礎架構


  - 建立 `app/core/ml_processing/` 目錄結構
  - 建立 `__init__.py` 檔案並定義模組介面
  - 建立 `config.py` 定義 MLModelConfig 資料類別
  - _Requirements: 1.1, 1.3_



- [ ] 1.1 實作 MLModelLoader 類別
  - 在 `app/core/ml_processing/model_loader.py` 建立 MLModelLoader 類別
  - 實作 `load_model()` 方法載入 Random Forest 模型
  - 實作 `load_scaler()` 方法載入 StandardScaler
  - 實作 `load_roi_list()` 方法從 CSV 讀取 ROI 列表
  - 實作 `load_feature_names()` 方法讀取特徵名稱
  - 實作 `get_all_components()` 方法一次性載入所有組件
  - 加入錯誤處理，當檔案缺失時拋出 ModelLoadError


  - _Requirements: 1.1, 1.2, 1.4_

- [ ] 1.2 實作 ROIFeatureExtractor 類別
  - 在 `app/core/ml_processing/feature_extractor.py` 建立 ROIFeatureExtractor 類別
  - 實作 `load_atlas()` 方法載入 AAL atlas (使用 nilearn.datasets)
  - 實作 `get_roi_mapping()` 方法建立 ROI 名稱到索引的映射
  - 實作 `extract_features()` 方法使用 NiftiLabelsMasker 提取 ROI 特徵


  - 驗證提取的特徵向量維度為 32
  - 加入錯誤處理，當影像格式不正確時拋出 FeatureExtractionError
  - _Requirements: 2.1, 2.2, 2.3, 2.6_

- [ ] 1.3 撰寫核心模組的單元測試
  - 建立 `tests/test_ml_model_loader.py`
  - 測試成功載入模型的情境
  - 測試檔案缺失時的錯誤處理
  - 建立 `tests/test_roi_feature_extractor.py`
  - 測試特徵提取輸出形狀正確性
  - 測試無效 ROI 名稱的處理


  - 建立 mock 資料檔案於 `tests/fixtures/`
  - _Requirements: 1.2, 2.4, 2.5_

---

## Phase 2: Agent 節點實作

- [ ] 2. 實作 structural_mri_inference agent
  - 建立 `app/agents/structural_mri_inference.py`
  - 實作 `run_structural_mri_inference(state: AgentState) -> dict` 函式
  - 從 state 取得 fmri_scan_path (實際為 T1 MRI 路徑)
  - 使用 MLModelLoader 載入模型組件
  - 使用 ROIFeatureExtractor 提取 32 個 ROI 特徵


  - 使用 scaler 標準化特徵
  - 執行模型預測並取得分類結果和信心分數
  - 提取模型的 feature_importances_
  - 將結果更新到 state: classification_result, prediction_confidence, roi_features, feature_importances
  - 記錄處理過程到 trace_log
  - 加入完整的錯誤處理和 error_log 記錄
  - _Requirements: 3.1, 3.2, 3.3, 3.5, 4.2_

- [x] 2.1 實作 structural_feature_analyzer agent



  - 建立 `app/agents/structural_feature_analyzer.py`
  - 實作 `analyze_feature_importance(state: AgentState) -> dict` 函式
  - 從 state 取得 feature_importances
  - 排序特徵重要性並選擇 Top 10
  - 將 ROI 資訊轉換為 BrainRegionInfo 格式
  - 設定 activation_score 為 feature_importance 值
  - 設定 importance_rank 欄位
  - 更新 state 的 activated_regions 欄位
  - _Requirements: 5.1, 5.2_



- [ ] 2.2 實作 structural_visualizer agent
  - 建立 `app/agents/structural_visualizer.py`
  - 實作 `generate_structural_visualizations(state: AgentState) -> dict` 函式
  - 實作 `plot_feature_importance()` 輔助函式生成橫條圖
  - 使用 matplotlib/seaborn 繪製 Top 10 特徵重要性
  - 實作 `plot_roi_on_brain()` 輔助函式生成 3D 腦區視覺化
  - 使用 nilearn.plotting.plot_roi 在 MNI152 模板上標記 ROI
  - 根據重要性進行顏色編碼


  - 儲存視覺化圖片到 output 目錄
  - 將圖片路徑記錄到 state 的 visualization_paths
  - _Requirements: 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 2.3 撰寫 agent 節點的單元測試
  - 建立 `tests/test_structural_agents.py`
  - 測試 structural_mri_inference 的輸入輸出
  - 測試 structural_feature_analyzer 的排序邏輯
  - 測試 structural_visualizer 的圖片生成
  - 驗證所有 agent 正確更新 AgentState
  - _Requirements: 4.2_



---

## Phase 3: Workflow 整合

- [x] 3. 擴展 AgentState 定義


  - 修改 `app/graph/state.py`
  - 新增 analysis_mode 欄位 (Literal["structural", "functional"])
  - 新增 ml_model_type 欄位
  - 新增 roi_features 欄位 (Dict[str, float])
  - 新增 feature_importances 欄位 (Dict[str, float])
  - 新增 prediction_confidence 欄位 (float)
  - 新增 feature_importance_plot_path 欄位
  - 新增 roi_visualization_path 欄位
  - 擴展 BrainRegionInfo 加入 feature_value, importance_rank, clinical_relevance 欄位
  - _Requirements: 4.2_



- [ ] 3.1 實作 workflow 路由邏輯
  - 修改 `app/graph/workflow.py`
  - 實作 `route_by_analysis_mode(state: AgentState) -> str` 函式
  - 根據 state 的 analysis_mode 返回對應的節點名稱
  - "structural" → "structural_mri_inference"
  - "functional" → "inference" (現有節點)
  - _Requirements: 4.4_



- [ ] 3.2 更新 workflow 圖結構
  - 在 `app/graph/workflow.py` 中註冊新的 agent 節點
  - 加入 structural_mri_inference 節點
  - 加入 structural_feature_analyzer 節點
  - 加入 structural_visualizer 節點
  - 建立條件式邊從 START 到 router

  - 建立條件式邊從 router 到兩個分支
  - 建立 structural 分支的邊: inference → analyzer → visualizer → entity_linker
  - 確保 functional 分支保持不變
  - 兩個分支在 entity_linker 後匯合
  - _Requirements: 4.1, 4.3, 4.4, 4.5_


- [ ] 3.3 撰寫 workflow 整合測試
  - 建立 `tests/test_structural_workflow_integration.py`
  - 測試完整的 structural pipeline 執行
  - 驗證 workflow 正確路由到 structural 分支
  - 驗證 functional 分支不受影響
  - 測試 state 在各節點間正確傳遞
  - _Requirements: 4.5_


---

## Phase 4: UI 整合

- [ ] 4. 更新 Streamlit UI 加入模式選擇
  - 修改 `app.py`
  - 在側邊欄加入 "Analysis Mode" 選擇器

  - 選項: "Functional MRI (fMRI)" 和 "Structural MRI (T1)"
  - 根據選擇的模式更新 session_state 的 analysis_mode
  - _Requirements: 7.1, 7.2_

- [ ] 4.1 實作模式特定的 UI 元件
  - 當選擇 "Structural MRI" 時，顯示 ML 模型資訊卡片
  - 顯示模型類型、特徵數量、準確率等資訊
  - 當選擇 "Functional MRI" 時，顯示現有的深度學習模型選擇器


  - 根據模式調整檔案搜尋模式
  - _Requirements: 7.2, 7.3_

- [ ] 4.2 實作結構性 MRI 結果顯示頁面
  - 在 `app.py` 的結果區域加入模式判斷
  - 建立 structural MRI 專用的結果顯示區塊
  - 顯示預測結果卡片 (分類、信心分數、模型類型)

  - 顯示特徵重要性視覺化圖片
  - 顯示 3D 腦區視覺化圖片
  - 建立詳細 ROI 資訊表格 (使用 st.dataframe)
  - _Requirements: 7.4_

- [x] 4.3 加入進度指示和錯誤處理 UI

  - 實作分階段進度條顯示
  - 階段: 載入模型 → 提取特徵 → 執行預測 → 生成視覺化
  - 實作使用者友善的錯誤訊息顯示
  - 建立錯誤訊息映射表 (技術錯誤 → 友善訊息)
  - 加入 expander 顯示技術細節
  - _Requirements: 7.4, 9.4, 10.5_

- [x] 4.4 實作快取機制優化 UI 效能

  - 使用 @st.cache_resource 快取 ML 模型載入
  - 使用 @st.cache_data 快取 AAL atlas 載入
  - 確保模型只在首次使用時載入
  - _Requirements: 10.1, 10.2_

---

## Phase 5: 報告生成整合

- [x] 5. 擴展 report_generator 支援結構性 MRI

  - 修改 `app/agents/report_generator.py`
  - 更新 `generate_final_report()` 函式加入模式判斷
  - 實作 `generate_structural_report(state: AgentState) -> dict` 函式
  - 從 state 收集結構性 MRI 的分析結果
  - 建立結構性 MRI 專用的 prompt 模板
  - _Requirements: 8.1, 8.2_

- [ ] 5.1 整合臨床知識到報告生成
  - 讀取 `docs/MODEL_OVERALL.md` 提取臨床驗證資訊
  - 實作 `extract_clinical_relevance()` 函式解析 ROI 的臨床意義
  - 將 Top 10 ROI 的臨床相關性加入 prompt
  - 包含 Braak 分期、DMN 理論、功能系統等資訊
  - _Requirements: 8.3, 8.4_

- [ ] 5.2 實作中英文雙語報告生成
  - 使用 LLM 生成英文報告
  - Prompt 包含: 分類結果、信心分數、Top 10 ROI、臨床解釋
  - 使用 LLM 翻譯為繁體中文
  - 確保報告包含必要的免責聲明
  - 明確標註這是輔助診斷工具
  - 儲存報告到 state 的 generated_reports
  - _Requirements: 8.3, 8.5_

- [ ] 5.3 測試報告生成功能
  - 建立 `tests/test_report_generation.py`
  - 測試結構性 MRI 報告生成
  - 驗證報告包含所有必要章節
  - 測試中英文翻譯功能
  - 驗證免責聲明存在
  - _Requirements: 8.2, 8.3_

---

## Phase 6: 測試、優化與文件

- [ ] 6. 實作完整的錯誤處理機制
  - 建立 `app/core/ml_processing/exceptions.py`
  - 定義 MLIntegrationError 基礎類別
  - 定義 ModelLoadError, FeatureExtractionError, AtlasLoadError, PredictionError
  - 在所有 agent 節點加入 try-except 錯誤捕獲
  - 實作錯誤訊息記錄到 error_log
  - 實作錯誤恢復機制
  - _Requirements: 9.1, 9.2, 9.3_

- [ ] 6.1 實作效能監控和日誌記錄
  - 建立 `app/core/ml_processing/monitoring.py`
  - 實作 `monitor_performance` 裝飾器記錄執行時間
  - 實作 `log_inference_event()` 函式記錄推論事件
  - 設定結構化日誌格式 (JSON)
  - 加入時間戳記和 subject_id 追蹤
  - _Requirements: 9.3, 10.4_

- [ ] 6.2 實作記憶體管理優化
  - 在 feature extraction 後釋放影像記憶體
  - 使用 gc.collect() 強制垃圾回收
  - 實作 masker 物件的快取重用
  - _Requirements: 10.3_

- [ ] 6.3 撰寫端到端測試
  - 建立 `tests/test_e2e_structural_analysis.py`
  - 使用真實受試者數據進行完整流程測試
  - 驗證所有輸出檔案正確生成
  - 驗證報告內容完整性
  - 測試效能符合需求 (< 5 秒)
  - _Requirements: 10.4_

- [ ] 6.4 撰寫效能測試
  - 建立 `tests/test_performance.py`
  - 測試推論速度符合 5 秒限制
  - 測試模型快取有效性
  - 測試記憶體使用量
  - _Requirements: 10.1, 10.2, 10.4_

- [ ] 6.5 撰寫整合文件
  - 更新 `README.md` 加入結構性 MRI 分析說明
  - 建立 `docs/ml_model_integration.md` 詳細文件
  - 記錄 API 介面和使用範例
  - 記錄配置選項和環境需求
  - 建立故障排除指南
  - _Requirements: 所有_

---

## 實作順序建議

建議按照以下順序執行任務以確保增量式開發：

1. **Week 1**: Tasks 1, 1.1, 1.2 (核心模組)
2. **Week 2**: Tasks 1.3, 2, 2.1 (Agent 基礎)
3. **Week 3**: Tasks 2.2, 2.3, 3 (視覺化與 State)
4. **Week 4**: Tasks 3.1, 3.2, 3.3 (Workflow 整合)
5. **Week 5**: Tasks 4, 4.1, 4.2 (UI 基礎)
6. **Week 6**: Tasks 4.3, 4.4, 5 (UI 優化與報告)
7. **Week 7**: Tasks 5.1, 5.2, 5.3 (報告完善)
8. **Week 8**: Tasks 6, 6.1, 6.2, 6.3, 6.4, 6.5 (測試與文件)

每個任務完成後應該：
1. 執行相關的單元測試
2. 手動測試功能正確性
3. 提交程式碼變更
4. 更新任務狀態

## 驗收標準

整合完成後，系統應該能夠：

✅ 在 UI 中選擇 "Structural MRI" 模式
✅ 上傳 T1 MRI 檔案並執行分析
✅ 在 5 秒內完成推論（不含檔案載入）
✅ 顯示分類結果和信心分數
✅ 顯示特徵重要性圖表
✅ 顯示 3D 腦區視覺化
✅ 生成中英文臨床報告
✅ 正確處理錯誤情況並顯示友善訊息
✅ 不影響現有的 fMRI 分析功能
✅ 通過所有單元測試和整合測試
