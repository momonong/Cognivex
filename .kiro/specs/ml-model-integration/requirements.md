# Requirements Document

## Introduction

本專案旨在將已完成的機器學習預測模型（Random Forest，基於結構性 MRI 的 32 個 ROI 特徵）整合到現有的 `/app` 應用程式架構中。現有應用程式主要處理 fMRI 功能性影像分析，而新模型專注於結構性 MRI 的阿茲海默症分類。整合後的系統將支援雙模態分析能力。

## Glossary

- **ML_Model**: 位於 `/model/ml/final/` 的 Random Forest 機器學習模型，用於結構性 MRI 的 AD 分類
- **App_System**: 現有的 `/app` 應用程式系統，基於 LangGraph 的 agent 架構
- **Structural_MRI**: T1 加權結構性磁振造影，用於腦部結構分析
- **Functional_MRI**: 功能性磁振造影（fMRI），用於腦部活動分析
- **ROI**: Region of Interest，感興趣區域，本模型使用 32 個 AAL atlas 定義的腦區
- **AAL_Atlas**: Automated Anatomical Labeling atlas，自動解剖標記圖譜
- **Agent_Node**: LangGraph 工作流程中的單一處理節點
- **AgentState**: LangGraph 狀態物件，在節點間傳遞數據
- **Dual_Modal_Analysis**: 雙模態分析，同時支援結構性和功能性 MRI 分析

## Requirements

### Requirement 1: 模型載入與初始化

**User Story:** 作為系統開發者，我希望能夠正確載入和初始化 ML 模型及其相關檔案，以便進行結構性 MRI 分析

#### Acceptance Criteria

1. WHEN App_System 啟動時，THE ML_Model SHALL 從 `/model/ml/final/` 目錄載入所有必要檔案（final_model.pkl, final_scaler.pkl, final_roi_list.csv, final_feature_names.txt）
2. IF 任何必要檔案缺失，THEN THE App_System SHALL 記錄錯誤訊息並提供明確的檔案路徑資訊
3. THE App_System SHALL 驗證載入的模型版本與預期的 32 個 ROI 特徵數量一致
4. THE App_System SHALL 快取已載入的模型物件以避免重複載入
5. WHEN 模型載入失敗時，THE App_System SHALL 允許系統繼續運行但禁用結構性 MRI 分析功能

### Requirement 2: ROI 特徵提取

**User Story:** 作為系統開發者，我希望能夠從結構性 MRI 影像中提取 32 個 ROI 特徵，以便輸入到 ML 模型進行預測

#### Acceptance Criteria

1. WHEN 使用者提供 Structural_MRI 檔案路徑時，THE App_System SHALL 使用 nilearn 的 NiftiLabelsMasker 提取 ROI 特徵
2. THE App_System SHALL 載入 AAL_Atlas 並根據 final_roi_list.csv 選擇對應的 32 個腦區
3. THE App_System SHALL 計算每個 ROI 的平均強度值作為特徵
4. THE App_System SHALL 使用 final_scaler.pkl 對提取的特徵進行標準化
5. IF 影像檔案格式不正確或無法讀取，THEN THE App_System SHALL 返回明確的錯誤訊息並記錄到 error_log
6. THE App_System SHALL 驗證提取的特徵向量維度為 32

### Requirement 3: 預測與分類

**User Story:** 作為臨床使用者，我希望系統能夠對結構性 MRI 進行 AD 分類預測，並提供預測信心分數

#### Acceptance Criteria

1. WHEN 特徵提取完成後，THE ML_Model SHALL 對標準化特徵進行預測
2. THE ML_Model SHALL 返回分類結果（"NC" 或 "AD"）
3. THE ML_Model SHALL 計算並返回預測的信心分數（probability score）
4. THE App_System SHALL 將預測結果和信心分數儲存到 AgentState 的 classification_result 欄位
5. THE App_System SHALL 記錄預測過程到 trace_log，包含使用的特徵數量和模型類型

### Requirement 4: Agent 節點整合

**User Story:** 作為系統架構師，我希望將 ML 模型整合為 LangGraph workflow 中的一個 agent 節點，以便與現有流程無縫協作

#### Acceptance Criteria

1. THE App_System SHALL 建立新的 agent 節點 "structural_mri_inference" 在 `/app/agents/` 目錄
2. THE structural_mri_inference 節點 SHALL 接收 AgentState 作為輸入並返回更新後的 AgentState
3. THE App_System SHALL 在 workflow.py 中註冊 structural_mri_inference 節點
4. THE App_System SHALL 支援條件式路由，根據輸入影像類型（結構性或功能性）選擇對應的推論節點
5. WHEN 使用者選擇結構性 MRI 分析時，THE workflow SHALL 執行 structural_mri_inference 節點而非現有的 fMRI inference 節點

### Requirement 5: 特徵重要性視覺化

**User Story:** 作為臨床研究者，我希望能夠視覺化模型的特徵重要性，以便理解哪些腦區對預測最重要

#### Acceptance Criteria

1. THE App_System SHALL 提取 ML_Model 的 feature_importances_ 屬性
2. THE App_System SHALL 將特徵重要性與對應的 ROI 名稱配對
3. THE App_System SHALL 生成 Top 10 重要特徵的橫條圖視覺化
4. THE App_System SHALL 將視覺化圖片儲存到輸出目錄並記錄路徑到 AgentState
5. THE App_System SHALL 在視覺化中標註每個 ROI 的重要性百分比

### Requirement 6: 腦區活化視覺化

**User Story:** 作為臨床使用者，我希望能夠在 3D 腦部影像上視覺化重要的 ROI 區域，以便直觀理解模型關注的腦區

#### Acceptance Criteria

1. THE App_System SHALL 使用 nilearn 的 plotting 功能在標準腦模板上標記重要 ROI
2. THE App_System SHALL 根據特徵重要性對 ROI 進行顏色編碼（重要性越高顏色越深）
3. THE App_System SHALL 生成多視角（矢狀面、冠狀面、軸向）的腦部視覺化
4. THE App_System SHALL 將視覺化圖片儲存並記錄路徑到 AgentState 的 visualization_paths
5. THE App_System SHALL 在視覺化中包含色條（colorbar）以指示重要性範圍

### Requirement 7: 雙模態支援與 UI 整合

**User Story:** 作為終端使用者，我希望能夠在 Streamlit UI 中選擇分析模式（結構性或功能性 MRI），並查看對應的分析結果

#### Acceptance Criteria

1. THE App_System SHALL 在 Streamlit 側邊欄新增 "Analysis Mode" 選擇器，選項包含 "Structural MRI" 和 "Functional MRI"
2. WHEN 使用者選擇 "Structural MRI" 模式時，THE App_System SHALL 顯示 ML_Model 選項而非深度學習模型選項
3. THE App_System SHALL 根據選擇的模式自動調整檔案搜尋模式（.nii.gz 用於結構性 MRI）
4. THE App_System SHALL 在結果頁面顯示模型特定的輸出（特徵重要性圖、ROI 視覺化等）
5. THE App_System SHALL 保持現有 fMRI 分析功能完全不受影響

### Requirement 8: 臨床報告生成

**User Story:** 作為臨床醫師，我希望系統能夠生成包含結構性 MRI 分析結果的臨床報告，以便用於診斷參考

#### Acceptance Criteria

1. THE App_System SHALL 整合 ML 模型預測結果到現有的 report_generator agent
2. THE report_generator SHALL 生成包含以下內容的報告：分類結果、信心分數、Top 10 重要腦區及其臨床意義
3. THE report_generator SHALL 支援中英文雙語報告生成
4. THE report_generator SHALL 引用 MODEL_OVERALL.md 中的臨床驗證資訊來解釋預測結果
5. THE report_generator SHALL 在報告中明確標註這是輔助診斷工具，不能單獨用於臨床診斷

### Requirement 9: 錯誤處理與日誌記錄

**User Story:** 作為系統維護者，我希望系統能夠妥善處理各種錯誤情況並提供詳細的日誌記錄，以便問題排查

#### Acceptance Criteria

1. THE App_System SHALL 捕獲所有模型載入、特徵提取、預測過程中的異常
2. WHEN 發生錯誤時，THE App_System SHALL 記錄詳細的錯誤訊息到 AgentState 的 error_log
3. THE App_System SHALL 記錄每個處理步驟的時間戳記和狀態到 trace_log
4. THE App_System SHALL 在 UI 中以使用者友善的方式顯示錯誤訊息
5. IF 關鍵錯誤發生，THEN THE App_System SHALL 提供恢復建議或替代方案

### Requirement 10: 效能與快取優化

**User Story:** 作為系統使用者，我希望分析過程能夠快速完成，避免不必要的重複計算

#### Acceptance Criteria

1. THE App_System SHALL 使用 Streamlit 的 @st.cache_resource 快取已載入的模型物件
2. THE App_System SHALL 使用 @st.cache_data 快取 AAL atlas 載入結果
3. THE App_System SHALL 在特徵提取過程中避免重複載入相同的影像檔案
4. THE App_System SHALL 在 5 秒內完成單一受試者的結構性 MRI 分析（不含影像載入時間）
5. THE App_System SHALL 提供進度指示器顯示分析進度
