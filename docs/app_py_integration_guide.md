# app.py Integration Guide for Structural MRI

本文件說明如何將結構性 MRI 分析功能整合到現有的 `app.py` 中。

## 需要的修改

### 1. 在檔案開頭加入 imports

```python
# 在現有 imports 後加入
from app.ui import (
    render_analysis_mode_selector,
    render_ml_model_info,
    render_structural_results,
    render_progress_indicator,
    render_error_message
)
```

### 2. 在側邊欄加入分析模式選擇器

在 `st.sidebar.header("Analysis Controls")` 之後加入：

```python
# Analysis Mode Selection
analysis_mode = render_analysis_mode_selector()

# Store in session state
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = analysis_mode
else:
    st.session_state.analysis_mode = analysis_mode
```

### 3. 根據模式顯示不同的模型選擇器

將現有的模型選擇邏輯修改為：

```python
if st.session_state.analysis_mode == "structural":
    # Show ML model info for structural MRI
    render_ml_model_info()
    selected_model_key = "random_forest"  # Fixed for structural
    selected_model_display = "Random Forest"
else:
    # 現有的深度學習模型選擇邏輯
    models = {"ShuffleNet": "shufflenet", "CapsNet": "capsnet", "MCADNNet": "mcadnnet"}
    # ... 現有程式碼 ...
```

### 4. 在 Start Analysis 按鈕處理中加入 analysis_mode

修改 `initial_state` 的建立：

```python
initial_state = {
    "subject_id": selected_subject,
    "fmri_scan_path": nii_path,
    "model_path": model_path,
    "model_name": selected_model_key,
    "analysis_mode": st.session_state.analysis_mode,  # 新增
}
```

### 5. 在結果顯示區域加入模式判斷

在 `if st.session_state.get("run_complete", False):` 區塊中：

```python
if st.session_state.get("run_complete", False):
    final_state = st.session_state["final_state"]
    report_ground_truth = st.session_state.get("ground_truth_label", "N/A")
    
    # 判斷分析模式
    analysis_mode = final_state.get("analysis_mode", "functional")
    
    if analysis_mode == "structural":
        # 顯示結構性 MRI 結果
        render_structural_results(final_state, report_ground_truth)
    else:
        # 現有的功能性 MRI 結果顯示
        # ... 現有程式碼 ...
```

### 6. 加入進度指示（可選）

在分析執行時：

```python
if st.session_state.get("analysis_running", False):
    if st.session_state.analysis_mode == "structural":
        # Structural MRI progress
        progress_bar, status_text = render_progress_indicator("loading_model", 20)
        # ... 更新進度 ...
    else:
        # 現有的進度顯示
        # ... 現有程式碼 ...
```

### 7. 錯誤處理

在 except 區塊中：

```python
except Exception as e:
    st.error("Analysis failed. Please try again.")
    
    # 顯示友善的錯誤訊息
    error_log = st.session_state.get("error_log", [str(e)])
    render_error_message(error_log)
```

## 完整的修改範例

以下是關鍵部分的完整範例：

```python
# === 側邊欄 ===
st.sidebar.header("Analysis Controls")

# 分析模式選擇
analysis_mode = render_analysis_mode_selector()
st.session_state.analysis_mode = analysis_mode

# 根據模式顯示不同的選項
if analysis_mode == "structural":
    render_ml_model_info()
    model_config = {
        "model_key": "random_forest",
        "model_display": "Random Forest",
        "model_path": None  # Will use default from config
    }
else:
    # 現有的深度學習模型選擇
    models = {"ShuffleNet": "shufflenet", "CapsNet": "capsnet"}
    selected_model_display = st.sidebar.selectbox(...)
    model_config = {
        "model_key": models[selected_model_display],
        "model_display": selected_model_display,
        "model_path": f"model/{models[selected_model_display]}/..."
    }

# === 分析執行 ===
if start_button:
    initial_state = {
        "subject_id": selected_subject,
        "fmri_scan_path": nii_path,
        "model_path": model_config["model_path"],
        "model_name": model_config["model_key"],
        "analysis_mode": analysis_mode,
        "trace_log": [],
        "error_log": []
    }
    
    final_state = app.invoke(initial_state)
    st.session_state["final_state"] = final_state

# === 結果顯示 ===
if st.session_state.get("run_complete", False):
    final_state = st.session_state["final_state"]
    analysis_mode = final_state.get("analysis_mode", "functional")
    
    if analysis_mode == "structural":
        render_structural_results(final_state, ground_truth_label)
    else:
        # 現有的功能性 MRI 結果顯示
        pass
```

## 測試步驟

1. 啟動應用：`streamlit run app.py`
2. 在側邊欄選擇 "Structural MRI (T1)"
3. 選擇一個受試者
4. 點擊 "Start Analysis"
5. 驗證結果顯示正確

## 注意事項

- 確保 `model/ml/final/` 目錄下有所有必要的模型檔案
- 第一次執行時，系統會自動下載 AAL atlas（需要網路連接）
- 結構性 MRI 分析通常比功能性 MRI 快（約 5-10 秒）
