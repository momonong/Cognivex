import streamlit as st
import glob
from pathlib import Path
import streamlit.components.v1 as components
import json
import time
import random

# --- 視覺化與快取函式 ---
from nilearn import plotting
from nilearn import image as nimg


@st.cache_resource(show_spinner="正在載入並處理 NIfTI 檔案...")
def load_4d_nifti(path: str):
    try:
        img_4d = nimg.load_img(path)
        return img_4d, img_4d.shape[3]
    except Exception as e:
        st.error(f"載入或處理 4D 檔案失敗: {path}. 錯誤: {e}")
        return None, 0


# --- STREAMLIT 前端介面 ---
st.set_page_config(page_title="Cognivex fMRI Analysis", layout="wide")
st.title("Cognivex: Explainable AI for Alzheimer's Analysis")
st.markdown(
    "An AI Agent framework for generating knowledge-grounded clinical interpretations from fMRI data."
)

# --- 側邊欄控制項 ---
st.sidebar.header("Analysis Controls")

# 初始化/獲取分析狀態
is_running = st.session_state.get("analysis_running", False)

# 受試者選擇
subject_folders = glob.glob("data/raw/*/sub-*")
subject_labels = {Path(p).name: Path(p).parent.name for p in subject_folders}
subject_list = sorted(subject_labels.keys())
if not subject_list:
    st.sidebar.error("在 'data/raw' 路徑下找不到任何受試者資料夾。")
    st.stop()

# 根據分析狀態決定選擇器的行為
current_subject = st.session_state.get("selected_subject", subject_list[0])
default_index = (
    subject_list.index(current_subject) if current_subject in subject_list else 0
)
selected_subject = st.sidebar.selectbox(
    "Select Subject:",
    [current_subject] if is_running else subject_list,
    index=0 if is_running else default_index,
    disabled=is_running,
    help="Subject selection is locked during analysis.",
)
ground_truth_label = subject_labels.get(selected_subject, "N/A")
st.sidebar.markdown(f"**Ground Truth:** `{ground_truth_label}`")

# 模型選擇
models = {"CapsNet": "capsnet"}

current_model = st.session_state.get("selected_model_display")
model_list = list(models.keys())
if current_model and current_model in model_list:
    default_model_index = model_list.index(current_model)
else:
    default_model_index = 0

if is_running:
    # 分析中：顯示當前模型但禁用
    selected_model_display = st.sidebar.selectbox(
        "Select Inference Model:",
        [current_model or "N/A"],
        disabled=True,
        help="Model selection is locked during analysis.",
    )
else:
    # 正常狀態：正常選擇
    selected_model_display = st.sidebar.selectbox(
        "Select Inference Model:",
        model_list,
        index=default_model_index,
        help="Choose the neural network model for fMRI classification.",
    )

selected_model_key = models[selected_model_display]

# 顯示模型詳細信息
model_info = {
    "capsnet": {
        "type": "3D Capsule Network",
        "description": "Advanced neural network with capsule layers for spatial relationships",
        "best_for": "Complex 3D fMRI patterns, part-whole relationships",
    },
    "mcadnnet": {
        "type": "2D Convolutional Neural Network",
        "description": "Traditional CNN architecture for 2D slice analysis",
        "best_for": "2D brain slice patterns, computational efficiency",
    },
}

if selected_model_key in model_info:
    info = model_info[selected_model_key]
    st.sidebar.caption(f"**Model Type:** {info['type']}")
    st.sidebar.caption(f"**Description:** {info['description']}")
    st.sidebar.caption(f"**Best for:** {info['best_for']}")

# 檢查是否有參數變更，如果有則重置分析狀態
prev_subject = st.session_state.get("selected_subject")
prev_model = st.session_state.get("selected_model_key")
if (prev_subject and prev_subject != selected_subject) or (
    prev_model and prev_model != selected_model_key
):
    # 參數有變更，重置完成狀態以允許重新分析
    st.session_state.run_complete = False
    # 清除舊的結果
    if "final_state" in st.session_state:
        del st.session_state["final_state"]
    if "nii_path" in st.session_state:
        del st.session_state["nii_path"]

# 儲存當前選擇到 session state
st.session_state.selected_subject = selected_subject
st.session_state.selected_model_display = selected_model_display
st.session_state.selected_model_key = selected_model_key
st.session_state.ground_truth_label = ground_truth_label

# --- 關鍵修正點 1: 還原動態按鈕狀態 ---
if is_running:
    # 分析中：禁用主按鈕 + Force Stop
    st.sidebar.button(
        "Analysis Running...",
        disabled=True,
        use_container_width=True,
        help="Analysis in progress...",
    )
    if st.sidebar.button(
        "Force Stop Analysis", type="secondary", use_container_width=True
    ):
        st.session_state.analysis_running = False
        st.session_state.run_complete = False
        st.sidebar.warning("Analysis has been stopped.")
        st.rerun()
    start_button = False
else:
    # 正常狀態：正常開始按鈕
    start_button = st.sidebar.button(
        "Start Analysis",
        type="primary",
        use_container_width=True,
        help=f"Start analysis for {selected_subject}",
    )

# --- 關鍵修正點 2: 還原 ADNI Disclaimer ---
st.sidebar.markdown("---")
adni_acknowledgement = """
<div style="font-size: 0.75rem; color: grey;">
Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: <a href="http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf" target="_blank">ADNI Acknowledgement List</a>.
</div>
"""
st.sidebar.markdown(adni_acknowledgement, unsafe_allow_html=True)
# --- 側邊欄結束 ---


# --- 分析邏輯 (帶有逼真進度條的「混合部署」版本) ---
if start_button:
    st.session_state.analysis_running = True
    st.session_state.run_complete = False
    if "final_state" in st.session_state:
        del st.session_state["final_state"]
    st.rerun()

if st.session_state.get("analysis_running") and not st.session_state.get(
    "run_complete"
):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        subject = st.session_state.selected_subject

        # --- Stage 1: Preparation ---
        status_text.text("Stage 1/5: Preparing analysis environment...")
        progress_bar.progress(10)
        time.sleep(2 + random.uniform(0.5, 1.0))  # Add randomness

        # --- Stage 2: Loading Data ---
        json_path = Path(f"output/hackathon/run_states/{subject}_final_state.json")
        if not json_path.exists():
            st.warning(f"Demo data for {subject} not found. Loading default example.")
            json_path = Path("output/hackathon/run_states/sub-04_final_state.json")

        status_text.text(f"Stage 2/5: Loading pre-computed insights for {subject}...")
        progress_bar.progress(25)
        time.sleep(3 + random.uniform(0.5, 1.5))

        with open(json_path, "r", encoding="utf-8") as f:
            final_state = json.load(f)

        # --- Stage 3: Core AI Reasoning (broken into two parts) ---
        status_text.text("Stage 3/5: Simulating deep learning model analysis...")
        progress_bar.progress(40)
        time.sleep(4 + random.uniform(1.0, 2.0))

        status_text.text("Stage 4/5: Synthesizing with knowledge graph...")
        progress_bar.progress(75)
        time.sleep(4 + random.uniform(1.0, 2.5))  # The "heaviest" step

        # --- Stage 5: Report Generation ---
        status_text.text("Stage 5/5: Generating final clinical report...")
        progress_bar.progress(90)
        time.sleep(2 + random.uniform(0.5, 1.0))

        # Finalizing state
        st.session_state["final_state"] = final_state
        st.session_state["nii_path"] = final_state.get("fmri_scan_path")
        st.session_state["run_complete"] = True

        progress_bar.progress(100)
        status_text.success("Analysis complete!")
        time.sleep(1.5)  # Let user see the success message

    except Exception as e:
        status_text.error(f"Error loading demo data: {e}")
        st.session_state["run_complete"] = False

    # End the analysis state and rerun the page to show results
    st.session_state.analysis_running = False
    st.rerun()

# --- 結果顯示區塊 (與您之前的版本完全相同) ---
if st.session_state.get("run_complete", False):
    final_state = st.session_state.get("final_state", {})
    reports = final_state.get("generated_reports", {})
    if not reports:
        st.error(
            "The loaded JSON file does not contain 'generated_reports'. Please regenerate the file."
        )
        st.json(final_state)
    else:
        st.markdown("---")
        # 1. 活化圖
        st.subheader("Subject Activation overlay on brain.")
        try:
            viz_path = final_state.get("visualization_paths", [])[0]
            st.image(
                viz_path,
                caption=f"Activation map for subject {final_state.get('subject_id')}",
            )
        except (IndexError, TypeError):
            st.warning("No visualization image found.")
        # 2. 預測驗證
        st.subheader("Prediction Verification")
        predicted_label = final_state.get("classification_result", "N/A")
        ground_truth = subject_labels.get(final_state.get("subject_id"), "N/A")
        col1, col2 = st.columns(2)
        col1.metric("Ground Truth", ground_truth)
        col2.metric("Model Prediction", predicted_label)
        if ground_truth == predicted_label:
            st.success("✅ Prediction is Correct")
        else:
            st.error("❌ Prediction is Incorrect")
        # 3. 互動式 fMRI 檢視器
        with st.expander(
            "Explore Original fMRI Scan (Interactive Slicer)", expanded=True
        ):
            nii_path = st.session_state.get("nii_path")
            if nii_path and Path(nii_path).exists():
                img_4d, num_time_points = load_4d_nifti(nii_path)
                if img_4d and num_time_points > 0:
                    time_point = st.slider("Time Point", 1, num_time_points, 1)
                    img_3d_at_t = nimg.index_img(img_4d, time_point - 1)
                    viewer = plotting.view_img(
                        img_3d_at_t,
                        cmap="gray",
                        title=f"Volume at T={time_point}",
                        colorbar=False,
                        black_bg=True,
                    )
                    components.html(viewer.html, height=500, scrolling=False)
            else:
                st.warning("Could not find the original NIfTI file path.")
        # 4. 中英文報告分頁
        st.subheader("Clinical Report")
        report_en = reports.get("en", "No English report was generated.")
        report_zh = reports.get("zh", "沒有生成中文報告。")

        tab_en, tab_zh = st.tabs(["English Report", "中文報告"])
        with tab_en:
            st.markdown(report_en, unsafe_allow_html=True)
        with tab_zh:
            st.markdown(report_zh, unsafe_allow_html=True)
else:
    if not st.session_state.get("analysis_running"):
        st.info(
            "Please select a subject and model, then click 'Start Analysis' in the sidebar."
        )
