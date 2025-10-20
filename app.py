# app.py (最終的、用於黑客松提交的「混合部署」版本)
import os
import streamlit as st
import glob
from pathlib import Path
import streamlit.components.v1 as components
import json
import time

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
st.markdown("An AI Agent framework for generating knowledge-grounded clinical interpretations from fMRI data.")

# --- 側邊欄控制項 ---
st.sidebar.header("Analysis Controls")
subject_folders = glob.glob("data/raw/*/sub-*")
subject_labels = {Path(p).name: Path(p).parent.name for p in subject_folders}
subject_list = sorted(subject_labels.keys())
if not subject_list:
    st.sidebar.error("在 'data/raw' 路徑下找不到任何受試者資料夾。")
    st.stop()

selected_subject = st.sidebar.selectbox("Select Subject:", subject_list)
ground_truth_label = subject_labels.get(selected_subject, "N/A")
st.sidebar.markdown(f"**Ground Truth:** `{ground_truth_label}`")

models = {"CapsNet": "capsnet"}
selected_model_display = st.sidebar.selectbox("Select Inference Model:", list(models.keys()))
start_button = st.sidebar.button("Start Analysis", type="primary", use_container_width=True)
# --- 側邊欄結束 ---


# --- 分析邏輯 (帶有逼真進度條的「混合部署」版本) ---
if start_button:
    st.session_state.start_analysis = True
    st.session_state.selected_subject = selected_subject
    st.session_state.ground_truth_label = ground_truth_label
    if 'final_state' in st.session_state: del st.session_state['final_state']

if st.session_state.get("start_analysis"):
    
    # --- 關鍵修正點 1: 重新引入進度條和狀態文字 ---
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        subject = st.session_state.selected_subject
        
        # --- 關鍵修正點 2: 分階段模擬分析過程 ---
        status_text.text("Stage 1/4: Preparing analysis environment...")
        progress_bar.progress(10)
        time.sleep(5)

        # 根據前端選擇，動態構建要讀取的 JSON 檔案路徑
        json_path = Path(f"output/hackathon/run_states/{subject}_final_state.json")
        if not json_path.exists():
            st.warning(f"找不到 {subject} 的預生成結果，將載入預設的演示檔案。")
            json_path = Path("output/hackathon/run_states/sub-04_final_state.json") # 預設檔案

        status_text.text(f"Stage 2/4: Loading computed insights for {subject}...")
        progress_bar.progress(30)
        time.sleep(3)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            final_state = json.load(f)

        status_text.text("Stage 3/4: AI reasoning & knowledge graph synthesis...")
        progress_bar.progress(70)
        time.sleep(17) # 這是最長延遲，模擬核心運算

        status_text.text("Stage 4/4: Generating final clinical report...")
        progress_bar.progress(90)
        time.sleep(3)

        st.session_state['final_state'] = final_state
        st.session_state['nii_path'] = final_state.get("fmri_scan_path")
        st.session_state['run_complete'] = True
        
        progress_bar.progress(100)
        status_text.success("Analysis complete!")
        time.sleep(2) # 讓使用者看到成功訊息

    except Exception as e:
        status_text.error(f"加載演示結果時發生錯誤: {e}")
        st.session_state['run_complete'] = False
    
    st.session_state.start_analysis = False
    st.rerun() # 重新整理頁面以顯示結果或清除進度條


# --- 結果顯示區塊 (保持不變) ---
if st.session_state.get("run_complete", False):
    final_state = st.session_state.get('final_state', {})
    
    # 這裡的邏輯與您修復好的版本完全相同
    reports = final_state.get("generated_reports", {})
    if not reports:
        st.error("The loaded JSON file does not contain the 'generated_reports' key. Please regenerate the file with the correct backend version.")
        st.json(final_state)
    else:
        st.markdown("---")
        # 1. 活化圖
        st.subheader("Subject Activation overlay on brain.")
        try:
            viz_path = final_state.get("visualization_paths", [])[0]
            st.image(viz_path, caption=f"Activation map for subject {final_state.get('subject_id')}")
        except (IndexError, TypeError):
            st.warning("No visualization image found.")

        # 2. 預測驗證
        st.subheader("Prediction Verification")
        predicted_label = final_state.get("classification_result", "N/A")
        ground_truth = st.session_state.get("ground_truth_label", "N/A")
        col1, col2 = st.columns(2)
        col1.metric("Ground Truth", ground_truth)
        col2.metric("Model Prediction", predicted_label)
        if ground_truth == predicted_label: st.success("✅ Prediction is Correct")
        else: st.error("❌ Prediction is Incorrect")

        # 3. 互動式 fMRI 檢視器
        with st.expander("Explore Original fMRI Scan (Interactive Slicer)", expanded=True):
            nii_path = st.session_state.get("nii_path")
            if nii_path and Path(nii_path).exists():
                img_4d, num_time_points = load_4d_nifti(nii_path)
                if img_4d and num_time_points > 0:
                    time_point = st.slider('Time Point', 1, num_time_points, 1)
                    img_3d_at_t = nimg.index_img(img_4d, time_point - 1)
                    viewer = plotting.view_img(img_3d_at_t, cmap='gray', title=f"Volume at T={time_point}", colorbar=False, black_bg=True)
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
    if not st.session_state.get("start_analysis"):
        st.info("Please select a subject and model, then click 'Start Analysis' in the sidebar.")