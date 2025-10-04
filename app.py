# app/main.py (Final Integrated Version)
import os
import streamlit as st
import glob 
from pathlib import Path
import streamlit.components.v1 as components
import pandas as pd  # <--- 新增的匯入

# --- 視覺化相關 ---
from nilearn import plotting
from nilearn import image as nimg

# --- LangGraph App ---
from app.graph.workflow import app

# --- 快取函式 ---
@st.cache_resource(show_spinner="正在載入並處理 NIfTI 檔案...")
def load_4d_nifti(path: str):
    """
    載入 4D NIfTI 檔案並回傳 nilearn 影像物件和時間點總數。
    """
    try:
        img_4d = nimg.load_img(path)
        num_time_points = img_4d.shape[3]
        return img_4d, num_time_points
    except Exception as e:
        st.error(f"載入或處理 4D 檔案失敗: {path}. 錯誤: {e}")
        return None, 0

# --- STREAMLIT 前端介面 ---

st.set_page_config(page_title="fMRI Analysis Framework", layout="wide")
st.title("Explainable fMRI Analysis for Alzheimer's Disease")
st.markdown("An agent-based framework for generating knowledge-grounded clinical interpretations from fMRI data.")

# 初始化 session state 來追蹤分析狀態
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
# --- 側邊欄控制項 (保持不變) ---
st.sidebar.header("Analysis Controls")
subject_folders = glob.glob("data/raw/*/sub-*")
subject_labels = {} 
for folder_path in subject_folders:
    parts = folder_path.split(os.sep) 
    if len(parts) >= 4:
        subject_id = parts[-1]
        label = parts[-2]      
        subject_labels[subject_id] = label
subject_list = sorted(subject_labels.keys())
if not subject_list:
    st.sidebar.error("在 'data/raw' 路徑下找不到任何 'AD/sub-XX' 或 'NC/sub-XX' 資料夾。")
    st.stop()
selected_subject = st.sidebar.selectbox('Select Subject:', subject_list, disabled=st.session_state.analysis_running)
ground_truth_label = subject_labels.get(selected_subject, "N/A")
st.sidebar.markdown(f"**Ground Truth:** `{ground_truth_label}`")
models = ["CapsNetRNN"] 
selected_model = st.sidebar.selectbox('Select Inference Model:', models, disabled=st.session_state.analysis_running)
start_button = st.sidebar.button('Start Analysis', type="primary", disabled=st.session_state.analysis_running)
st.sidebar.markdown("---") 
adni_acknowledgement = """
<div style="font-size: 0.75rem; color: grey;">
Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: <a href="http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf" target="_blank">ADNI Acknowledgement List</a>.
</div>
"""
st.sidebar.markdown(adni_acknowledgement, unsafe_allow_html=True)


# --- 分析邏輯 ---
if start_button:
    st.session_state.analysis_running = True
    st.rerun()

# 如果狀態為正在分析，則執行分析流程
if st.session_state.get('analysis_running', False):
    st.session_state.viewer_expanded = True # 確保 viewer 預設展開
    with st.spinner('Analysis in progress... This may take a few minutes. Please wait.'):
        try:
            model_paths_map = { "CapsNetRNN": "model/capsnet/best_capsnet_rnn.pth" }
            model_path = model_paths_map.get(selected_model)
            if not model_path: raise FileNotFoundError(f"找不到模型 '{selected_model}' 的路徑設定。")
            
            nii_search_pattern = f"data/raw/*/{selected_subject}/*.nii.gz"
            nii_file_list = glob.glob(nii_search_pattern)
            if not nii_file_list: raise FileNotFoundError(f"找不到受試者 '{selected_subject}' 的 .nii.gz 檔案。")
            nii_path = nii_file_list[0]
            st.info(f"Files found:\n- NIfTI: {nii_path}\n- Model: {model_path}")

            initial_state = {
                "subject_id": selected_subject,
                "fmri_scan_path": nii_path,
                "model_path": model_path,
            }
            final_state = app.invoke(initial_state)
            
            if final_state:
                st.session_state['nii_path'] = nii_path 
                st.session_state['final_state'] = final_state
                st.session_state['ground_truth_label'] = ground_truth_label
                st.session_state['run_complete'] = True
            else:
                st.error("Analysis finished but the agent returned no content.")
                st.session_state['run_complete'] = False
                
        except Exception as e:
            st.error("Please try again later.")
            st.error(f"Critical error occurred during analysis: {e}")
            st.session_state['run_complete'] = False
        finally:
            # 分析結束後，將狀態設回 False 並重新整理，以重新啟用側邊欄
            st.session_state.analysis_running = False
            st.rerun()

# --- 結果顯示區塊 (儀表板版本) ---
if st.session_state.get('run_complete', False):
    final_state = st.session_state['final_state']
    
    # 從新的 JSON 結構中獲取報告
    report_data = final_state.get("final_report_json")
    
    if not report_data:
        st.error("分析完成，但未生成結構化報告。請檢查 Agent 狀態。")
        st.json(final_state) # 顯示原始狀態以供偵錯
        st.stop()

    st.markdown("---")
    st.header("📊 Analysis Dashboard")

    # --- 區塊 1: 診斷摘要 (Diagnostic Summary) ---
    st.subheader("📋 Diagnostic Summary")
    summary = report_data.get("diagnostic_summary", {})
    pred = summary.get("prediction", "N/A")
    truth = st.session_state.get('ground_truth_label', "N/A")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Ground Truth", truth)
    col2.metric("Model Prediction", pred)
    with col3:
        if truth == pred:
            st.success("✅ Prediction Correct")
        else:
            st.error("❌ Prediction Incorrect")

    with st.expander("Show Key Finding / 核心發現", expanded=True):
        finding_en = summary.get("key_finding", {}).get("en", "N/A")
        finding_zh = summary.get("key_finding", {}).get("zh", "N/A")
        st.write(f"**EN:** {finding_en}")
        st.write(f"**ZH:** {finding_zh}")

    # --- 區塊 2: 視覺化與互動檢視器 ---
    st.subheader("🧠 Brain Scans & Activation Maps")
    col1, col2 = st.columns([0.9, 1.1]) 
    with col1:
        st.markdown("**Activation Map**")
        try:
            viz_path = report_data.get("visualization_paths", {}).get("brain_map_png")
            if viz_path and Path(viz_path).exists():
                st.image(viz_path, caption=f"Activation map for subject {selected_subject}")
            else:
                st.warning("找不到活化圖影像。")
        except Exception as e:
            st.error(f"無法顯示影像: {e}")

    with col2:
        with st.expander("Explore Original fMRI Scan (Interactive Slicer)", expanded=True):
            nii_path = st.session_state.get('nii_path')
            if nii_path and Path(nii_path).exists():
                img_4d, num_time_points = load_4d_nifti(nii_path)
                if img_4d and num_time_points > 0:
                    selected_time_point_display = st.slider(
                        'Time Point (Volume)', min_value=1, max_value=num_time_points, value=1
                    )
                    selected_time_point_index = selected_time_point_display - 1
                    img_3d_at_t = nimg.index_img(img_4d, selected_time_point_index)
                    viewer = plotting.view_img(
                        img_3d_at_t, cmap='gray', title=f"Volume at T={selected_time_point_display}",
                        colorbar=False, black_bg=True
                    )
                    components.html(viewer.html, height=450, scrolling=False)
            else:
                st.warning("找不到原始 NIfTI 檔案。")

    # --- 區塊 3: 腦區活化分析 (Activation Analysis) ---
    st.subheader("📈 Brain Region Activation Analysis")
    analysis_data = report_data.get("activation_analysis", {})
    regions_list = analysis_data.get("regions", [])
    if regions_list:
        regions_df = pd.DataFrame(regions_list)
        # 設定欄位順序以獲得更好的可讀性
        display_columns = ["name", "network", "function", "activation"]
        regions_df = regions_df[display_columns]
        st.dataframe(regions_df, use_container_width=True)
    else:
        st.info("No significant brain region activations were identified.")

    # --- 區塊 4: 臨床推理 (Clinical Reasoning) ---
    st.subheader("🔬 Clinical Reasoning")
    reasoning = report_data.get("clinical_reasoning", {})
    narrative_en = reasoning.get("narrative", {}).get("en", "N/A")
    narrative_zh = reasoning.get("narrative", {}).get("zh", "N/A")
    
    tab_en, tab_zh = st.tabs(["English Reasoning", "中文推理說明"])
    with tab_en:
        st.markdown(narrative_en)
    with tab_zh:
        st.markdown(narrative_zh)

else:
    st.info("Please select a subject and model, then click 'Start Analysis' in the sidebar to view results.")