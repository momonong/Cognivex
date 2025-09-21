# app/main.py (Final Integrated Version)
import os
import streamlit as st
import glob 
from pathlib import Path
import streamlit.components.v1 as components

# --- 視覺化相關 ---
from nilearn import plotting
from nilearn import image as nimg

# ---### 變更點 1: 匯入 LangGraph App ###---
from app.graph.workflow import app

# ---### 變更點 2: 更新快取函式以處理 4D 數據 ###---
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

# --- STREAMLIT 前端介面 (文字與佈局保持您提供的版本) ---

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

# --- 結果顯示區塊 ---
if st.session_state.get('run_complete', False):
    final_state = st.session_state['final_state']
    report_ground_truth = st.session_state.get('ground_truth_label', "N/A")
    
    st.markdown("---")
    st.header("Analysis Results")
    
    # 活化圖與預測結果顯示 (保持不變)
    st.subheader("Subject Activation overlay on brain.")
    try:
        viz_path = final_state.get("visualization_paths", [])[0]
        st.image(viz_path, caption=f"Activation map for subject {selected_subject}")
    except Exception as e:
        st.error(f"Cannot display image. Path is missing or invalid: {e}")
    
    predicted_label = final_state.get("classification_result", "N/A")
    st.subheader("Prediction Verification")
    col1, col2 = st.columns(2)
    col1.metric("Ground Truth", report_ground_truth)
    col2.metric("Model Prediction", predicted_label)
    if report_ground_truth == predicted_label: st.success("✅ Prediction is Correct")
    else: st.error("❌ Prediction is Incorrect")
    
    # ---### 變更點 3: 整合最終版互動式檢視器 ###---
    is_expanded_default = st.session_state.get('viewer_expanded', False)
    with st.expander("Explore Original fMRI Scan (Interactive Slicer)", expanded=is_expanded_default):
        nii_path = st.session_state.get('nii_path')
        if nii_path and Path(nii_path).exists():
            # 呼叫新的 4D 數據載入函數
            img_4d, num_time_points = load_4d_nifti(nii_path)
            
            if img_4d and num_time_points > 0:
                # 顯示時間軸滑桿，讓使用者可以選擇
                # 為了讓使用者介面從 1 開始，我們設定 min_value=1, max_value=num_time_points
                selected_time_point_display = st.slider(
                    'Time Point (Volume)', 
                    min_value=1, 
                    max_value=num_time_points, 
                    value=1,
                    help=f"This fMRI scan has {num_time_points} volumes."
                )
                
                # 在後端處理時，我們需要將使用者的 1-based 索引轉換為 0-based 索引
                selected_time_point_index = selected_time_point_display - 1
                
                # 根據選擇的時間點，產生對應的 3D 檢視器
                img_3d_at_t = nimg.index_img(img_4d, selected_time_point_index)

                viewer = plotting.view_img(
                    img_3d_at_t, 
                    bg_img=None, 
                    cmap='gray', 
                    threshold=None, 
                    title=f"Volume at T={selected_time_point_display}", # 顯示 1-based 的時間點
                    resampling_interpolation='nearest',
                    colorbar=False,
                    annotate=True,
                    black_bg=True
                )
                
                components.html(viewer.html, height=600, scrolling=False)
        else:
            st.warning("Could not find the original NIfTI file for this viewer.")

    # 中英文報告分頁 (保持不變)
    reports = final_state.get("generated_reports", {})
    report_en = reports.get("en", "No English report was generated.")
    report_zh = reports.get("zh", "沒有生成中文報告。")
    
    tab_en, tab_zh = st.tabs(["English Report", "中文報告"])
    with tab_en:
        st.subheader("Clinical Report (English)")
        st.markdown(report_en, unsafe_allow_html=True)
    with tab_zh:
        st.subheader("臨床分析報告 (繁體中文)")
        st.markdown(report_zh, unsafe_allow_html=True)
else:
    st.info("Please select a subject and model, then click 'Start Analysis' in the sidebar to view results.")