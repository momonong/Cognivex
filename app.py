import os
import streamlit as st
import json
import glob 

# --- 核心功能 Import ---
from nilearn import plotting
from nilearn import image as nimg
import streamlit.components.v1 as components
import matplotlib.pyplot as plt 

# --- 從您的 Agent 腳本匯入 FinalReport ---
# 確保這個路徑是您專案中定義 FinalReport Class 的正確位置
from agents.sub_agents.final_report.agent import FinalReport
# --- 從您的後端腳本匯入執行函式 ---
from backend.backend_runner import run_analysis_sync

# --- 【整合進來】從獨立檢視器複製過來的快取函式 ---
@st.cache_data
def load_and_process_nifti(path: str):
    """
    載入一個 NIfTI 檔案，並將其轉換為一個 3D 平均影像。
    使用快取來避免重複載入和處理，提升滑桿互動的流暢度。
    """
    img_4d = nimg.load_img(path)
    img_3d = nimg.mean_img(img_4d)
    return img_3d

# --- STREAMLIT 前端介面 ---

st.set_page_config(page_title="fMRI Analysis Framework", layout="wide")
st.title("Explainable fMRI Analysis for Alzheimer's Disease")
st.markdown("An agent-based framework for generating knowledge-grounded clinical interpretations from fMRI data.")

# --- 側邊欄控制項 ---
st.sidebar.header("Analysis Controls")

# 動態尋找 Subject 並同時儲存其真實標籤
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
    subject_list = ["sub-14"] 
    subject_labels["sub-14"] = "Unknown" 

selected_subject = st.sidebar.selectbox('Select Subject:', subject_list)
ground_truth_label = subject_labels.get(selected_subject, "N/A")
st.sidebar.markdown(f"**Ground Truth:** `{ground_truth_label}`")

models = ["CapsNetRNN"] 
selected_model = st.sidebar.selectbox('Select Inference Model:', models)

start_button = st.sidebar.button('Start Analysis', type="primary")

# --- ADNI 致謝詞 ---
st.sidebar.markdown("---") 
adni_acknowledgement = """
<div style="font-size: 0.75rem; color: grey;">
Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: <a href="http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf" target="_blank">ADNI Acknowledgement List</a>.
</div>
"""
st.sidebar.markdown(adni_acknowledgement, unsafe_allow_html=True)


# --- 分析與結果顯示邏輯 ---
if start_button:
    with st.spinner('Analysis in progress... This may take a minute. Please wait.'):
        try:
            model_paths_map = { "CapsNetRNN": "model/capsnet/best_capsnet_rnn.pth" }
            model_path = model_paths_map.get(selected_model)
            if not model_path: raise FileNotFoundError(f"找不到模型 '{selected_model}' 的路徑設定。")
            
            nii_search_pattern = f"data/raw/*/{selected_subject}/*.nii.gz"
            nii_file_list = glob.glob(nii_search_pattern)
            if not nii_file_list: raise FileNotFoundError(f"找不到受試者 '{selected_subject}' 的 .nii.gz 檔案。")
            nii_path = nii_file_list[0]

            st.info(f"Files found:\n- NIfTI: {nii_path}\n- Model: {model_path}")

            result_json_string = run_analysis_sync(selected_subject, nii_path, model_path)
            
            if result_json_string:
                final_report_obj = FinalReport.model_validate_json(result_json_string)
                
                st.session_state['nii_path'] = nii_path 
                st.session_state['final_report'] = final_report_obj
                st.session_state['run_complete'] = True
            else:
                st.error("Analysis finished but the agent returned no content.")
                st.session_state['run_complete'] = False

        except Exception as e:
            st.error("Please retry after at least 1 minute.")
            st.error(f"Critical error occurred during analysis: {e}")
            st.session_state['run_complete'] = False


# --- 結果顯示區塊 ---
if 'run_complete' in st.session_state and st.session_state['run_complete']:
    report_data = st.session_state['final_report']
    
    st.markdown("---")
    st.header("Analysis Results")
    
    # 預測結果與真實標籤的比對
    predicted_label = "Unknown"
    if "Alzheimer's Disease (AD)" in report_data.final_report_markdown: predicted_label = "AD"
    elif "Healthy Control (CN)" in report_data.final_report_markdown: predicted_label = "NC"
    st.subheader("Prediction Verification")
    col1, col2 = st.columns(2)
    col1.metric("Ground Truth", ground_truth_label)
    col2.metric("Model Prediction", predicted_label)
    if ground_truth_label == predicted_label: st.success("✅ Prediction is Correct")
    else: st.error("❌ Prediction is Incorrect")
    
    with st.expander("Explore Original fMRI Scan (Interactive Slicer)"):
        # 從 session_state 取得原始 fMRI 檔案的路徑
        nii_path = st.session_state.get('nii_path')
        
        if nii_path and os.path.exists(nii_path):
            # 呼叫快取函式來載入影像
            img_3d = load_and_process_nifti(nii_path)
            
            st.info("Use the sliders below to freely explore the subject's brain anatomy.")
            
            coords = img_3d.shape
            # 建立三個並排的滑桿，更節省空間
            col1, col2, col3 = st.columns(3)
            with col1:
                x = st.slider('X (Sagittal)', int(-coords[0]/2), int(coords[0]/2), 0, key='slice_x')
            with col2:
                y = st.slider('Y (Coronal)', int(-coords[1]/2), int(coords[1]/2), 0, key='slice_y')
            with col3:
                z = st.slider('Z (Axial)', int(-coords[2]/2), int(coords[2]/2), 0, key='slice_z')
            
            selected_coords = (x, y, z)
            
            # 使用 Nilearn 繪圖
            fig, axes = plt.subplots(1, 1, figsize=(12, 6))
            plotting.plot_anat(
                img_3d,
                display_mode='ortho',
                cut_coords=selected_coords,
                axes=axes,
                title=f"Orthogonal Views at {selected_coords}",
                draw_cross=True,
                annotate=True,
                black_bg=True,
            )
            fig.patch.set_facecolor('black')
            plt.rcParams['text.color'] = 'white'
            st.pyplot(fig)
        else:
            st.warning("Could not find the original NIfTI file for this viewer.")


    # 中英文報告分頁
    tab_en, tab_zh = st.tabs(["English Report", "中文報告"])
    with tab_en:
        st.subheader("Clinical Report (English)")
        st.markdown(report_data.final_report_markdown, unsafe_allow_html=True)
    with tab_zh:
        st.subheader("臨床分析報告 (繁體中文)")
        st.markdown(report_data.final_report_chinese, unsafe_allow_html=True)
else:
    st.info("Please select a subject and model, then click 'Start Analysis' in the sidebar to view results.")