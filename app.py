import os
import streamlit as st
import json
import glob 

# --- 可視化相關 (保持不變) ---
from nilearn import plotting
from nilearn import image as nimg
import matplotlib.pyplot as plt 

# ---【修改 1】替換 Imports ---
from app.graph.workflow import app

# --- 快取函式 (保持不變) ---
@st.cache_data
def load_and_process_nifti(path: str):
    """
    載入一個 NIfTI 檔案，並將其轉換為一個 3D 平均影像。
    """
    img_4d = nimg.load_img(path)
    img_3d = nimg.mean_img(img_4d, copy_header=True)
    return img_3d

# --- STREAMLIT 前端介面 (保持不變) ---

st.set_page_config(page_title="fMRI Analysis Framework", layout="wide")
st.title("Explainable fMRI Analysis for Alzheimer's Disease")
st.markdown("An agent-based framework for generating knowledge-grounded clinical interpretations from fMRI data.")

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
selected_subject = st.sidebar.selectbox('Select Subject:', subject_list)
ground_truth_label = subject_labels.get(selected_subject, "N/A")
st.sidebar.markdown(f"**Ground Truth:** `{ground_truth_label}`")
models = ["CapsNetRNN"] 
selected_model = st.sidebar.selectbox('Select Inference Model:', models)
start_button = st.sidebar.button('Start Analysis', type="primary")
st.sidebar.markdown("---") 
adni_acknowledgement = """
<div style="font-size: 0.75rem; color: grey;">
Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: <a href="http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf" target="_blank">ADNI Acknowledgement List</a>.
</div>
"""
st.sidebar.markdown(adni_acknowledgement, unsafe_allow_html=True)


# --- 分析與結果顯示邏輯 ---
if start_button:
    st.session_state.viewer_expanded = True
    with st.spinner('Analysis in progress... This may take a few minutes. Please wait.'):
        try:
            # --- 這部分尋找檔案的邏輯保持不變 ---
            model_paths_map = { "CapsNetRNN": "model/capsnet/best_capsnet_rnn.pth" }
            model_path = model_paths_map.get(selected_model)
            if not model_path: raise FileNotFoundError(f"找不到模型 '{selected_model}' 的路徑設定。")
            
            nii_search_pattern = f"data/raw/*/{selected_subject}/*.nii.gz"
            nii_file_list = glob.glob(nii_search_pattern)
            if not nii_file_list: raise FileNotFoundError(f"找不到受試者 '{selected_subject}' 的 .nii.gz 檔案。")
            nii_path = nii_file_list[0]
            st.info(f"Files found:\n- NIfTI: {nii_path}\n- Model: {model_path}")

            # ---【修改 2】替換後端呼叫邏輯 ---
            # 1. 準備 LangGraph 的輸入
            initial_state = {
                "subject_id": selected_subject,
                "fmri_scan_path": nii_path,
                "model_path": model_path,
            }
            # 2. 呼叫 LangGraph app (使用 .invoke() 來模擬原本的同步行為)
            final_state = app.invoke(initial_state)
            
            # 3. 將回傳的 state (一個字典) 存入 session
            if final_state:
                st.session_state['nii_path'] = nii_path 
                st.session_state['final_state'] = final_state # 我們將整個 state 字典存起來
                st.session_state['ground_truth_label'] = ground_truth_label
                st.session_state['run_complete'] = True
            else:
                st.error("Analysis finished but the agent returned no content.")
                st.session_state['run_complete'] = False
            # --- 修改結束 ---
                
        except Exception as e:
            st.error("Please try again later.")
            st.error(f"Critical error occurred during analysis: {e}")
            st.session_state['run_complete'] = False


# --- 結果顯示區塊 ---
if 'run_complete' in st.session_state and st.session_state['run_complete']:
    # ---【修改 3】更新結果讀取方式 ---
    # 從 session state 讀取我們儲存的 final_state 字典
    final_state = st.session_state['final_state']
    report_ground_truth = st.session_state.get('ground_truth_label', "N/A")
    
    st.markdown("---")
    st.header("Analysis Results")
    st.subheader("Subject Activation overlay on brain.")
    try:
        # 從字典中安全地獲取數據
        viz_path = final_state.get("visualization_paths", [])[0]
        st.image(viz_path, caption=f"Activation map for subject {selected_subject}")
    except (IndexError, TypeError, FileNotFoundError) as e:
        st.error(f"Cannot display image. Path is missing or invalid: {e}")
    
    # 預測結果比對
    predicted_label = final_state.get("classification_result", "N/A")
    st.subheader("Prediction Verification")
    col1, col2 = st.columns(2)
    col1.metric("Ground Truth", report_ground_truth)
    col2.metric("Model Prediction", predicted_label)
    if report_ground_truth == predicted_label: st.success("✅ Prediction is Correct")
    else: st.error("❌ Prediction is Incorrect")
    
    # 互動式原始檔案檢視器 (保持不變)
    is_expanded_default = st.session_state.get('viewer_expanded', False)
    with st.expander("Explore Original fMRI Scan (Interactive Slicer)", expanded=is_expanded_default):
        # ... (這部分的程式碼與您原本的完全相同，無需修改) ...
        nii_path = st.session_state.get('nii_path')
        if nii_path and os.path.exists(nii_path):
            img_3d = load_and_process_nifti(nii_path)
            st.info("Use the sliders below to freely explore the subject's brain anatomy.")
            coords = img_3d.shape
            x_c, y_c, z_c = st.columns(3)
            x = x_c.slider('X (Sagittal)', int(-coords[0]/2), int(coords[0]/2), 0)
            y = y_c.slider('Y (Coronal)', int(-coords[1]/2), int(coords[1]/2), 0)
            z = z_c.slider('Z (Axial)', int(-coords[2]/2), int(coords[2]/2), 0)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.patch.set_facecolor('black')
            plotting.plot_anat(img_3d, display_mode='x', cut_coords=[x], axes=axes[0], title=f"Sagittal (X={x})", black_bg=True)
            plotting.plot_anat(img_3d, display_mode='y', cut_coords=[y], axes=axes[1], title=f"Coronal (Y={y})", black_bg=True)
            plotting.plot_anat(img_3d, display_mode='z', cut_coords=[z], axes=axes[2], title=f"Axial (Z={z})", black_bg=True)
            st.pyplot(fig)
        else:
            st.warning("Could not find the original NIfTI file for this viewer.")

    # 中英文報告分頁
    # 從 final_state 中的 generated_reports 字典讀取報告
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