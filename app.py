import os
import streamlit as st
import json
import glob 

from agents.sub_agents.final_report.agent import FinalReport
from backend.backend_runner import run_analysis_sync

# --- STREAMLIT Frontend Interface ---

st.set_page_config(page_title="fMRI Analysis Framework", layout="wide")
st.title("Explainable fMRI Analysis for Alzheimer's Disease")
st.markdown("An agent-based framework for generating knowledge-grounded clinical interpretations from fMRI data.")

# --- Sidebar Controls ---
st.sidebar.header("Analysis Controls")

subject_folders = glob.glob("data/raw/*/sub-*")
subject_labels = {} # { 'sub-14': 'AD', 'sub-21': 'NC', ... }

for folder_path in subject_folders:
    # folder_path looks like: 'data/raw/AD/sub-14'
    parts = folder_path.split(os.sep) # 
    if len(parts) >= 4:
        subject_id = parts[-1] # 'sub-14'
        label = parts[-2]      # 'AD' or 'NC'
        subject_labels[subject_id] = label

subject_list = sorted(subject_labels.keys())

if not subject_list:
    st.sidebar.error("Please ensure the data is correctly placed.")
    subject_list = ["sub-14"] 

selected_subject = st.sidebar.selectbox('Select Subject:', subject_list)
ground_truth_label = subject_labels.get(selected_subject, "N/A")
st.sidebar.markdown(f"**Ground Truth:** `{ground_truth_label}`")

models = ["CapsNetRNN"] 
selected_model = st.sidebar.selectbox('Select Inference Model:', models)

# --- Analysis Trigger Button ---
if st.sidebar.button('Start Analysis', type="primary"):
    with st.spinner('Analysis in progress... This may take a minute. Please wait.'):
        try:
            # --- (B) Handle all file paths on frontend ---
            
            # 1. Handle model paths
            model_paths_map = {
                "CapsNetRNN": "model/capsnet/best_capsnet_rnn.pth",
            }
            model_path = model_paths_map.get(selected_model)
            if not model_path:
                raise FileNotFoundError(f"Cannot find path configuration for model '{selected_model}'.")

            # 2. Handle NIfTI file paths
            nii_search_pattern = f"data/raw/*/{selected_subject}/*.nii.gz"
            nii_file_list = glob.glob(nii_search_pattern)
            if not nii_file_list:
                raise FileNotFoundError(f"Cannot find .nii.gz files for subject '{selected_subject}'.")
            nii_path = nii_file_list[0]
            
            st.info(f"Files found:\n- NIfTI: {nii_path}\n- Model: {model_path}")

            # --- Path handling completed ---

            # 3. [Important Modification] Call backend function with three complete path parameters
            result_json_string = run_analysis_sync(selected_subject, nii_path, model_path)
            
            if result_json_string:
                final_report_obj = FinalReport.model_validate_json(result_json_string)
                st.session_state['final_report'] = final_report_obj
                st.session_state['run_complete'] = True
            else:
                st.error("Analysis finished but the agent returned no content.")
                st.session_state['run_complete'] = False

        except FileNotFoundError as e:
            st.error(f"File error: {e}")
            st.session_state['run_complete'] = False
        except Exception as e:
            st.error("Please wait a minute and try again.")
            st.error(f"Critical error occurred during analysis: {e}")
            st.session_state['run_complete'] = False

# --- Acknowledgement Section ---
st.sidebar.markdown("---") 
adni_acknowledgement = """
<div style="font-size: 0.75rem; color: grey;">
Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: <a href="http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf" target="_blank">ADNI Acknowledgement List</a>.
</div>
"""
st.sidebar.markdown(adni_acknowledgement, unsafe_allow_html=True)

# --- Results display section (this part remains unchanged) ---
if 'run_complete' in st.session_state and st.session_state['run_complete']:
    # ... (Your original result display code)
    report_data = st.session_state['final_report']
    st.markdown("---")
    st.header("Analysis Results")
    st.subheader("Subject Activation overlay on brain.")
    try:
        st.image(report_data.visualization_path, caption=f"Activation map for subject {selected_subject}")
    except Exception as e:
        st.error(f"Cannot display image. Please check path: {report_data.visualization_path}. Error: {e}")
    tab_en, tab_zh = st.tabs(["English Report", "中文報告"])
    with tab_en:
        st.subheader("Clinical Report (English)")
        st.markdown(report_data.final_report_markdown, unsafe_allow_html=True)
    with tab_zh:
        st.subheader("臨床分析報告 (繁體中文)")
        st.markdown(report_data.final_report_chinese, unsafe_allow_html=True)
else:
    st.info("Please select a subject and model, then click 'Start Analysis' in the sidebar to view results.")