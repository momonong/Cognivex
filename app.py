import streamlit as st
import json
import glob 

from agents.sub_agents.final_report.agent import FinalReport
from backend.backend_runner import run_analysis_sync

# --- STREAMLIT 前端介面 ---

st.set_page_config(page_title="fMRI Analysis Framework", layout="wide")
st.title("Explainable fMRI Analysis for Alzheimer's Disease")
st.markdown("An agent-based framework for generating knowledge-grounded clinical interpretations from fMRI data.")

# --- 側邊欄控制項 ---
st.sidebar.header("Analysis Controls")
subjects = ["sub-14", "sub-21", "sub-35"]
models = ["CapsNetRNN", "MCADNNet"]

selected_subject = st.sidebar.selectbox('Select Subject:', subjects)
selected_model = st.sidebar.selectbox('Select Inference Model:', models)

# --- 觸發分析的按鈕 ---
if st.sidebar.button('Start Analysis', type="primary"):
    with st.spinner('Analysis in progress... This may take a minute. Please wait.'):
        try:
            # --- (B) 在前端處理所有檔案路徑 ---
            
            # 1. 處理模型路徑
            model_paths_map = {
                "CapsNetRNN": "model/capsnet/best_capsnet_rnn.pth",
                "MCADNNet": "model/mcadnnet/best_mcadnnet.pth"
            }
            model_path = model_paths_map.get(selected_model)
            if not model_path:
                raise FileNotFoundError(f"找不到模型 '{selected_model}' 的路徑設定。")

            # 2. 處理 NIfTI 檔案路徑
            nii_search_pattern = f"data/raw/*/{selected_subject}/*.nii.gz"
            nii_file_list = glob.glob(nii_search_pattern)
            if not nii_file_list:
                raise FileNotFoundError(f"找不到受試者 '{selected_subject}' 的 .nii.gz 檔案。")
            nii_path = nii_file_list[0]
            
            st.info(f"找到檔案：\n- NIfTI: {nii_path}\n- Model: {model_path}")

            # --- 路徑處理結束 ---

            # 3. 【重要修改】用三個完整的路徑參數呼叫後端函式
            result_json_string = run_analysis_sync(selected_subject, nii_path, model_path)
            
            if result_json_string:
                final_report_obj = FinalReport.model_validate_json(result_json_string)
                st.session_state['final_report'] = final_report_obj
                st.session_state['run_complete'] = True
            else:
                st.error("Analysis finished but the agent returned no content.")
                st.session_state['run_complete'] = False

        except FileNotFoundError as e:
            st.error(f"檔案錯誤：{e}")
            st.session_state['run_complete'] = False
        except Exception as e:
            st.error(f"分析流程中發生嚴重錯誤：{e}")
            st.session_state['run_complete'] = False

# --- 顯示結果的區塊 (這部分完全不變) ---
if 'run_complete' in st.session_state and st.session_state['run_complete']:
    # ... (您原有的結果顯示程式碼)
    report_data = st.session_state['final_report']
    st.markdown("---")
    st.header("Analysis Results")
    st.subheader("Group Activation with DMN Mask")
    try:
        st.image(report_data.visualization_path, caption=f"Activation map for subject {selected_subject}")
    except Exception as e:
        st.error(f"無法顯示圖片。請檢查路徑：{report_data.visualization_path}。錯誤：{e}")
    tab_en, tab_zh = st.tabs(["English Report", "中文報告"])
    with tab_en:
        st.subheader("Clinical Report (English)")
        st.markdown(report_data.final_report_markdown, unsafe_allow_html=True)
    with tab_zh:
        st.subheader("臨床分析報告 (繁體中文)")
        st.markdown(report_data.final_report_chinese, unsafe_allow_html=True)
else:
    st.info("Please select a subject and model, then click 'Start Analysis' in the sidebar to view results.")