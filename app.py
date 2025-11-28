# app/main.py (Professional Dashboard Version)
import os
import streamlit as st
import glob
from pathlib import Path
import streamlit.components.v1 as components
import json

# --- Visualization ---
from nilearn import plotting
from nilearn import image as nimg

# --- LangGraph App ---
from app.graph.workflow import app

# --- CDDA Agent for Executive Summary ---
from app.agents.cdda_agent import CDDAAgent

# --- Structural MRI UI Components ---
from app.ui.structural_mri_components import (
    render_analysis_mode_selector,
    render_structural_results
)


# ---### 變更點 2: 更新快取函式以處理 4D 數據 ###---
@st.cache_resource(show_spinner="正在載入並處理 NIfTI 檔案...")
def load_nifti(path: str):
    """
    載入 NIfTI 檔案並回傳 nilearn 影像物件和時間點總數。
    支援 3D (結構性 MRI) 和 4D (功能性 MRI) 影像。
    """
    try:
        img = nimg.load_img(path)
        # 檢查維度
        if len(img.shape) == 4:
            # 4D 影像（功能性 MRI）
            num_time_points = img.shape[3]
            return img, num_time_points
        elif len(img.shape) == 3:
            # 3D 影像（結構性 MRI）
            # 返回影像本身，時間點為 1
            return img, 1
        else:
            st.error(f"不支援的影像維度: {img.shape}")
            return None, 0
    except Exception as e:
        st.error(f"載入 NIfTI 檔案失敗: {path}. 錯誤: {e}")
        return None, 0


# --- STREAMLIT FRONTEND ---

st.set_page_config(
    page_title="CDDA Clinical Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Header
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🧠 CDDA Clinical Dashboard</h1>
    <p style="color: #e0e7ff; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        Cognitive Discrepancy-Driven Agent for Alzheimer's Disease Diagnosis
    </p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
# Initialize analysis state
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False

st.sidebar.markdown("""
<div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 1rem;">
    <h3 style="margin: 0; color: #1e40af;">⚙️ Analysis Configuration</h3>
</div>
""", unsafe_allow_html=True)

# 分析模式選擇
analysis_mode = render_analysis_mode_selector()
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = analysis_mode
else:
    st.session_state.analysis_mode = analysis_mode


# 受試者選擇 - 分析時禁用但保持在原位
# 根據分析模式使用不同的資料路徑
subject_labels = {}

if st.session_state.analysis_mode == "structural":
    # 結構性 MRI: 使用 data/sMRI（子資料夾結構）
    smri_folders = glob.glob("data/sMRI/*/sub-*")
    for folder_path in smri_folders:
        parts = folder_path.split(os.sep)
        if len(parts) >= 3:
            subject_id = parts[-1]  # sub-0005
            label = parts[-2]  # AD or NC
            # 統一格式為 sub_XXXX
            subject_id_normalized = subject_id.replace("-", "_")
            subject_labels[subject_id_normalized] = label
else:
    # 功能性 MRI: 使用 data/fMRI（子資料夾結構）
    fmri_folders = glob.glob("data/fMRI/*/sub-*")
    for folder_path in fmri_folders:
        parts = folder_path.split(os.sep)
        if len(parts) >= 3:
            subject_id = parts[-1]
            label = parts[-2]
            # 處理 CN -> NC 的標籤轉換
            if label == "CN":
                label = "NC"
            subject_labels[subject_id] = label

subject_list = sorted(subject_labels.keys())
if not subject_list:
    mode_name = "Structural MRI (sMRI)" if st.session_state.analysis_mode == "structural" else "Functional MRI (fMRI)"
    data_path = "data/sMRI/" if st.session_state.analysis_mode == "structural" else "data/fMRI/"
    st.sidebar.error(
        f"⚠️ No {mode_name} subject data found.\n"
        f"Please ensure data exists in {data_path} directory."
    )
    st.stop()

# Maintain current selection (if exists)
current_subject = st.session_state.get("selected_subject")
if current_subject and current_subject in subject_list:
    default_index = subject_list.index(current_subject)
else:
    default_index = 0

is_running = st.session_state.get("analysis_running", False)

st.sidebar.markdown("#### 👤 Subject Selection")
if is_running:
    # Analysis running: show current selection but disabled
    selected_subject = st.sidebar.selectbox(
        "Subject ID",
        [current_subject or "N/A"],
        disabled=True,
        help="Subject selection is locked during analysis.",
        label_visibility="collapsed"
    )
else:
    # Normal state: normal selection
    selected_subject = st.sidebar.selectbox(
        "Subject ID",
        subject_list,
        index=default_index,
        help="Choose a subject for analysis.",
        label_visibility="collapsed"
    )
ground_truth_label = subject_labels.get(selected_subject, "N/A")
st.sidebar.markdown(f"""
<div style="background: #f1f5f9; padding: 0.5rem; border-radius: 4px; margin-top: 0.5rem;">
    <span style="color: #64748b; font-size: 0.875rem;">Ground Truth:</span>
    <span style="color: #1e293b; font-weight: bold; margin-left: 0.5rem;">{ground_truth_label}</span>
</div>
""", unsafe_allow_html=True)


# Model Selection - Different options based on analysis mode
st.sidebar.markdown("#### 🤖 Model Selection")

if st.session_state.analysis_mode == "structural":
    # Structural MRI - Machine Learning Models
    models = {"Random Forest": "random_forest"}
    
    current_model = st.session_state.get("selected_model_display")
    model_list = list(models.keys())
    if current_model and current_model in model_list:
        default_model_index = model_list.index(current_model)
    else:
        default_model_index = 0
    
    if is_running:
        selected_model_display = st.sidebar.selectbox(
            "ML Model",
            [current_model or "N/A"],
            disabled=True,
            help="Model selection is locked during analysis.",
            label_visibility="collapsed"
        )
    else:
        selected_model_display = st.sidebar.selectbox(
            "ML Model",
            model_list,
            index=default_model_index,
            help="Choose the machine learning model for structural MRI classification.",
            label_visibility="collapsed"
        )
    selected_model_key = models[selected_model_display]
    model_path = None
    
    # Model information
    model_info = {
        "random_forest": {
            "type": "Random Forest Classifier",
            "description": "Ensemble learning with ROI-based features from AAL atlas",
            "best_for": "Interpretable structural MRI analysis",
        }
    }
    if selected_model_key in model_info:
        info = model_info[selected_model_key]
        st.sidebar.markdown(f"""
        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 6px; margin-top: 0.5rem; font-size: 0.875rem;">
            <div style="color: #64748b; margin-bottom: 0.25rem;">Model Type</div>
            <div style="color: #1e293b; font-weight: 500;">{info['type']}</div>
            <div style="color: #64748b; margin-top: 0.5rem; margin-bottom: 0.25rem;">Best For</div>
            <div style="color: #1e293b;">{info['best_for']}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    # Functional MRI - Deep Learning Models
    models = {"ShuffleNet": "shufflenet", "CapsNet": "capsnet", "MCADNNet": "mcadnnet"}

    current_model = st.session_state.get("selected_model_display")
    model_list = list(models.keys())
    if current_model and current_model in model_list:
        default_model_index = model_list.index(current_model)
    else:
        default_model_index = 0

    if is_running:
        selected_model_display = st.sidebar.selectbox(
            "Neural Network Model",
            [current_model or "N/A"],
            disabled=True,
            help="Model selection is locked during analysis.",
            label_visibility="collapsed"
        )
    else:
        selected_model_display = st.sidebar.selectbox(
            "Neural Network Model",
            model_list,
            index=default_model_index,
            help="Choose the neural network model for fMRI classification.",
            label_visibility="collapsed"
        )
    selected_model_key = models[selected_model_display]

    # Model information
    model_info = {
        "shufflenet": {
            "type": "2D ShuffleNet + ECA Attention",
            "description": "High-accuracy 2D CNN with attention mechanism",
            "best_for": "High-accuracy AD/NC classification (80%+)",
        },
        "capsnet": {
            "type": "3D Capsule Network",
            "description": "Advanced capsule layers for spatial relationships",
            "best_for": "Complex 3D fMRI patterns",
        },
        "mcadnnet": {
            "type": "2D Convolutional Neural Network",
            "description": "Traditional CNN for 2D slice analysis",
            "best_for": "Computational efficiency",
        },
    }
    if selected_model_key in model_info:
        info = model_info[selected_model_key]
        st.sidebar.markdown(f"""
        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 6px; margin-top: 0.5rem; font-size: 0.875rem;">
            <div style="color: #64748b; margin-bottom: 0.25rem;">Model Type</div>
            <div style="color: #1e293b; font-weight: 500;">{info['type']}</div>
            <div style="color: #64748b; margin-top: 0.5rem; margin-bottom: 0.25rem;">Best For</div>
            <div style="color: #1e293b;">{info['best_for']}</div>
        </div>
        """, unsafe_allow_html=True)

# 檢查是否有參數變更，如果有則重置分析狀態
prev_subject = st.session_state.get('selected_subject')
prev_model = st.session_state.get('selected_model_key')

if (prev_subject and prev_subject != selected_subject) or (prev_model and prev_model != selected_model_key):
    # 參數有變更，重置完成狀態以允許重新分析
    st.session_state.run_complete = False
    # 清除舊的結果
    if 'final_state' in st.session_state:
        del st.session_state['final_state']
    if 'nii_path' in st.session_state:
        del st.session_state['nii_path']

# 儲存當前選擇到 session state
st.session_state.selected_subject = selected_subject
st.session_state.selected_model_display = selected_model_display
st.session_state.selected_model_key = selected_model_key
st.session_state.ground_truth_label = ground_truth_label

# Action Buttons
st.sidebar.markdown("<br>", unsafe_allow_html=True)
if is_running:
    # Analysis running: disabled main button + Force Stop
    st.sidebar.button(
        "🔄 Analysis Running...",
        type="primary",
        use_container_width=True,
        disabled=True,
        help="Analysis in progress...",
    )
    # Force Stop button
    if st.sidebar.button(
        "⏹️ Force Stop Analysis",
        type="secondary",
        use_container_width=True,
    ):
        st.session_state.analysis_running = False
        st.session_state.run_complete = False
        st.sidebar.warning("⚠️ Analysis has been stopped.")
        st.rerun()
    start_button = False
else:
    # Normal state: normal start button
    start_button = st.sidebar.button(
        "▶️ Start Analysis",
        type="primary",
        use_container_width=True,
        help=f"Start analysis for {selected_subject} using {selected_model_display}",
    )

st.sidebar.markdown("---")

# ADNI Acknowledgement
st.sidebar.markdown("""
<div style="background: #f8fafc; padding: 1rem; border-radius: 6px; margin-top: 1rem;">
    <div style="font-size: 0.75rem; color: #64748b; line-height: 1.5;">
        <strong style="color: #475569;">Data Source:</strong><br>
        Alzheimer's Disease Neuroimaging Initiative (ADNI)<br>
        <a href="http://adni.loni.usc.edu" target="_blank" style="color: #3b82f6;">adni.loni.usc.edu</a>
    </div>
</div>
""", unsafe_allow_html=True)

# System Information
st.sidebar.markdown("""
<div style="background: #f8fafc; padding: 1rem; border-radius: 6px; margin-top: 0.5rem;">
    <div style="font-size: 0.75rem; color: #64748b;">
        <strong style="color: #475569;">CDDA Framework v1.0</strong><br>
        Dual-LLM A2A Architecture<br>
        Phi-4-mini + Llama3.1-Aloe-Beta-8B
    </div>
</div>
""", unsafe_allow_html=True)


# --- 分析邏輯 ---
# --- 分析邏輯 ---
if start_button:
    # 重置所有分析狀態，尤其是 run_complete
    st.session_state.analysis_running = True
    st.session_state.run_complete = False  # 重置完成狀態
    st.session_state.viewer_expanded = True
    
    # 清除之前的結果狀態（防止干擾）
    if 'final_state' in st.session_state:
        del st.session_state['final_state']
    if 'nii_path' in st.session_state:
        del st.session_state['nii_path']
    
    # 強制重新載入頁面以更新側邊欄狀態
    st.rerun()

# 檢查是否有正在進行的分析
if st.session_state.get("analysis_running", False) and not st.session_state.get(
    "run_complete", False
):
    # 進度條和狀態更新
    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("Analyzing brain patterns... This may take a few minutes."):
        try:
            # 從 session state 取得設定值
            selected_subject = st.session_state.selected_subject
            selected_model_key = st.session_state.selected_model_key
            ground_truth_label = st.session_state.ground_truth_label

            # 進度階段更新
            import time

            status_text.text("Preparing analysis...")
            progress_bar.progress(10)

            # 根據分析模式設定模型路徑
            if st.session_state.analysis_mode == "structural":
                model_path = None  # 使用 config 中的預設路徑
            else:
                model_paths_map = {
                    "shufflenet": "model/shufflenet/fold_3_best_model.pth",
                    "capsnet": "model/capsnet/best_capsnet_rnn.pth",
                    "mcadnnet": "model/macadnnet/._best_overall_model.pth",
                }
                model_path = model_paths_map.get(selected_model_key)
                if not model_path:
                    raise FileNotFoundError(
                        f"找不到模型 '{selected_model_key}' 的路徑設定。"
                    )

            status_text.text("Loading data files...")
            progress_bar.progress(20)

            # 根據分析模式搜尋不同的檔案
            if st.session_state.analysis_mode == "structural":
                # 結構性 MRI: 從 data/sMRI 搜尋 T1 檔案
                label = ground_truth_label
                # 將 sub_XXXX 轉換為 sub-XXXX（資料夾格式）
                subject_folder = selected_subject.replace("_", "-")
                nii_search_pattern = f"data/sMRI/{label}/{subject_folder}/*_T1.nii.gz"
                nii_file_list = glob.glob(nii_search_pattern)
                
                if not nii_file_list:
                    raise FileNotFoundError(
                        f"找不到受試者 '{selected_subject}' 的 T1 檔案。\n"
                        f"搜尋路徑: {nii_search_pattern}\n"
                        f"請確認檔案存在於 data/sMRI/{label}/{subject_folder}/ 目錄下"
                    )
                
                nii_path = nii_file_list[0]
                st.info(f"Files found:\n- T1 MRI: {nii_path}")
            else:
                # 功能性 MRI: 從 data/fMRI 搜尋檔案
                nii_search_pattern = f"data/fMRI/*/{selected_subject}/*.nii.gz"
                nii_file_list = glob.glob(nii_search_pattern)
                if not nii_file_list:
                    raise FileNotFoundError(
                        f"找不到受試者 '{selected_subject}' 的 .nii.gz 檔案。\n"
                        f"搜尋路徑: {nii_search_pattern}"
                    )
                nii_path = nii_file_list[0]
                st.info(f"Files found:\n- NIfTI: {nii_path}\n- Model: {model_path}")

            status_text.text("Starting brain analysis workflow...")
            progress_bar.progress(30)

            initial_state = {
                "subject_id": selected_subject,
                "fmri_scan_path": nii_path,
                "model_path": model_path,
                "model_name": selected_model_key,
                "analysis_mode": st.session_state.analysis_mode,  # 新增分析模式
                "trace_log": [],
                "error_log": []
            }

            status_text.text("Running AI analysis pipeline...")
            progress_bar.progress(50)

            final_state = app.invoke(initial_state)

            status_text.text("Finalizing results...")
            progress_bar.progress(90)

            if final_state:
                status_text.text("Analysis completed successfully!")
                progress_bar.progress(100)

                st.session_state["nii_path"] = nii_path
                st.session_state["final_state"] = final_state
                st.session_state["ground_truth_label"] = ground_truth_label
                st.session_state["run_complete"] = True
                # 分析完成，恢復正常狀態
                st.session_state.analysis_running = False

                time.sleep(1)  # 稍微等待讓用戶看到完成狀態
                st.success("Analysis completed successfully!")
                st.rerun()
            else:
                status_text.text("Analysis completed with issues")
                progress_bar.progress(100)

                st.error("Analysis finished but the agent returned no content.")
                st.session_state["run_complete"] = False
                st.session_state.analysis_running = False

        except Exception as e:
            # 錯誤時的進度更新
            status_text.text("Analysis failed")
            progress_bar.progress(0)

            st.error("Please try again later.")
            st.error(f"Critical error occurred during analysis: {e}")
            st.session_state["run_complete"] = False
            # 發生錯誤時也要恢復正常狀態
            st.session_state.analysis_running = False

# --- RESULTS DISPLAY ---
if st.session_state.get("run_complete", False):
    final_state = st.session_state["final_state"]
    report_ground_truth = st.session_state.get("ground_truth_label", "N/A")
    analyzed_subject = final_state.get(
        "subject_id", st.session_state.get("selected_subject", "Unknown")
    )
    
    analysis_mode = final_state.get("analysis_mode", "functional")
    
    # Generate Executive Summary
    if "executive_summary" not in st.session_state:
        predicted_label = final_state.get("classification_result", "N/A")
        
        # Simple rule-based summary (no LLM required)
        # This avoids complex dependencies and provides immediate results
        
        # Determine risk level based on prediction match
        is_correct = report_ground_truth == predicted_label
        if is_correct:
            risk_level = "Low"
            headline = f"Confirmed {predicted_label} diagnosis with model agreement"
        else:
            risk_level = "High"
            headline = f"Predicted {predicted_label} (Ground Truth: {report_ground_truth}) - Discrepancy detected"
        
        # Generate key findings
        key_findings = [
            f"AI Model Prediction: {predicted_label}",
            f"Clinical Ground Truth: {report_ground_truth}",
            f"Prediction Accuracy: {'Correct ✓' if is_correct else 'Incorrect ✗'}"
        ]
        
        # Add analysis mode specific findings
        if analysis_mode == "structural":
            key_findings.append("Analysis Type: Structural MRI (sMRI) with Random Forest")
        else:
            model_name = st.session_state.get("selected_model_display", "Unknown")
            key_findings.append(f"Analysis Type: Functional MRI (fMRI) with {model_name}")
        
        # Generate recommended actions
        if is_correct:
            recommended_actions = [
                "Model prediction aligns with clinical diagnosis",
                "Review detailed report for feature analysis",
                "Standard clinical follow-up appropriate"
            ]
        else:
            recommended_actions = [
                "⚠️ Prediction discrepancy requires clinical review",
                "Examine detailed report for potential explanations",
                "Consider additional diagnostic tests or imaging",
                "Consult with clinical team for final diagnosis"
            ]
        
        st.session_state.executive_summary = {
            "headline": headline,
            "key_findings": key_findings,
            "recommended_actions": recommended_actions,
            "risk_level": risk_level
        }
    
    executive_summary = st.session_state.get("executive_summary", {})
    
    # Professional Dashboard Header
    st.markdown("---")
    st.markdown("""
    <div style="background: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3b82f6; margin-bottom: 2rem;">
        <h2 style="margin: 0; color: #1e40af;">📊 Clinical Executive Summary</h2>
        <p style="margin: 0.5rem 0 0 0; color: #64748b;">AI-Generated Diagnostic Overview</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk Level Badge
    risk_level = executive_summary.get("risk_level", "Medium")
    risk_colors = {
        "High": ("#dc2626", "#fef2f2", "⚠️"),
        "Medium": ("#f59e0b", "#fffbeb", "⚡"),
        "Low": ("#10b981", "#f0fdf4", "✅")
    }
    risk_color, risk_bg, risk_icon = risk_colors.get(risk_level, ("#6b7280", "#f9fafb", "ℹ️"))
    
    # Headline with Risk Badge
    headline = executive_summary.get("headline", "Analysis completed")
    st.markdown(f"""
    <div style="background: {risk_bg}; padding: 1.5rem; border-radius: 8px; border: 2px solid {risk_color}; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="font-size: 2rem;">{risk_icon}</span>
            <div style="flex: 1;">
                <div style="background: {risk_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 4px; display: inline-block; font-size: 0.75rem; font-weight: bold; margin-bottom: 0.5rem;">
                    RISK LEVEL: {risk_level.upper()}
                </div>
                <h3 style="margin: 0; color: {risk_color}; font-size: 1.5rem;">{headline}</h3>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    predicted_label = final_state.get("classification_result", "N/A")
    is_correct = report_ground_truth == predicted_label
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.5rem;">SUBJECT ID</div>
            <div style="color: #1e293b; font-size: 1.5rem; font-weight: bold;">{analyzed_subject}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.5rem;">GROUND TRUTH</div>
            <div style="color: #1e293b; font-size: 1.5rem; font-weight: bold;">{report_ground_truth}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.5rem;">AI PREDICTION</div>
            <div style="color: #1e293b; font-size: 1.5rem; font-weight: bold;">{predicted_label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        accuracy_color = "#10b981" if is_correct else "#dc2626"
        accuracy_icon = "✓" if is_correct else "✗"
        accuracy_text = "CORRECT" if is_correct else "INCORRECT"
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.5rem;">ACCURACY</div>
            <div style="color: {accuracy_color}; font-size: 1.5rem; font-weight: bold;">{accuracy_icon} {accuracy_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Findings and Recommended Actions
    col_findings, col_actions = st.columns(2)
    
    with col_findings:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); height: 100%;">
            <h4 style="margin: 0 0 1rem 0; color: #1e40af;">🔍 Key Findings</h4>
        """, unsafe_allow_html=True)
        
        key_findings = executive_summary.get("key_findings", ["Analysis completed"])
        for finding in key_findings:
            st.markdown(f"• {finding}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_actions:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); height: 100%;">
            <h4 style="margin: 0 0 1rem 0; color: #1e40af;">💡 Recommended Actions</h4>
        """, unsafe_allow_html=True)
        
        recommended_actions = executive_summary.get("recommended_actions", ["Review detailed report"])
        for action in recommended_actions:
            st.markdown(f"• {action}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualization Section
    if analysis_mode == "functional":
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #8b5cf6; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: #6b21a8;">🎨 Brain Activation Visualization</h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            viz_path = final_state.get("visualization_paths", [])[0]
            st.image(viz_path, caption=f"Activation map for subject {analyzed_subject}", use_container_width=True)
        except Exception as e:
            st.error(f"Cannot display visualization: {e}")
    
    # Detailed Report (Collapsible)
    with st.expander("📄 View Detailed Clinical Report", expanded=False):
        if analysis_mode == "structural":
            render_structural_results(final_state, report_ground_truth)
        else:
            reports = final_state.get("generated_reports", {})
            report_en = reports.get("en", "No English report was generated.")
            report_zh = reports.get("zh", "沒有生成中文報告。")
            
            tab_en, tab_zh = st.tabs(["English Report", "中文報告"])
            with tab_en:
                st.markdown("### Clinical Report (English)")
                st.markdown(report_en, unsafe_allow_html=True)
            with tab_zh:
                st.markdown("### 臨床分析報告 (繁體中文)")
                st.markdown(report_zh, unsafe_allow_html=True)

    # Interactive MRI Viewer
    is_expanded_default = st.session_state.get("viewer_expanded", False)
    with st.expander("🔬 Interactive MRI Viewer", expanded=is_expanded_default):
        st.markdown("""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
            <p style="color: #64748b; margin: 0; font-size: 0.875rem;">
                Explore the original MRI scan with interactive 3D visualization. 
                Use the controls to navigate through different brain slices.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        nii_path = st.session_state.get("nii_path")
        if nii_path and Path(nii_path).exists():
            # Load NIfTI file (supports 3D and 4D)
            img, num_time_points = load_nifti(nii_path)

            if img and num_time_points > 0:
                # Determine whether to show time axis slider based on image dimensions
                if num_time_points > 1:
                    # 4D image: show time axis slider
                    selected_time_point_display = st.slider(
                        "Time Point (Volume)",
                        min_value=1,
                        max_value=num_time_points,
                        value=1,
                        help=f"This scan has {num_time_points} volumes.",
                    )
                    # Convert to 0-based index
                    selected_time_point_index = selected_time_point_display - 1
                    # Extract 3D image at specified time point
                    img_3d_at_t = nimg.index_img(img, selected_time_point_index)
                else:
                    # 3D image: use directly
                    img_3d_at_t = img
                    selected_time_point_display = None

                # Set title based on image type
                if selected_time_point_display:
                    title = f"Volume at T={selected_time_point_display}"
                else:
                    title = "Structural MRI (T1-weighted)"

                viewer = plotting.view_img(
                    img_3d_at_t,
                    bg_img=None,
                    cmap="gray",
                    threshold=None,
                    title=title,
                    resampling_interpolation="nearest",
                    colorbar=False,
                    annotate=True,
                    black_bg=True,
                )

                components.html(viewer.html, height=600, scrolling=False)
        else:
            st.warning("⚠️ Could not find the original NIfTI file for this viewer.")
else:
    # Welcome Screen
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
        <h2 style="color: #1e40af; margin-bottom: 1rem;">Welcome to CDDA Clinical Dashboard</h2>
        <p style="color: #64748b; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;">
            Select a subject and model from the sidebar, then click <strong>"Start Analysis"</strong> 
            to generate AI-powered diagnostic insights with complete reasoning transparency.
        </p>
        <div style="background: #f8fafc; padding: 2rem; border-radius: 10px; max-width: 800px; margin: 0 auto;">
            <h3 style="color: #1e40af; margin-bottom: 1rem;">Key Features</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; text-align: left;">
                <div>
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎯</div>
                    <strong style="color: #1e293b;">Adaptive Decision-Making</strong>
                    <p style="color: #64748b; font-size: 0.875rem; margin: 0.25rem 0 0 0;">
                        Dynamic pathway selection based on uncertainty
                    </p>
                </div>
                <div>
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔍</div>
                    <strong style="color: #1e293b;">Counterfactual Analysis</strong>
                    <p style="color: #64748b; font-size: 0.875rem; margin: 0.25rem 0 0 0;">
                        Causal reasoning for diagnostic drivers
                    </p>
                </div>
                <div>
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
                    <strong style="color: #1e293b;">Executive Summary</strong>
                    <p style="color: #64748b; font-size: 0.875rem; margin: 0.25rem 0 0 0;">
                        AI-generated structured overview
                    </p>
                </div>
                <div>
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔗</div>
                    <strong style="color: #1e293b;">Knowledge Integration</strong>
                    <p style="color: #64748b; font-size: 0.875rem; margin: 0.25rem 0 0 0;">
                        Clinical context from knowledge graph
                    </p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
