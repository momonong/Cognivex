#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognivex CDDA - Streamlit Web Interface
整合 CDDA Framework 的完整診斷系統
"""

import os
import sys
import streamlit as st
import glob
from pathlib import Path
import streamlit.components.v1 as components
from datetime import datetime

# 視覺化相關
from nilearn import plotting
from nilearn import image as nimg

# CDDA Framework
from app.agents.cdda_agent import CDDAAgent

# 傳統 LangGraph 工作流（保留作為備選）
from app.graph.workflow import app as langgraph_app

# 結構性 MRI UI 組件
from app.ui.structural_mri_components import (
    render_analysis_mode_selector,
    render_structural_results
)


# ============================================================================
# 工具函數
# ============================================================================

@st.cache_resource(show_spinner="正在載入並處理 NIfTI 檔案...")
def load_nifti(path: str):
    """
    載入 NIfTI 檔案並回傳 nilearn 影像物件和時間點總數。
    支援 3D (結構性 MRI) 和 4D (功能性 MRI) 影像。
    """
    try:
        img = nimg.load_img(path)
        if len(img.shape) == 4:
            # 4D 影像（功能性 MRI）
            num_time_points = img.shape[3]
            return img, num_time_points
        elif len(img.shape) == 3:
            # 3D 影像（結構性 MRI）
            return img, 1
        else:
            st.error(f"不支援的影像維度: {img.shape}")
            return None, 0
    except Exception as e:
        st.error(f"載入 NIfTI 檔案失敗: {path}. 錯誤: {e}")
        return None, 0


@st.cache_resource(show_spinner="正在初始化 CDDA Agent...")
def initialize_cdda_agent(use_llm=False):
    """初始化 CDDA Agent（快取以避免重複載入）"""
    try:
        agent = CDDAAgent(
            use_llm=use_llm,
            verbose=True
        )
        return agent
    except Exception as e:
        st.error(f"CDDA Agent 初始化失敗: {e}")
        return None


def format_cdda_report(result) -> str:
    """格式化 CDDA 分析結果為 HTML 報告"""
    html = f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h3 style="color: #1f77b4;">🧠 CDDA 診斷報告</h3>
        
        <div style="margin: 15px 0;">
            <h4>📊 診斷摘要</h4>
            <ul>
                <li><strong>受試者:</strong> {result.subject_id}</li>
                <li><strong>預測:</strong> <span style="color: {'#d32f2f' if result.prediction == 'AD' else '#388e3c'}; font-weight: bold;">{result.prediction}</span></li>
                <li><strong>信心度:</strong> {result.confidence:.1%}</li>
                <li><strong>不確定性評分 (UQ):</strong> {result.uq_score:.3f}</li>
                <li><strong>代理決策:</strong> {result.agent_decision}</li>
            </ul>
        </div>
        
        <div style="margin: 15px 0;">
            <h4>🔍 關鍵發現</h4>
            <p>{result.report}</p>
        </div>
    """
    
    # 添加反事實分析（如果有）
    if result.metadata.get('counterfactual_result'):
        cf = result.metadata['counterfactual_result']
        html += f"""
        <div style="margin: 15px 0; background-color: #fff3cd; padding: 15px; border-radius: 5px;">
            <h4>🔄 反事實分析</h4>
            <p><strong>原始預測:</strong> {cf.get('original_prediction', 'N/A')} ({cf.get('original_confidence', 0):.1%})</p>
            <p><strong>模擬後預測:</strong> {cf.get('counterfactual_prediction', 'N/A')} ({cf.get('counterfactual_confidence', 0):.1%})</p>
            <p><strong>信心度變化:</strong> {cf.get('confidence_delta', 0):.1%}</p>
            <p><strong>解釋:</strong> {cf.get('interpretation', 'N/A')}</p>
        </div>
        """
    
    # 添加異常分析（如果有）
    if result.metadata.get('anomalous_regions'):
        anomalies = result.metadata['anomalous_regions']
        html += f"""
        <div style="margin: 15px 0; background-color: #f8d7da; padding: 15px; border-radius: 5px;">
            <h4>⚠️ 異常區域檢測</h4>
            <p>檢測到 {len(anomalies)} 個異常腦區:</p>
            <ul>
        """
        for region in anomalies[:5]:  # 只顯示前 5 個
            html += f"<li>{region}</li>"
        html += "</ul></div>"
    
    html += "</div>"
    return html


def display_reasoning_chain(reasoning_chain):
    """顯示推理鏈"""
    with st.expander("🔗 查看完整推理鏈", expanded=False):
        st.markdown("### 代理推理過程")
        for i, step in enumerate(reasoning_chain, 1):
            if step.startswith("="*80):
                st.markdown(f"**{step.replace('=', '')}**")
            elif step.startswith("-"*80):
                st.markdown(f"*{step.replace('-', '')}*")
            else:
                st.text(step)


# ============================================================================
# Streamlit 應用主體
# ============================================================================

st.set_page_config(
    page_title="Cognivex CDDA - Explainable fMRI Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Cognivex CDDA Framework")
st.markdown("""
**Cognitive Discrepancy-Driven Agent** - 自主診斷代理系統

整合雙 LLM 架構、MCP 協議和 A2A 模式的可解釋 AI 診斷系統
""")

# ============================================================================
# 側邊欄控制
# ============================================================================

# 初始化分析狀態
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False
if "cdda_mode" not in st.session_state:
    st.session_state.cdda_mode = True  # 預設使用 CDDA

st.sidebar.header("⚙️ 分析設定")

# 分析模式選擇
analysis_framework = st.sidebar.radio(
    "選擇分析框架:",
    ["CDDA Framework (推薦)", "傳統 LangGraph"],
    help="CDDA Framework 提供自主決策、反事實分析和混合病理檢測"
)
st.session_state.cdda_mode = (analysis_framework == "CDDA Framework (推薦)")

# 受試者選擇
st.sidebar.markdown("---")
st.sidebar.subheader("📁 受試者選擇")

# 掃描可用的受試者
subject_labels = {}
# 修正：使用 sub-* 而不是 sub_*（匹配實際的目錄命名格式）
data_folders = glob.glob("data/MRI_processed/*/sub-*")
for folder_path in data_folders:
    parts = folder_path.split(os.sep)
    if len(parts) >= 3:
        subject_id = parts[-1]  # sub-0001
        label = parts[-2]  # AD, MCI, or NC
        subject_labels[subject_id] = label

subject_list = sorted(subject_labels.keys())
if not subject_list:
    st.sidebar.error("找不到任何受試者資料。請確認資料在 data/MRI_processed/ 目錄下。")
    st.stop()

# 保持當前選擇
current_subject = st.session_state.get("selected_subject")
if current_subject and current_subject in subject_list:
    default_index = subject_list.index(current_subject)
else:
    default_index = 0

is_running = st.session_state.get("analysis_running", False)
if is_running:
    selected_subject = st.sidebar.selectbox(
        "選擇受試者:",
        [current_subject or "N/A"],
        disabled=True,
        help="分析進行中，受試者選擇已鎖定",
    )
else:
    selected_subject = st.sidebar.selectbox(
        "選擇受試者:",
        subject_list,
        index=default_index,
        help="選擇要分析的受試者",
    )

ground_truth_label = subject_labels.get(selected_subject, "N/A")
st.sidebar.markdown(f"**真實標籤:** `{ground_truth_label}`")

# CDDA 設定
if st.session_state.cdda_mode:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 CDDA 設定")
    
    use_llm = st.sidebar.checkbox(
        "啟用 LLM 模式",
        value=False,
        help="啟用雙 LLM 架構（Agent A + Agent B）。關閉則使用規則式降級"
    )
    
    show_reasoning = st.sidebar.checkbox(
        "顯示推理鏈",
        value=True,
        help="顯示完整的代理推理過程"
    )
    
    st.session_state.use_llm = use_llm
    st.session_state.show_reasoning = show_reasoning

# 檢查參數變更
prev_subject = st.session_state.get('selected_subject')
if prev_subject and prev_subject != selected_subject:
    st.session_state.run_complete = False
    if 'cdda_result' in st.session_state:
        del st.session_state['cdda_result']

st.session_state.selected_subject = selected_subject
st.session_state.ground_truth_label = ground_truth_label

# 按鈕區域
st.sidebar.markdown("---")
if is_running:
    st.sidebar.button(
        "分析進行中...",
        type="primary",
        use_container_width=True,
        disabled=True,
    )
    if st.sidebar.button("強制停止", type="secondary", use_container_width=True):
        st.session_state.analysis_running = False
        st.session_state.run_complete = False
        st.sidebar.warning("分析已停止")
        st.rerun()
    start_button = False
else:
    start_button = st.sidebar.button(
        "🚀 開始分析",
        type="primary",
        use_container_width=True,
        help=f"開始分析 {selected_subject}"
    )

# ADNI 致謝
st.sidebar.markdown("---")
adni_acknowledgement = """
<div style="font-size: 0.75rem; color: grey;">
Data used in preparation of this article were obtained from the Alzheimer's Disease 
Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu).
</div>
"""
st.sidebar.markdown(adni_acknowledgement, unsafe_allow_html=True)

# ============================================================================
# 分析邏輯
# ============================================================================

if start_button:
    st.session_state.analysis_running = True
    st.session_state.run_complete = False
    
    if 'cdda_result' in st.session_state:
        del st.session_state['cdda_result']
    if 'langgraph_result' in st.session_state:
        del st.session_state['langgraph_result']
    
    st.rerun()

# 執行分析
if st.session_state.get("analysis_running", False) and not st.session_state.get("run_complete", False):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("正在分析腦部影像... 這可能需要幾分鐘。"):
        try:
            selected_subject = st.session_state.selected_subject
            ground_truth_label = st.session_state.ground_truth_label
            
            import time
            
            if st.session_state.cdda_mode:
                # ============================================================
                # CDDA Framework 分析
                # ============================================================
                status_text.text("初始化 CDDA Agent...")
                progress_bar.progress(10)
                
                use_llm = st.session_state.get('use_llm', False)
                agent = initialize_cdda_agent(use_llm=use_llm)
                
                if agent is None:
                    raise Exception("CDDA Agent 初始化失敗")
                
                status_text.text("執行 CDDA 分析...")
                progress_bar.progress(30)
                
                # 執行分析
                result = agent.run_analysis(selected_subject)
                
                status_text.text("生成診斷報告...")
                progress_bar.progress(70)
                
                # 儲存結果
                st.session_state['cdda_result'] = result
                st.session_state['analysis_framework'] = 'CDDA'
                
                status_text.text("分析完成！")
                progress_bar.progress(100)
                
            else:
                # ============================================================
                # 傳統 LangGraph 分析
                # ============================================================
                status_text.text("準備分析...")
                progress_bar.progress(10)
                
                # 尋找 NIfTI 檔案
                nii_search_pattern = f"data/fMRI/*/{selected_subject}/*.nii.gz"
                nii_file_list = glob.glob(nii_search_pattern)
                if not nii_file_list:
                    raise FileNotFoundError(f"找不到受試者 '{selected_subject}' 的 .nii.gz 檔案")
                
                nii_path = nii_file_list[0]
                model_path = "model/capsnet/best_capsnet_rnn.pth"
                
                status_text.text("載入資料檔案...")
                progress_bar.progress(20)
                
                initial_state = {
                    "subject_id": selected_subject,
                    "fmri_scan_path": nii_path,
                    "model_path": model_path,
                    "model_name": "capsnet",
                    "analysis_mode": "functional",
                    "trace_log": [],
                    "error_log": []
                }
                
                status_text.text("執行 AI 分析管線...")
                progress_bar.progress(50)
                
                final_state = langgraph_app.invoke(initial_state)
                
                status_text.text("完成結果...")
                progress_bar.progress(90)
                
                st.session_state['langgraph_result'] = final_state
                st.session_state['nii_path'] = nii_path
                st.session_state['analysis_framework'] = 'LangGraph'
                
                status_text.text("分析完成！")
                progress_bar.progress(100)
            
            st.session_state['run_complete'] = True
            st.session_state.analysis_running = False
            
            time.sleep(1)
            st.success("✅ 分析成功完成！")
            st.rerun()
            
        except Exception as e:
            status_text.text("分析失敗")
            progress_bar.progress(0)
            
            st.error(f"分析過程中發生錯誤: {e}")
            st.session_state['run_complete'] = False
            st.session_state.analysis_running = False

# ============================================================================
# 結果顯示
# ============================================================================

if st.session_state.get("run_complete", False):
    st.markdown("---")
    st.header("📊 分析結果")
    
    analysis_framework = st.session_state.get('analysis_framework', 'Unknown')
    
    if analysis_framework == 'CDDA':
        # ================================================================
        # CDDA 結果顯示
        # ================================================================
        result = st.session_state['cdda_result']
        ground_truth = st.session_state.get("ground_truth_label", "N/A")
        
        # 顯示診斷報告
        report_html = format_cdda_report(result)
        st.markdown(report_html, unsafe_allow_html=True)
        
        # 預測驗證
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("真實標籤", ground_truth)
        col2.metric("模型預測", result.prediction)
        col3.metric("信心度", f"{result.confidence:.1%}")
        col4.metric("UQ 評分", f"{result.uq_score:.3f}")
        
        if ground_truth == result.prediction:
            st.success("✅ 預測正確")
        else:
            st.error("❌ 預測錯誤")
        
        # 顯示推理鏈
        if st.session_state.get('show_reasoning', True) and result.reasoning_chain:
            display_reasoning_chain(result.reasoning_chain)
        
        # 顯示元數據
        with st.expander("📋 詳細元數據", expanded=False):
            st.json(result.metadata)
    
    else:
        # ================================================================
        # LangGraph 結果顯示（保留原有邏輯）
        # ================================================================
        final_state = st.session_state['langgraph_result']
        ground_truth = st.session_state.get("ground_truth_label", "N/A")
        
        # 顯示激活圖
        st.subheader("腦部激活圖")
        try:
            viz_path = final_state.get("visualization_paths", [])[0]
            st.image(viz_path, caption=f"激活圖 - {selected_subject}")
        except Exception as e:
            st.error(f"無法顯示圖像: {e}")
        
        # 預測驗證
        predicted_label = final_state.get("classification_result", "N/A")
        st.subheader("預測驗證")
        col1, col2 = st.columns(2)
        col1.metric("真實標籤", ground_truth)
        col2.metric("模型預測", predicted_label)
        
        if ground_truth == predicted_label:
            st.success("✅ 預測正確")
        else:
            st.error("❌ 預測錯誤")
        
        # 顯示報告
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
    
    # ====================================================================
    # 互動式 fMRI 檢視器（兩種模式共用）
    # ====================================================================
    with st.expander("🔍 探索原始 fMRI 掃描（互動式切片器）", expanded=False):
        nii_path = st.session_state.get("nii_path")
        if nii_path and Path(nii_path).exists():
            img, num_time_points = load_nifti(nii_path)
            
            if img and num_time_points > 0:
                if num_time_points > 1:
                    selected_time_point_display = st.slider(
                        "時間點（Volume）",
                        min_value=1,
                        max_value=num_time_points,
                        value=1,
                        help=f"此掃描有 {num_time_points} 個 volumes"
                    )
                    selected_time_point_index = selected_time_point_display - 1
                    img_3d_at_t = nimg.index_img(img, selected_time_point_index)
                    title = f"Volume at T={selected_time_point_display}"
                else:
                    img_3d_at_t = img
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
            st.warning("找不到原始 NIfTI 檔案")

else:
    st.info("👈 請在側邊欄選擇受試者和分析框架，然後點擊「開始分析」查看結果。")

# ============================================================================
# 頁尾資訊
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: grey; font-size: 0.9rem;">
    <p><strong>Cognivex CDDA Framework</strong> - Making neuroimaging AI explainable and trustworthy</p>
    <p>整合雙 LLM 架構、MCP 協議和 A2A 模式的自主診斷代理系統</p>
</div>
""", unsafe_allow_html=True)
