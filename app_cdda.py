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
def initialize_cdda_agent(use_llm=False, orchestrator_model_path=None, consultant_model_path=None):
    """初始化 CDDA Agent（快取以避免重複載入）"""
    try:
        agent = CDDAAgent(
            orchestrator_model="gpt-oss-20b",
            orchestrator_model_path=orchestrator_model_path,
            consultant_model="medgemma-27b",
            consultant_model_path=consultant_model_path,
            use_llm=use_llm,
            load_in_8bit=True,  # 使用 8-bit 量化節省記憶體
            verbose=True
        )
        return agent
    except Exception as e:
        st.error(f"CDDA Agent 初始化失敗: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def format_cdda_report(result) -> str:
    """格式化 CDDA 分析結果為臨床報告"""
    
    # 診斷結果的中文翻譯
    diagnosis_map = {
        'AD': '阿茲海默症 (Alzheimer\'s Disease)',
        'MCI': '輕度認知障礙 (Mild Cognitive Impairment)',
        'NC': '正常認知 (Normal Cognition)'
    }
    
    # 代理決策的中文翻譯
    decision_map = {
        'SIMULATION_TRIGGERED': '高不確定性 - 已執行反事實模擬',
        'ANOMALY_INVESTIGATION': '異常模式 - 已查詢知識圖譜',
        'STANDARD_REPORT': '標準診斷流程'
    }
    
    diagnosis_text = diagnosis_map.get(result.prediction, result.prediction)
    decision_text = decision_map.get(result.agent_decision, result.agent_decision)
    
    # 信心度解釋
    if result.confidence > 0.8:
        confidence_level = "高信心度"
        confidence_color = "#388e3c"
    elif result.confidence > 0.6:
        confidence_level = "中等信心度"
        confidence_color = "#f57c00"
    else:
        confidence_level = "低信心度"
        confidence_color = "#d32f2f"
    
    # 不確定性解釋
    if result.uq_score > 0.8:
        uq_level = "高不確定性 - 建議進一步檢查"
        uq_color = "#d32f2f"
    elif result.uq_score > 0.5:
        uq_level = "中等不確定性"
        uq_color = "#f57c00"
    else:
        uq_level = "低不確定性"
        uq_color = "#388e3c"
    
    html = f"""
    <div style="background-color: #ffffff; padding: 25px; border-radius: 10px; margin: 10px 0; border: 2px solid #e0e0e0;">
        <h2 style="color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 10px;">
            🏥 臨床診斷報告
        </h2>
        
        <div style="margin: 20px 0; padding: 15px; background-color: #e3f2fd; border-left: 4px solid #1976d2;">
            <h3 style="color: #1565c0; margin-top: 0;">📋 診斷結果</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px; font-weight: bold; width: 30%;">受試者編號</td>
                    <td style="padding: 8px;">{result.subject_id}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">診斷</td>
                    <td style="padding: 8px; color: {'#d32f2f' if result.prediction == 'AD' else '#388e3c'}; font-weight: bold; font-size: 1.1em;">
                        {diagnosis_text}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">診斷信心度</td>
                    <td style="padding: 8px;">
                        <span style="color: {confidence_color}; font-weight: bold;">{result.confidence:.1%}</span>
                        <span style="color: #666; margin-left: 10px;">({confidence_level})</span>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">模型不確定性</td>
                    <td style="padding: 8px;">
                        <span style="color: {uq_color}; font-weight: bold;">{result.uq_score:.3f}</span>
                        <span style="color: #666; margin-left: 10px;">({uq_level})</span>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">分析模式</td>
                    <td style="padding: 8px;">{decision_text}</td>
                </tr>
            </table>
        </div>
    """
    
    # 顯示臨床報告（來自 Agent B）
    if hasattr(result, 'clinical_report') and result.clinical_report:
        html += f"""
        <div style="margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px;">
            <h3 style="color: #424242;">📝 臨床分析</h3>
            <div style="white-space: pre-wrap; font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6;">
{result.clinical_report}
            </div>
        </div>
        """
    
    # 添加反事實分析（如果有）
    if result.context_object and result.context_object.tool_results:
        tool_results = result.context_object.tool_results
        
        if 'counterfactual' in tool_results:
            cf = tool_results['counterfactual']
            confidence_delta = cf.get('confidence_delta', 0)
            
            # 判斷影響程度
            if abs(confidence_delta) > 0.1:
                impact_level = "關鍵診斷驅動因子"
                impact_color = "#d32f2f"
            elif abs(confidence_delta) > 0.05:
                impact_level = "中等影響"
                impact_color = "#f57c00"
            else:
                impact_level = "輕微影響"
                impact_color = "#388e3c"
            
            html += f"""
            <div style="margin: 20px 0; padding: 15px; background-color: #fff3e0; border-left: 4px solid #ff9800; border-radius: 5px;">
                <h3 style="color: #e65100;">🔬 反事實模擬分析</h3>
                <p style="color: #666; font-style: italic;">
                    此分析用於識別哪些腦區對診斷結果影響最大
                </p>
                <table style="width: 100%; margin-top: 10px;">
                    <tr>
                        <td style="padding: 5px; font-weight: bold;">原始預測</td>
                        <td style="padding: 5px;">{cf.get('original_prediction', 'N/A')} ({cf.get('original_confidence', 0):.1%})</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; font-weight: bold;">遮蔽關鍵特徵後</td>
                        <td style="padding: 5px;">{cf.get('new_prediction', 'N/A')} ({cf.get('new_confidence', 0):.1%})</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; font-weight: bold;">信心度變化</td>
                        <td style="padding: 5px;">
                            <span style="color: {impact_color}; font-weight: bold;">{confidence_delta:+.1%}</span>
                            <span style="color: #666; margin-left: 10px;">({impact_level})</span>
                        </td>
                    </tr>
                </table>
                <div style="margin-top: 10px; padding: 10px; background-color: #ffffff; border-radius: 3px;">
                    <strong>臨床意義:</strong> {cf.get('interpretation', 'N/A')}
                </div>
            </div>
            """
        
        # 添加異常分析（如果有）
        if 'knowledge_context' in tool_results:
            kc = tool_results['knowledge_context']
            anomalous_regions = kc.get('query_regions', [])
            
            if anomalous_regions:
                html += f"""
                <div style="margin: 20px 0; padding: 15px; background-color: #ffebee; border-left: 4px solid #f44336; border-radius: 5px;">
                    <h3 style="color: #c62828;">⚠️ 異常模式檢測</h3>
                    <p style="color: #666;">
                        檢測到 <strong>{len(anomalous_regions)}</strong> 個腦區呈現統計異常模式（可能提示混合病理或非典型表現）
                    </p>
                    <div style="margin: 10px 0;">
                        <strong>異常腦區:</strong>
                        <ul style="margin: 5px 0;">
                """
                for region in anomalous_regions[:5]:
                    html += f"<li>{region}</li>"
                html += """
                        </ul>
                    </div>
                """
                
                # 顯示知識圖譜摘要
                summary = kc.get('summary', '')
                if summary:
                    html += f"""
                    <div style="margin-top: 10px; padding: 10px; background-color: #ffffff; border-radius: 3px;">
                        <strong>臨床背景知識:</strong>
                        <p style="margin: 5px 0;">{summary}</p>
                    </div>
                    """
                
                html += "</div>"
    
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
    
    # HuggingFace 模型路徑設定
    if use_llm:
        st.sidebar.markdown("**HuggingFace 模型設定**")
        
        orchestrator_model_path = st.sidebar.text_input(
            "Agent A 模型路徑",
            value="D:/hf_models/gpt-oss-20b",
            help="Agent A (Orchestrator) 的 HuggingFace 模型路徑"
        )
        
        consultant_model_path = st.sidebar.text_input(
            "Agent B 模型路徑",
            value="D:/hf_models/medgemma-27b",
            help="Agent B (Consultant) 的 HuggingFace 模型路徑"
        )
        
        st.session_state.orchestrator_model_path = orchestrator_model_path
        st.session_state.consultant_model_path = consultant_model_path
    
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
                orchestrator_model_path = st.session_state.get('orchestrator_model_path', None) if use_llm else None
                consultant_model_path = st.session_state.get('consultant_model_path', None) if use_llm else None
                
                agent = initialize_cdda_agent(
                    use_llm=use_llm,
                    orchestrator_model_path=orchestrator_model_path,
                    consultant_model_path=consultant_model_path
                )
                
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
        
        # 診斷結果的中文翻譯
        diagnosis_map = {
            'AD': '阿茲海默症',
            'MCI': '輕度認知障礙',
            'NC': '正常認知'
        }
        
        # 顯示診斷報告
        report_html = format_cdda_report(result)
        st.markdown(report_html, unsafe_allow_html=True)
        
        # 預測驗證（更清晰的顯示）
        st.markdown("---")
        st.subheader("🎯 診斷驗證")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "真實診斷", 
                diagnosis_map.get(ground_truth, ground_truth),
                help="受試者的實際診斷標籤"
            )
        
        with col2:
            st.metric(
                "AI 預測", 
                diagnosis_map.get(result.prediction, result.prediction),
                help="CDDA 系統的預測結果"
            )
        
        with col3:
            if ground_truth == result.prediction:
                st.success("✅ 預測正確")
            else:
                st.error("❌ 預測錯誤")
                st.warning(f"預測為 {diagnosis_map.get(result.prediction, result.prediction)}，實際為 {diagnosis_map.get(ground_truth, ground_truth)}")
        
        # 關鍵指標
        st.markdown("---")
        st.subheader("📊 關鍵診斷指標")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence_delta = None
            if result.confidence > 0.8:
                confidence_delta = "高"
                confidence_color = "normal"
            elif result.confidence > 0.6:
                confidence_delta = "中"
                confidence_color = "off"
            else:
                confidence_delta = "低"
                confidence_color = "inverse"
            
            st.metric(
                "診斷信心度", 
                f"{result.confidence:.1%}",
                delta=confidence_delta,
                delta_color=confidence_color,
                help="模型對診斷結果的信心程度"
            )
        
        with col2:
            uq_delta = None
            if result.uq_score > 0.8:
                uq_delta = "高 - 需注意"
                uq_color = "inverse"
            elif result.uq_score > 0.5:
                uq_delta = "中等"
                uq_color = "off"
            else:
                uq_delta = "低"
                uq_color = "normal"
            
            st.metric(
                "不確定性評分", 
                f"{result.uq_score:.3f}",
                delta=uq_delta,
                delta_color=uq_color,
                help="模型的不確定性程度，越高表示需要更多臨床驗證"
            )
        
        with col3:
            # 顯示異常區域數量
            anomaly_count = 0
            if result.context_object and result.context_object.tool_results:
                kc = result.context_object.tool_results.get('knowledge_context', {})
                anomaly_count = len(kc.get('query_regions', []))
            
            st.metric(
                "異常腦區數量", 
                anomaly_count,
                help="檢測到統計異常的腦區數量"
            )
        
        with col4:
            # 顯示是否執行了反事實分析
            cf_performed = "是" if (result.context_object and 
                                   result.context_object.tool_results and 
                                   'counterfactual' in result.context_object.tool_results) else "否"
            
            st.metric(
                "反事實分析", 
                cf_performed,
                help="是否執行了反事實模擬以識別關鍵診斷驅動因子"
            )
        
        # 顯示推理鏈
        if st.session_state.get('show_reasoning', True) and result.reasoning_chain:
            display_reasoning_chain(result.reasoning_chain)
        
        # 顯示元數據
        with st.expander("📋 技術細節與元數據", expanded=False):
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
    # 互動式 MRI 檢視器（兩種模式共用）
    # ====================================================================
    with st.expander("🔍 探索原始 MRI 掃描（互動式切片器）", expanded=False):
        # 嘗試找到 NIfTI 檔案
        nii_path = st.session_state.get("nii_path")
        
        # 如果沒有 nii_path，嘗試從 MRI_processed 目錄找
        if not nii_path or not Path(nii_path).exists():
            selected_subject = st.session_state.get("selected_subject")
            ground_truth_label = st.session_state.get("ground_truth_label", "N/A")
            
            if selected_subject and ground_truth_label != "N/A":
                # 嘗試多個可能的路徑和檔案名稱模式
                possible_paths = [
                    # MRI_processed 目錄 (處理後的檔案)
                    f"data/MRI_processed/{ground_truth_label}/{selected_subject}/{selected_subject}_GM_to_MNI.nii.gz",
                    f"data/MRI_processed/{ground_truth_label}/{selected_subject}/{selected_subject}_FA_to_MNI.nii.gz",
                    f"data/MRI_processed/{ground_truth_label}/{selected_subject}/{selected_subject}_MD_to_MNI.nii.gz",
                    # 標準 BIDS 格式
                    f"data/MRI_processed/{ground_truth_label}/{selected_subject}/anat/{selected_subject}_T1w.nii.gz",
                    f"data/MRI_processed/{ground_truth_label}/{selected_subject}/{selected_subject}_T1w.nii.gz",
                    # sMRI 目錄
                    f"data/sMRI/{ground_truth_label}/{selected_subject}/anat/{selected_subject}_T1w.nii.gz",
                    f"data/sMRI/{ground_truth_label}/{selected_subject}/{selected_subject}_T1w.nii.gz",
                    # fMRI 目錄
                    f"data/fMRI/{ground_truth_label}/{selected_subject}/func/{selected_subject}_task-rest_bold.nii.gz",
                    f"data/fMRI/{ground_truth_label}/{selected_subject}/{selected_subject}_task-rest_bold.nii.gz",
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        nii_path = path
                        st.session_state["nii_path"] = nii_path
                        break
        
        if nii_path and Path(nii_path).exists():
            st.info(f"📁 載入檔案: {nii_path}")
            
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
            st.warning(f"找不到原始 NIfTI 檔案。受試者: {st.session_state.get('selected_subject', 'N/A')}")
            st.info("請確認資料位於以下目錄之一：\n- data/MRI_processed/\n- data/sMRI/\n- data/fMRI/")

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
