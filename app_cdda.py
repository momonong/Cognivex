#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognivex CDDA - Dashboard-First Interface
專業醫療儀表板，整合互動式 AI 顧問
"""

import os
import sys
import streamlit as st
import glob
import plotly.graph_objects as go
from pathlib import Path
import streamlit.components.v1 as components
from datetime import datetime

# 視覺化相關
from nilearn import plotting
from nilearn import image as nimg

# CDDA Framework
from app.agents.cdda_agent import CDDAAgent

# LLM providers for chat
from app.services.llm_providers import llm_response


# ============================================================================
# 工具函數
# ============================================================================

@st.cache_resource(show_spinner="正在載入 NIfTI 檔案...")
def load_nifti(path: str):
    """載入 NIfTI 檔案"""
    try:
        img = nimg.load_img(path)
        if len(img.shape) == 4:
            return img, img.shape[3]
        elif len(img.shape) == 3:
            return img, 1
        else:
            st.error(f"不支援的影像維度: {img.shape}")
            return None, 0
    except Exception as e:
        st.error(f"載入 NIfTI 檔案失敗: {path}. 錯誤: {e}")
        return None, 0


@st.cache_resource(show_spinner="正在初始化 CDDA Agent...")
def initialize_cdda_agent(use_llm=False, orchestrator_model_path=None, consultant_model_path=None):
    """初始化 CDDA Agent"""
    try:
        agent = CDDAAgent(
            orchestrator_model="phi-4-mini",
            orchestrator_model_path=orchestrator_model_path or "D:/hf_models/Phi-4-mini-instruct",
            consultant_model="llama3.1-aloe-beta-8b",
            consultant_model_path=consultant_model_path or r"D:\hf_models\Llama3.1-Aloe-Beta-8B",
            use_llm=use_llm,
            use_4bit=True,
            verbose=True
        )
        return agent
    except Exception as e:
        st.error(f"CDDA Agent 初始化失敗: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def extract_key_insights(result) -> list:
    """從推理鏈中提取關鍵洞察"""
    insights = []
    
    # 檢查反事實分析
    if result.context_object and result.context_object.tool_results:
        tool_results = result.context_object.tool_results
        
        if 'counterfactual' in tool_results:
            cf = tool_results['counterfactual']
            confidence_delta = cf.get('confidence_delta', 0)
            impact = abs(confidence_delta) * 100
            insights.append({
                'icon': '🔬',
                'text': f"執行反事實模擬 (影響: {impact:.1f}%)",
                'type': 'simulation',
                'details': cf.get('interpretation', '')
            })
        
        if 'knowledge_context' in tool_results:
            kc = tool_results['knowledge_context']
            anomalous_regions = kc.get('query_regions', [])
            if anomalous_regions:
                insights.append({
                    'icon': '⚠️',
                    'text': f"檢測到 {len(anomalous_regions)} 個異常腦區",
                    'type': 'anomaly',
                    'details': kc.get('summary', '')
                })
    
    # 添加標準診斷洞察
    if result.confidence > 0.8:
        insights.append({
            'icon': '✅',
            'text': f"高信心度診斷 ({result.confidence:.1%})",
            'type': 'confidence',
            'details': f"模型對 {result.prediction} 診斷具有高度信心"
        })
    elif result.uq_score > 0.8:
        insights.append({
            'icon': '🔍',
            'text': f"高不確定性警示 (UQ: {result.uq_score:.3f})",
            'type': 'uncertainty',
            'details': "建議進行額外臨床驗證"
        })
    
    return insights


def _safe_get_feature_attr(feature, attr_name, default=None):
    """安全地從 Feature 對象或字典中獲取屬性"""
    if isinstance(feature, dict):
        return feature.get(attr_name, default)
    else:
        return getattr(feature, attr_name, default)


def create_shap_chart(result):
    """創建 SHAP 值條形圖"""
    try:
        # 從 context_object 提取 top features
        if not result.context_object or not result.context_object.diagnostic_report:
            return None
        
        report = result.context_object.diagnostic_report
        top_features = report.top_features[:10] if hasattr(report, 'top_features') else []
        
        if not top_features:
            return None
        
        # 準備數據（支持 Feature 對象和字典）
        roi_names = [_safe_get_feature_attr(f, 'roi_name', 'Unknown') for f in top_features]
        shap_values = [_safe_get_feature_attr(f, 'shap_value', 0) for f in top_features]
        z_scores = [_safe_get_feature_attr(f, 'z_score', 0) for f in top_features]
        
        # 創建 Plotly 圖表
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=roi_names,
            x=shap_values,
            orientation='h',
            marker=dict(
                color=shap_values,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="SHAP Value")
            ),
            text=[f"Z={z:.2f}" for z in z_scores],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>SHAP: %{x:.4f}<br>Z-score: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Top 10 診斷驅動因子 (SHAP Values)",
            xaxis_title="SHAP Value (特徵重要性)",
            yaxis_title="腦區 (ROI)",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    except Exception as e:
        print(f"[WARNING] Failed to create SHAP chart: {e}")
        return None


# ============================================================================
# Streamlit 應用主體
# ============================================================================

st.set_page_config(
    page_title="Cognivex CDDA - Clinical Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 側邊欄控制
# ============================================================================

if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False

st.sidebar.header("⚙️ 分析設定")

# 受試者選擇
st.sidebar.subheader("📁 受試者選擇")

subject_labels = {}
data_folders = glob.glob("data/MRI_processed/*/sub-*")
for folder_path in data_folders:
    parts = folder_path.split(os.sep)
    if len(parts) >= 3:
        subject_id = parts[-1]
        label = parts[-2]
        
        # 檢查是否有完整的 MRI 文件（至少 3 個 .nii.gz 文件）
        nii_files = list(Path(folder_path).glob("*.nii.gz"))
        if len(nii_files) >= 3:
            subject_labels[subject_id] = label

subject_list = sorted(subject_labels.keys())
if not subject_list:
    st.sidebar.error("找不到任何有完整數據的受試者。")
    st.sidebar.info("請確認 data/MRI_processed/ 目錄下的受試者資料夾包含至少 3 個 .nii.gz 文件（GM, FA, MD）")
    st.stop()

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
    )
else:
    selected_subject = st.sidebar.selectbox(
        "選擇受試者:",
        subject_list,
        index=default_index,
    )

ground_truth_label = subject_labels.get(selected_subject, "N/A")
st.sidebar.markdown(f"**真實標籤:** `{ground_truth_label}`")

# CDDA 設定
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 CDDA 設定")

use_llm = st.sidebar.checkbox(
    "啟用 LLM 模式",
    value=False,
    help="啟用雙 LLM 架構（Phi-4 + Llama3.1-Aloe-Beta）"
)

if use_llm:
    st.sidebar.markdown("**模型路徑**")
    orchestrator_model_path = st.sidebar.text_input(
        "Agent A (Phi-4)",
        value="D:/hf_models/Phi-4-mini-instruct"
    )
    consultant_model_path = st.sidebar.text_input(
        "Agent B (Llama3.1-Aloe-Beta)",
        value=r"D:\hf_models\Llama3.1-Aloe-Beta-8B"
    )
    st.session_state.orchestrator_model_path = orchestrator_model_path
    st.session_state.consultant_model_path = consultant_model_path

st.session_state.use_llm = use_llm

# 檢查參數變更
prev_subject = st.session_state.get('selected_subject')
if prev_subject and prev_subject != selected_subject:
    st.session_state.run_complete = False
    if 'cdda_result' in st.session_state:
        del st.session_state['cdda_result']
    if 'chat_history' in st.session_state:
        del st.session_state['chat_history']

st.session_state.selected_subject = selected_subject
st.session_state.ground_truth_label = ground_truth_label

# 按鈕區域
st.sidebar.markdown("---")
if is_running:
    st.sidebar.button("分析進行中...", type="primary", use_container_width=True, disabled=True)
    if st.sidebar.button("強制停止", type="secondary", use_container_width=True):
        st.session_state.analysis_running = False
        st.session_state.run_complete = False
        st.rerun()
    start_button = False
else:
    start_button = st.sidebar.button(
        "🚀 開始分析",
        type="primary",
        use_container_width=True,
    )

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.75rem; color: grey;">
Data from ADNI database (adni.loni.usc.edu)
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 分析邏輯
# ============================================================================

if start_button:
    st.session_state.analysis_running = True
    st.session_state.run_complete = False
    if 'cdda_result' in st.session_state:
        del st.session_state['cdda_result']
    if 'chat_history' in st.session_state:
        del st.session_state['chat_history']
    st.rerun()

if st.session_state.get("analysis_running", False) and not st.session_state.get("run_complete", False):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("正在分析..."):
        try:
            import time
            
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
            
            result = agent.run_analysis(st.session_state.selected_subject)
            
            status_text.text("生成報告...")
            progress_bar.progress(70)
            
            st.session_state['cdda_result'] = result
            
            status_text.text("完成！")
            progress_bar.progress(100)
            
            st.session_state['run_complete'] = True
            st.session_state.analysis_running = False
            
            time.sleep(1)
            st.success("✅ 分析完成！")
            st.rerun()
            
        except Exception as e:
            status_text.text("分析失敗")
            progress_bar.progress(0)
            st.error(f"錯誤: {e}")
            st.session_state['run_complete'] = False
            st.session_state.analysis_running = False

# ============================================================================
# Dashboard 顯示
# ============================================================================

if st.session_state.get("run_complete", False):
    result = st.session_state['cdda_result']
    ground_truth = st.session_state.get("ground_truth_label", "N/A")
    
    diagnosis_map = {
        'AD': '阿茲海默症',
        'MCI': '輕度認知障礙',
        'NC': '正常認知'
    }
    
    # ========================================================================
    # 1. Header & Metrics (Top Row)
    # ========================================================================
    
    st.title("🧠 CDDA 臨床診斷儀表板")
    st.markdown(f"**受試者:** {result.subject_id} | **真實標籤:** {diagnosis_map.get(ground_truth, ground_truth)}")
    
    st.markdown("---")
    
    # 主要指標行
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pred_color = "🔴" if result.prediction == 'AD' else ("🟡" if result.prediction == 'MCI' else "🟢")
        st.metric(
            "AI 診斷",
            f"{pred_color} {diagnosis_map.get(result.prediction, result.prediction)}",
            delta="正確" if ground_truth == result.prediction else "錯誤",
            delta_color="normal" if ground_truth == result.prediction else "inverse"
        )
    
    with col2:
        conf_delta = "高" if result.confidence > 0.8 else ("中" if result.confidence > 0.6 else "低")
        st.metric(
            "信心度",
            f"{result.confidence:.1%}",
            delta=conf_delta,
            delta_color="normal" if result.confidence > 0.7 else "inverse"
        )
    
    with col3:
        uq_status = "高不確定性" if result.uq_score > 0.8 else ("中等" if result.uq_score > 0.5 else "低")
        st.metric(
            "不確定性評分",
            f"{result.uq_score:.3f}",
            delta=uq_status,
            delta_color="inverse" if result.uq_score > 0.8 else "normal"
        )
    
    with col4:
        anomaly_count = 0
        anomaly_status = "Pass"
        if result.context_object and result.context_object.tool_results:
            kc = result.context_object.tool_results.get('knowledge_context', {})
            anomaly_count = len(kc.get('query_regions', []))
            anomaly_status = "Alert" if anomaly_count > 0 else "Pass"
        
        st.metric(
            "異常檢查",
            anomaly_status,
            delta=f"{anomaly_count} 個異常區域" if anomaly_count > 0 else "無異常",
            delta_color="inverse" if anomaly_count > 0 else "normal"
        )
    
    with col5:
        decision_icon = "🔬" if result.agent_decision == 'SIMULATION_TRIGGERED' else ("⚠️" if result.agent_decision == 'ANOMALY_INVESTIGATION' else "📋")
        decision_short = {
            'SIMULATION_TRIGGERED': '反事實',
            'ANOMALY_INVESTIGATION': '異常調查',
            'STANDARD_REPORT': '標準'
        }
        st.metric(
            "分析模式",
            f"{decision_icon} {decision_short.get(result.agent_decision, result.agent_decision)}"
        )
    
    st.markdown("---")
    
    # ========================================================================
    # 2. Clinical Executive Summary (At-a-Glance)
    # ========================================================================
    
    st.subheader("📋 臨床執行摘要")
    
    # Extract executive summary from metadata
    executive_summary = result.metadata.get('executive_summary', {})
    
    if executive_summary:
        # Headline
        headline = executive_summary.get('headline', 'No summary available')
        risk_level = executive_summary.get('risk_level', 'Medium')
        
        # Display headline with appropriate styling
        if risk_level == 'High':
            st.error(f"⚠️ **{headline}**")
        elif risk_level == 'Medium':
            st.warning(f"⚡ **{headline}**")
        else:
            st.info(f"✅ **{headline}**")
        
        # Key Findings and Recommended Actions side-by-side
        col_findings, col_actions = st.columns(2)
        
        with col_findings:
            st.markdown("#### 🔍 關鍵發現")
            key_findings = executive_summary.get('key_findings', [])
            if key_findings:
                for finding in key_findings:
                    st.markdown(f"• {finding}")
            else:
                st.caption("無特殊發現")
        
        with col_actions:
            st.markdown("#### 💡 建議行動")
            recommended_actions = executive_summary.get('recommended_actions', [])
            if recommended_actions:
                for action in recommended_actions:
                    st.markdown(f"• {action}")
            else:
                st.caption("標準追蹤即可")
    else:
        # Fallback: use old insight extraction
        st.markdown("**AI 分析邏輯**")
        insights = extract_key_insights(result)
        
        if insights:
            for insight in insights:
                with st.container():
                    col_icon, col_text = st.columns([1, 20])
                    with col_icon:
                        st.markdown(f"### {insight['icon']}")
                    with col_text:
                        st.markdown(f"**{insight['text']}**")
                        if insight['details']:
                            st.caption(insight['details'])
    
    # 顯示完整臨床報告（摺疊）
    if hasattr(result, 'clinical_report') and result.clinical_report:
        with st.expander("📄 查看完整詳細報告", expanded=False):
            st.markdown("### 完整臨床分析報告")
            st.markdown(result.clinical_report)
    
    st.markdown("---")
    
    # ========================================================================
    # 3. Visual Evidence (Middle Row)
    # ========================================================================
    
    st.subheader("📊 視覺化證據")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🧠 MRI 影像")
        
        # 尋找 NIfTI 檔案
        nii_path = None
        possible_paths = [
            f"data/MRI_processed/{ground_truth}/{result.subject_id}/{result.subject_id}_GM_to_MNI.nii.gz",
            f"data/MRI_processed/{ground_truth}/{result.subject_id}/{result.subject_id}_T1w.nii.gz",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                nii_path = path
                break
        
        if nii_path:
            img, _ = load_nifti(nii_path)
            if img:
                try:
                    # 創建靜態切片視圖
                    from nilearn import plotting as niplot
                    import matplotlib
                    matplotlib.use('Agg')  # 使用非互動式後端
                    import matplotlib.pyplot as plt
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    niplot.plot_anat(img, display_mode='x', cut_coords=1, axes=axes[0], title="Sagittal")
                    niplot.plot_anat(img, display_mode='y', cut_coords=1, axes=axes[1], title="Coronal")
                    niplot.plot_anat(img, display_mode='z', cut_coords=1, axes=axes[2], title="Axial")
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning(f"無法顯示 MRI 切片視圖: {e}")
                    st.info("使用互動式檢視器作為替代")
                    # 備用：使用 nilearn 的互動式檢視器
                    viewer = plotting.view_img(img, bg_img=None, cmap="gray", black_bg=True)
                    components.html(viewer.html, height=400, scrolling=False)
        else:
            st.warning("找不到 MRI 影像檔案")
    
    with col_right:
        st.markdown("#### 📈 診斷驅動因子 (SHAP)")
        
        shap_fig = create_shap_chart(result)
        if shap_fig:
            st.plotly_chart(shap_fig, use_container_width=True)
        else:
            st.info("SHAP 數據不可用")
    
    st.markdown("---")
    
    # ========================================================================
    # 4. Contextual Chatbot (Bottom/Expandable)
    # ========================================================================
    
    with st.expander("💬 與 CDDA 顧問討論此案例", expanded=False):
        st.markdown("### 互動式 AI 顧問")
        st.caption("詢問關於此診斷的任何問題，AI 顧問將基於當前分析結果回答")
        
        # 初始化聊天歷史
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            # 添加系統上下文
            system_context = f"""
你是一位專業的神經影像學 AI 顧問。當前案例資訊：

受試者: {result.subject_id}
診斷: {result.prediction} (信心度: {result.confidence:.1%})
不確定性: {result.uq_score:.3f}
分析模式: {result.agent_decision}

臨床報告摘要:
{result.clinical_report[:500] if hasattr(result, 'clinical_report') else '無報告'}

請基於以上資訊回答用戶的問題。保持專業、簡潔，並提供臨床相關的見解。
"""
            st.session_state.system_context = system_context
        
        # 顯示聊天歷史
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
        
        # 聊天輸入
        if prompt := st.chat_input("詢問關於此診斷的問題..."):
            # 添加用戶消息
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            with st.chat_message('user'):
                st.markdown(prompt)
            
            # 生成 AI 回應
            with st.chat_message('assistant'):
                with st.spinner("思考中..."):
                    try:
                        # 構建基於規則的智能回應
                        response_parts = []
                        
                        # 基本回應
                        response_parts.append(f"關於「{prompt}」的問題：\n")
                        
                        # 根據問題關鍵詞提供相關資訊
                        prompt_lower = prompt.lower()
                        
                        if any(word in prompt_lower for word in ['為什麼', 'why', '原因', 'reason']):
                            response_parts.append(f"\n基於當前分析，{result.prediction} 診斷的信心度為 {result.confidence:.1%}。")
                            
                            # 添加關鍵驅動因子
                            if result.context_object and result.context_object.diagnostic_report:
                                top_features = result.context_object.diagnostic_report.top_features[:3]
                                if top_features:
                                    response_parts.append("\n\n**關鍵診斷驅動因子：**")
                                    for i, feat in enumerate(top_features, 1):
                                        roi_name = _safe_get_feature_attr(feat, 'roi_name', 'Unknown')
                                        shap_val = _safe_get_feature_attr(feat, 'shap_value', 0)
                                        z_score = _safe_get_feature_attr(feat, 'z_score', 0)
                                        response_parts.append(f"\n{i}. {roi_name} (SHAP: {shap_val:.4f}, Z-score: {z_score:.2f})")
                        
                        elif any(word in prompt_lower for word in ['不確定', 'uncertainty', '可靠', 'reliable', 'trust']):
                            response_parts.append(f"\n**不確定性評估：**")
                            response_parts.append(f"\n- 不確定性評分: {result.uq_score:.3f}")
                            
                            if result.uq_score > 0.8:
                                response_parts.append("\n- ⚠️ 高不確定性：建議進行額外的臨床驗證")
                                response_parts.append("\n- 系統已執行反事實分析以識別關鍵因子")
                            elif result.uq_score > 0.5:
                                response_parts.append("\n- 中等不確定性：診斷結果需要臨床醫師確認")
                            else:
                                response_parts.append("\n- 低不確定性：模型對此診斷較有信心")
                        
                        elif any(word in prompt_lower for word in ['異常', 'anomaly', '特殊', 'unusual']):
                            if result.context_object and result.context_object.tool_results:
                                kc = result.context_object.tool_results.get('knowledge_context', {})
                                anomalous_regions = kc.get('query_regions', [])
                                
                                if anomalous_regions:
                                    response_parts.append(f"\n**異常檢測結果：**")
                                    response_parts.append(f"\n檢測到 {len(anomalous_regions)} 個統計異常腦區：")
                                    for region in anomalous_regions[:5]:
                                        response_parts.append(f"\n- {region}")
                                    
                                    summary = kc.get('summary', '')
                                    if summary:
                                        response_parts.append(f"\n\n**臨床意義：**\n{summary}")
                                else:
                                    response_parts.append("\n未檢測到統計異常的腦區。")
                        
                        elif any(word in prompt_lower for word in ['反事實', 'counterfactual', '模擬', 'simulation']):
                            if result.context_object and result.context_object.tool_results:
                                cf = result.context_object.tool_results.get('counterfactual', {})
                                
                                if cf:
                                    response_parts.append("\n**反事實模擬結果：**")
                                    response_parts.append(f"\n- 原始預測: {cf.get('original_prediction', 'N/A')} ({cf.get('original_confidence', 0):.1%})")
                                    response_parts.append(f"\n- 遮蔽關鍵特徵後: {cf.get('new_prediction', 'N/A')} ({cf.get('new_confidence', 0):.1%})")
                                    response_parts.append(f"\n- 信心度變化: {cf.get('confidence_delta', 0):+.1%}")
                                    response_parts.append(f"\n\n**解釋：** {cf.get('interpretation', 'N/A')}")
                                else:
                                    response_parts.append("\n此案例未執行反事實模擬（不確定性較低）。")
                        
                        else:
                            # 通用回應
                            response_parts.append(f"\n基於當前分析：")
                            response_parts.append(f"\n- 診斷: {result.prediction}")
                            response_parts.append(f"\n- 信心度: {result.confidence:.1%}")
                            response_parts.append(f"\n- 不確定性: {result.uq_score:.3f}")
                            response_parts.append(f"\n- 分析模式: {result.agent_decision}")
                        
                        # 添加 LLM 模式提示
                        if not st.session_state.get('use_llm', False):
                            response_parts.append("\n\n💡 *提示：啟用 LLM 模式可獲得更深入的 AI 對話體驗*")
                        
                        response = ''.join(response_parts)
                        
                        st.markdown(response)
                        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                    
                    except Exception as e:
                        error_msg = f"抱歉，處理您的問題時發生錯誤：{str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({'role': 'assistant', 'content': error_msg})
    
    # ========================================================================
    # 技術細節（可選）
    # ========================================================================
    
    with st.expander("🔧 技術細節與推理鏈", expanded=False):
        tab1, tab2 = st.tabs(["元數據", "完整推理鏈"])
        
        with tab1:
            st.json(result.metadata)
        
        with tab2:
            for step in result.reasoning_chain:
                if step.startswith("="*80):
                    st.markdown(f"**{step.replace('=', '')}**")
                elif step.startswith("-"*80):
                    st.markdown(f"*{step.replace('-', '')}*")
                else:
                    st.text(step)

else:
    # 歡迎畫面
    st.title("🧠 Cognivex CDDA 臨床診斷儀表板")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 智能診斷")
        st.markdown("雙 LLM 架構提供高精度 AD/MCI/NC 分類")
    
    with col2:
        st.markdown("### 🔬 可解釋 AI")
        st.markdown("反事實分析與 SHAP 值揭示診斷邏輯")
    
    with col3:
        st.markdown("### 💬 互動顧問")
        st.markdown("與 AI 討論案例，獲得深入見解")
    
    st.markdown("---")
    st.info("👈 請在側邊欄選擇受試者，然後點擊「開始分析」")

# 頁尾
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: grey; font-size: 0.9rem;">
    <p><strong>Cognivex CDDA Framework</strong> - Dashboard-First Clinical AI</p>
    <p>Phi-4-mini (Orchestrator) + Llama3.1-Aloe-Beta-8B (Consultant)</p>
</div>
""", unsafe_allow_html=True)
