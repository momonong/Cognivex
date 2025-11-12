"""
Professional UI components for Structural MRI analysis
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def render_analysis_mode_selector() -> str:
    """Render analysis mode selector in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Configuration")
    
    mode_display = st.sidebar.selectbox(
        "Analysis Mode",
        options=["Structural MRI (T1)", "Functional MRI (fMRI)"],
        index=0,
        help="Select the type of MRI analysis to perform",
        key="analysis_mode_selector"
    )
    
    mode_map = {
        "Functional MRI (fMRI)": "functional",
        "Structural MRI (T1)": "structural"
    }
    
    return mode_map[mode_display]


def render_ml_model_info():
    """Render ML model information"""
    st.sidebar.info("Using Random Forest ML Model")
    
    model_info = {
        "Model Type": "Random Forest Classifier",
        "Features": "32 AAL ROIs",
        "CV Accuracy": "75.4%",
        "Training Data": "65 subjects (ADNI)"
    }
    
    for key, value in model_info.items():
        st.sidebar.caption(f"**{key}:** {value}")


def render_structural_results(final_state: Dict[str, Any], ground_truth: str):
    """
    Render professional clinical dashboard
    
    Args:
        final_state: Final state from workflow
        ground_truth: Ground truth label
    """
    st.markdown("---")
    
    # Language selector
    lang = st.selectbox(
        "Language / 語言",
        options=["中文", "English"],
        index=0,
        key="language_selector"
    )
    
    # Map numeric prediction to label
    classification_raw = final_state.get("classification_result", "N/A")
    if classification_raw == 0 or classification_raw == "0":
        classification = "NC"
    elif classification_raw == 1 or classification_raw == "1":
        classification = "AD"
    else:
        classification = str(classification_raw)
    
    confidence = final_state.get("prediction_confidence", 0)
    risk_color = "#FF5252" if classification == "AD" else "#4CAF50"
    
    # === HEADER ===
    if lang == "中文":
        st.markdown(f"""
        <div style='background: white; border-left: 4px solid {risk_color};
                    padding: 15px; margin-bottom: 15px;'>
            <h3 style='color: {risk_color}; margin: 0;'>阿茲海默症風險評估報告</h3>
            <div style='color: #666; font-size: 0.9em; margin-top: 5px;'>
                受試者: {final_state.get('subject_id', 'N/A')} | 
                分析時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: white; border-left: 4px solid {risk_color};
                    padding: 15px; margin-bottom: 15px;'>
            <h3 style='color: {risk_color}; margin: 0;'>Alzheimer's Disease Risk Assessment Report</h3>
            <div style='color: #666; font-size: 0.9em; margin-top: 5px;'>
                Subject: {final_state.get('subject_id', 'N/A')} | 
                Analysis Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # === CLINICAL METRICS ===
    if lang == "中文":
        st.markdown("### 臨床指標")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("臨床診斷", ground_truth)
        with col2:
            st.metric("AI 預測", classification)
        with col3:
            st.metric("信心度", f"{confidence:.1%}")
        with col4:
            st.metric("模型", "Random Forest")
        with col5:
            # Get number of analyzed regions
            activated_regions = final_state.get("activated_regions", [])
            st.metric("分析腦區", f"{len(activated_regions)}")
        
        # Diagnostic agreement and analysis summary
        if ground_truth != "N/A" and classification != "ERROR":
            if ground_truth == classification:
                st.success("診斷一致")
            else:
                st.warning("診斷不一致 - 建議進一步評估")
        
        # Analysis summary
        with st.expander("📊 分析摘要", expanded=False):
            activated_regions = final_state.get("activated_regions", [])
            if activated_regions:
                # Calculate statistics
                importances = [r.get('activation_score', 0) for r in activated_regions]
                avg_importance = sum(importances) / len(importances) if importances else 0
                max_importance = max(importances) if importances else 0
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("總腦區數", len(activated_regions))
                with col_b:
                    st.metric("平均重要性", f"{avg_importance:.4f}")
                with col_c:
                    st.metric("最高重要性", f"{max_importance:.4f}")
                
                # Top 3 regions
                st.markdown("**前三重要腦區:**")
                for i, region in enumerate(activated_regions[:3], 1):
                    roi_name = region.get("region_name", "N/A")
                    importance = region.get("activation_score", 0)
                    st.markdown(f"{i}. {roi_name}: {importance:.4f}")
    else:
        st.markdown("### Clinical Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Clinical Diagnosis", ground_truth)
        with col2:
            st.metric("AI Prediction", classification)
        with col3:
            st.metric("Confidence", f"{confidence:.1%}")
        with col4:
            st.metric("Model", "Random Forest")
        with col5:
            # Get number of analyzed regions
            activated_regions = final_state.get("activated_regions", [])
            st.metric("Brain Regions", f"{len(activated_regions)}")
        
        # Diagnostic agreement and analysis summary
        if ground_truth != "N/A" and classification != "ERROR":
            if ground_truth == classification:
                st.success("Diagnostic Agreement")
            else:
                st.warning("Diagnostic Disagreement - Further evaluation recommended")
        
        # Analysis summary
        with st.expander("📊 Analysis Summary", expanded=False):
            activated_regions = final_state.get("activated_regions", [])
            if activated_regions:
                # Calculate statistics
                importances = [r.get('activation_score', 0) for r in activated_regions]
                avg_importance = sum(importances) / len(importances) if importances else 0
                max_importance = max(importances) if importances else 0
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Regions", len(activated_regions))
                with col_b:
                    st.metric("Avg Importance", f"{avg_importance:.4f}")
                with col_c:
                    st.metric("Max Importance", f"{max_importance:.4f}")
                
                # Top 3 regions
                st.markdown("**Top 3 Important Regions:**")
                for i, region in enumerate(activated_regions[:3], 1):
                    roi_name = region.get("region_name", "N/A")
                    importance = region.get("activation_score", 0)
                    st.markdown(f"{i}. {roi_name}: {importance:.4f}")
    
    # === STRUCTURED CLINICAL REPORT ===
    st.markdown("---")
    
    structured_report = final_state.get("structured_report", {})
    if structured_report:
        # Map language selection to report keys
        lang_key = "zh" if lang == "中文" else "en"
        report_data = structured_report.get(lang_key, structured_report.get("en", {}))
        
        if report_data:
            # Primary Finding
            if lang == "中文":
                st.markdown("### 主要發現")
            else:
                st.markdown("### Primary Finding")
            
            primary_finding = report_data.get("risk_assessment", {}).get("primary_finding", "")
            if primary_finding:
                st.info(primary_finding)
            
            # Key Findings
            st.markdown("---")
            if lang == "中文":
                st.markdown("### 關鍵發現")
            else:
                st.markdown("### Key Findings")
            
            key_findings = report_data.get("key_findings", {})
            
            # Structural Changes
            structural_changes = key_findings.get("structural_changes", [])
            if structural_changes:
                if lang == "中文":
                    st.markdown("**結構性變化**")
                else:
                    st.markdown("**Structural Changes**")
                
                for change in structural_changes:
                    severity = change.get("severity", "Unknown")
                    severity_icon = {
                        "Severe": "🔴",
                        "Moderate": "🟡",
                        "Mild": "🟢"
                    }.get(severity, "⚪")
                    
                    finding = change.get("finding", "")
                    st.markdown(f"{severity_icon} {finding} ({severity})")
            
            # Volumetric Analysis
            volumetric = key_findings.get("volumetric_analysis", [])
            if volumetric:
                st.markdown("")
                if lang == "中文":
                    st.markdown("**體積分析**")
                else:
                    st.markdown("**Volumetric Analysis**")
                
                for vol in volumetric:
                    region = vol.get("region", "")
                    change = vol.get("change", "")
                    percentage = vol.get("percentage", "")
                    
                    if percentage.startswith("-"):
                        icon = "▼"
                    elif percentage.startswith("+"):
                        icon = "▲"
                    else:
                        icon = "="
                    
                    st.markdown(f"• {region}: {change} {percentage} {icon}")
            
            # Clinical Interpretation
            st.markdown("---")
            if lang == "中文":
                st.markdown("### 臨床解釋")
            else:
                st.markdown("### Clinical Interpretation")
            
            interpretation = report_data.get("clinical_interpretation", {})
            summary = interpretation.get("summary", "")
            if summary:
                st.markdown(f"*{summary}*")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if lang == "中文":
                    st.markdown("**AD 指標**")
                else:
                    st.markdown("**AD Indicators**")
                
                ad_indicators = interpretation.get("ad_indicators", [])
                if ad_indicators:
                    for indicator in ad_indicators:
                        st.markdown(f"⚠️ {indicator}")
                else:
                    if lang == "中文":
                        st.markdown("*未發現明顯 AD 指標*")
                    else:
                        st.markdown("*No significant AD indicators detected*")
            
            with col2:
                if lang == "中文":
                    st.markdown("**保護因子**")
                else:
                    st.markdown("**Protective Factors**")
                
                protective = interpretation.get("protective_factors", [])
                if protective:
                    for factor in protective:
                        st.markdown(f"✓ {factor}")
                else:
                    if lang == "中文":
                        st.markdown("*無保護因子資訊*")
                    else:
                        st.markdown("*No protective factors information*")
            
            # Recommendations
            st.markdown("---")
            if lang == "中文":
                st.markdown("### 建議")
            else:
                st.markdown("### Recommendations")
            
            recommendations = report_data.get("recommendations", {})
            
            immediate = recommendations.get("immediate_actions", [])
            if lang == "中文":
                st.markdown("**立即行動**")
            else:
                st.markdown("**Immediate Actions**")
            
            if immediate:
                for i, action in enumerate(immediate, 1):
                    st.markdown(f"{i}. {action}")
            else:
                if lang == "中文":
                    st.markdown("*無需立即行動*")
                else:
                    st.markdown("*No immediate actions required*")
            
            monitoring = recommendations.get("monitoring", [])
            st.markdown("")
            if lang == "中文":
                st.markdown("**監測項目**")
            else:
                st.markdown("**Monitoring**")
            
            if monitoring:
                for item in monitoring:
                    st.markdown(f"• {item}")
            else:
                if lang == "中文":
                    st.markdown("*定期臨床追蹤*")
                else:
                    st.markdown("*Regular clinical follow-up*")
            
            additional = recommendations.get("additional_tests", [])
            if additional:
                st.markdown("")
                if lang == "中文":
                    st.markdown("**額外檢查**")
                else:
                    st.markdown("**Additional Tests**")
                for test in additional:
                    st.markdown(f"• {test}")
            
            # Limitations
            limitations = report_data.get("limitations", [])
            if limitations:
                with st.expander("⚠️ Limitations" if lang == "English" else "⚠️ 限制"):
                    for limitation in limitations:
                        st.markdown(f"• {limitation}")
        else:
            # Report data is empty - show debug info
            if lang == "中文":
                st.warning("⚠️ 報告資料為空")
                with st.expander("調試資訊"):
                    st.write("structured_report 內容:", structured_report)
                    st.write("語言鍵:", lang_key)
            else:
                st.warning("⚠️ Report data is empty")
                with st.expander("Debug Info"):
                    st.write("structured_report content:", structured_report)
                    st.write("Language key:", lang_key)
    else:
        # No structured_report in final_state
        if lang == "中文":
            st.warning("⚠️ 未找到報告資料")
            with st.expander("調試資訊"):
                st.write("final_state 鍵:", list(final_state.keys()))
                st.write("是否包含 structured_report:", "structured_report" in final_state)
        else:
            st.warning("⚠️ No report data found")
            with st.expander("Debug Info"):
                st.write("final_state keys:", list(final_state.keys()))
                st.write("Has structured_report:", "structured_report" in final_state)
    
    # === KEY BRAIN REGIONS ===
    st.markdown("---")
    if lang == "中文":
        st.markdown("### 重要腦區分析")
    else:
        st.markdown("### Key Brain Regions")
    
    activated_regions = final_state.get("activated_regions", [])
    if activated_regions:
        try:
            from app.core.ml_processing.roi_names_zh import get_roi_display_name, get_roi_category
        except:
            def get_roi_display_name(name, lang_code="zh"):
                return name
            def get_roi_category(name):
                return "未分類" if lang == "中文" else "Uncategorized"
        
        # Prepare table data
        table_data = []
        for i, region in enumerate(activated_regions[:10], 1):
            roi_name_en = region.get("region_name", "N/A")
            importance = region.get('activation_score', 0)
            hemisphere = region.get("hemisphere", "N/A")
            
            if lang == "中文":
                roi_name = get_roi_display_name(roi_name_en, "zh")
                category = get_roi_category(roi_name_en)
                hemisphere_display = {
                    "Left": "左側",
                    "Right": "右側",
                    "Bilateral": "雙側"
                }.get(hemisphere, hemisphere)
                
                table_data.append({
                    "排名": i,
                    "腦區名稱": roi_name,
                    "重要性": f"{importance:.4f}",
                    "功能分類": category,
                    "半球": hemisphere_display
                })
            else:
                category = get_roi_category(roi_name_en)
                # Use English category if available
                category_en_map = {
                    "記憶系統": "Memory System",
                    "預設模式網絡": "Default Mode Network",
                    "視覺處理": "Visual Processing",
                    "語言功能": "Language Function",
                    "執行功能": "Executive Function"
                }
                category = category_en_map.get(category, category)
                
                table_data.append({
                    "Rank": i,
                    "Brain Region": roi_name_en,
                    "Importance": f"{importance:.4f}",
                    "Category": category,
                    "Hemisphere": hemisphere
                })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, width='stretch', hide_index=True, height=400)
    else:
        if lang == "中文":
            st.info("無腦區資訊")
        else:
            st.info("No brain region information available")
    
    # === CLINICAL COMMENT ===
    st.markdown("---")
    if lang == "中文":
        st.markdown("### 臨床備註")
        st.text_area(
            "醫師備註",
            placeholder="請輸入臨床觀察、建議或其他相關資訊...",
            height=100,
            key="clinical_comment_zh"
        )
    else:
        st.markdown("### Clinical Comment")
        st.text_area(
            "Physician's Notes",
            placeholder="Enter clinical observations, recommendations, or other relevant information...",
            height=100,
            key="clinical_comment_en"
        )


def render_progress_indicator(stage: str, progress: int):
    """Render progress indicator"""
    progress_bar = st.progress(progress)
    status_text = st.empty()
    
    stage_messages = {
        "loading_model": "Loading ML model...",
        "extracting_features": "Extracting ROI features...",
        "predicting": "Running prediction...",
        "analyzing": "Analyzing features...",
        "visualizing": "Generating visualizations...",
        "complete": "Analysis complete!"
    }
    
    message = stage_messages.get(stage, "Processing...")
    status_text.text(message)
    
    return progress_bar, status_text



def render_error_message(error_log: list):
    """
    Render user-friendly error messages
    
    Args:
        error_log: List of error messages
    """
    if not error_log:
        return
    
    # Map technical errors to user-friendly messages
    error_map = {
        "Model loading failed": "Unable to load the analysis model. Please contact support.",
        "Feature extraction failed": "Could not process the MRI image. Please ensure it's a valid T1-weighted scan.",
        "Atlas loading failed": "Brain atlas not found. The system is attempting to download it...",
        "Prediction failed": "Analysis could not be completed. Please try again or contact support."
    }
    
    for error in error_log:
        # Find matching friendly message
        friendly_msg = None
        for key, msg in error_map.items():
            if key.lower() in error.lower():
                friendly_msg = msg
                break
        
        if friendly_msg:
            st.error(friendly_msg)
        else:
            st.error("An unexpected error occurred during analysis.")
        
        # Show technical details in expander
        with st.expander("Technical Details"):
            st.code(error)
