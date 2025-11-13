"""
UI components for Structural MRI analysis display
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def render_analysis_mode_selector() -> str:
    """
    Render analysis mode selector in sidebar
    
    Returns:
        Selected analysis mode: "structural" or "functional"
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Configuration")
    
    mode_display = st.sidebar.selectbox(
        "Analysis Mode",
        options=["Structural MRI (T1)", "Functional MRI (fMRI)"],
        index=0,  # Default to Structural MRI
        help="Select the type of MRI analysis to perform",
        key="analysis_mode_selector"
    )
    
    # Map display name to internal mode
    mode_map = {
        "Functional MRI (fMRI)": "functional",
        "Structural MRI (T1)": "structural"
    }
    
    return mode_map[mode_display]


def render_ml_model_info():
    """Render ML model information card for structural MRI"""
    st.sidebar.info("📊 Using Random Forest ML Model")
    
    model_info = {
        "Model Type": "Random Forest Classifier",
        "Features": "32 AAL ROIs",
        "CV Accuracy": "75.4%",
        "Training Data": "65 subjects (ADNI)",
        "Feature Selection": "Hybrid (Literature + Data-driven)"
    }
    
    for key, value in model_info.items():
        st.sidebar.caption(f"**{key}:** {value}")


def render_structural_results(final_state: Dict[str, Any], ground_truth: str):
    """
    Render structural MRI analysis results in professional clinical style
    
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
    
    # Determine risk level and color
    if classification == "AD":
        risk_color = "#FF5252"
    else:
        risk_color = "#4CAF50"
    
    # Professional Header (language-specific)
    if lang == "中文":
        st.markdown(f"""
        <div style='background: white; border-left: 4px solid {risk_color};
                    padding: 20px; margin-bottom: 20px; border-radius: 5px;'>
            <h2 style='color: {risk_color}; margin: 0 0 10px 0; font-size: 1.8em;'>
                阿茲海默症風險評估報告
            </h2>
            <div style='color: #666; font-size: 0.95em;'>
                受試者: {final_state.get('subject_id', 'N/A')} | 
                分析時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: white; border-left: 4px solid {risk_color};
                    padding: 20px; margin-bottom: 20px; border-radius: 5px;'>
            <h2 style='color: {risk_color}; margin: 0 0 10px 0; font-size: 1.8em;'>
                Alzheimer's Disease Risk Assessment Report
            </h2>
            <div style='color: #666; font-size: 0.95em;'>
                Subject: {final_state.get('subject_id', 'N/A')} | 
                Analysis Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        </div>
        <div style='background: white; padding: 20px; border-radius: 10px; margin-top: 20px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <span style='color: #666; font-size: 0.9em;'>評估結果 Assessment Result</span>
                    <h2 style='color: {risk_color}; margin: 5px 0; font-size: 2.5em; font-weight: 700;'>
                        {risk_level}
                    </h2>
                </div>
                <div style='text-align: right;'>
                    <span style='color: #666; font-size: 0.9em;'>信心度 Confidence</span>
                    <h2 style='color: {risk_color}; margin: 5px 0; font-size: 2.5em; font-weight: 700;'>
                        {confidence:.1%}
                    </h2>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical Metrics Dashboard
    st.markdown("""
    <div style='margin: 40px 0 20px 0;'>
        <h2 style='color: #333; margin-bottom: 15px; font-size: 1.8em;'>
            📊 臨床指標 Clinical Metrics
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Clinical metrics with modern cards
    metrics_data = [
        {
            "label": "臨床診斷\nClinical Diagnosis",
            "value": ground_truth,
            "icon": "🏥",
            "color": "#4CAF50" if ground_truth == "NC" else "#FF5252"
        },
        {
            "label": "AI 預測\nAI Prediction",
            "value": classification,
            "icon": "🤖",
            "color": "#4CAF50" if classification == "NC" else "#FF5252"
        },
        {
            "label": "預測信心度\nConfidence",
            "value": f"{confidence:.1%}",
            "icon": "📊",
            "color": "#2196F3"
        },
        {
            "label": "分析模型\nModel",
            "value": "Random Forest",
            "icon": "🌳",
            "color": "#FF9800"
        }
    ]
    
    cols = [col1, col2, col3, col4]
    for col, metric in zip(cols, metrics_data):
        with col:
            st.markdown(f"""
            <div style='background: white; border: 2px solid {metric['color']}30;
                        border-radius: 15px; padding: 20px; text-align: center;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 180px;
                        display: flex; flex-direction: column; justify-content: center;'>
                <div style='font-size: 2.5em; margin-bottom: 10px;'>{metric['icon']}</div>
                <div style='color: #666; font-size: 0.85em; margin-bottom: 10px; line-height: 1.3;
                            white-space: pre-line;'>{metric['label']}</div>
                <div style='font-size: 1.8em; font-weight: 700; color: {metric['color']};'>
                    {metric['value']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Diagnostic Agreement
    st.markdown("<div style='margin: 30px 0 20px 0;'></div>", unsafe_allow_html=True)
    if ground_truth != "N/A" and classification != "ERROR":
        if ground_truth == classification:
            st.success("✅ **診斷一致** AI prediction matches clinical diagnosis", icon="✅")
        else:
            st.warning("⚠️ **診斷不一致** AI prediction differs from clinical diagnosis - 建議進一步評估", icon="⚠️")
    
    # Clinical Report Section
    st.markdown("""
    <div style='margin: 50px 0 30px 0;'>
        <h2 style='color: #333; margin-bottom: 10px; font-size: 1.8em;'>
            📋 臨床報告 Clinical Report
        </h2>
        <p style='color: #666; margin: 0; font-size: 1em;'>
            AI-Generated Clinical Assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display clinical report if available
    generated_reports = final_state.get("generated_reports", {})
    if generated_reports:
        report_zh = generated_reports.get("zh", "")
        report_en = generated_reports.get("en", "")
        
        if report_zh:
            with st.expander("📄 中文報告 Chinese Report", expanded=True):
                st.markdown(f"""
                <div style='background: white; padding: 25px; border-radius: 10px;
                            border-left: 4px solid #2196F3; line-height: 1.8;'>
                    {report_zh}
                </div>
                """, unsafe_allow_html=True)
        
        if report_en:
            with st.expander("📄 English Report", expanded=False):
                st.markdown(f"""
                <div style='background: white; padding: 25px; border-radius: 10px;
                            border-left: 4px solid #2196F3; line-height: 1.8;'>
                    {report_en}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📝 Clinical report generation in progress...")
    
    # Important Brain Regions - Clinical Focus
    st.markdown("""
    <div style='margin: 50px 0 30px 0;'>
        <h2 style='color: #333; margin-bottom: 10px; font-size: 1.8em;'>
            🧠 重要腦區分析 Key Brain Regions
        </h2>
        <p style='color: #666; margin: 0; font-size: 1em;'>
            Top regions contributing to the diagnostic assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    activated_regions = final_state.get("activated_regions", [])
    if activated_regions:
        # Import ROI name mapping
        try:
            from app.core.ml_processing.roi_names_zh import get_roi_display_name, get_roi_category
        except:
            # Fallback if import fails
            def get_roi_display_name(name, lang="zh"):
                return name
            def get_roi_category(name):
                return "未分類"
        
        # Prepare clinical-focused table data
        table_data = []
        for i, region in enumerate(activated_regions[:10], 1):  # Show top 10
            roi_name_en = region.get("region_name", "N/A")
            roi_name_zh = get_roi_display_name(roi_name_en, "zh")
            roi_name_full = f"{roi_name_zh}\n{roi_name_en}"
            category = get_roi_category(roi_name_en)
            importance = region.get('activation_score', 0)
            hemisphere = region.get("hemisphere", "N/A")
            
            # Map hemisphere to Chinese
            hemisphere_zh = {
                "Left": "左側 Left",
                "Right": "右側 Right",
                "Bilateral": "雙側 Bilateral"
            }.get(hemisphere, hemisphere)
            
            table_data.append({
                "排名\nRank": i,
                "腦區名稱\nBrain Region": roi_name_full,
                "重要性\nImportance": f"{importance:.4f}",
                "功能分類\nCategory": category,
                "半球\nHemisphere": hemisphere_zh
            })
        
        df = pd.DataFrame(table_data)
        
        # Display with modern styling
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Download button for full data
        if st.button("📥 Download Full ROI Data"):
            full_data = []
            for region in activated_regions:
                full_data.append({
                    "Rank": region.get("importance_rank"),
                    "ROI_Name": region.get("region_name"),
                    "Importance": region.get("activation_score"),
                    "Hemisphere": region.get("hemisphere"),
                    "Feature_Value": region.get("feature_value"),
                    "Clinical_Relevance": region.get("clinical_relevance"),
                    "Associated_Networks": str(region.get("associated_networks")),
                    "Known_Functions": region.get("known_functions")
                })
            
            full_df = pd.DataFrame(full_data)
            csv = full_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"roi_analysis_{final_state.get('subject_id', 'unknown')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No ROI information available")
    
    # 5. Model Interpretation
    with st.expander("🔍 Model Interpretation Guide"):
        st.markdown("""
        ### Understanding the Results
        
        **Feature Importance**: Indicates which brain regions contributed most to the classification decision.
        Higher importance means the region's characteristics were more influential in the prediction.
        
        **Confidence Score**: Represents the model's certainty in its prediction. 
        - High (>80%): Strong confidence
        - Medium (60-80%): Moderate confidence  
        - Low (<60%): Uncertain prediction
        
        **Clinical Relevance**: Each ROI's known association with Alzheimer's Disease based on 
        neuroscience literature and the model's training data.
        
        ⚠️ **Important**: This is an assistive diagnostic tool and should not be used as the sole 
        basis for clinical decisions. Always consult with qualified healthcare professionals.
        """)


def render_progress_indicator(stage: str, progress: int):
    """
    Render progress indicator for structural MRI analysis
    
    Args:
        stage: Current processing stage
        progress: Progress percentage (0-100)
    """
    progress_bar = st.progress(progress)
    status_text = st.empty()
    
    stage_messages = {
        "loading_model": "Loading ML model components...",
        "extracting_features": "Extracting ROI features from MRI...",
        "predicting": "Running prediction...",
        "analyzing": "Analyzing feature importance...",
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
        "Model loading failed": "⚠️ Unable to load the analysis model. Please contact support.",
        "Feature extraction failed": "⚠️ Could not process the MRI image. Please ensure it's a valid T1-weighted scan.",
        "Atlas loading failed": "⚠️ Brain atlas not found. The system is attempting to download it...",
        "Prediction failed": "⚠️ Analysis could not be completed. Please try again or contact support."
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
            st.error("⚠️ An unexpected error occurred during analysis.")
        
        # Show technical details in expander
        with st.expander("Technical Details"):
            st.code(error)
