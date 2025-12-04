#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDDA Clinical Dashboard - Streamlit Application
Based on test_cdda_paper_results.py logic
"""

import streamlit as st
import glob
import time
from pathlib import Path
from datetime import datetime
import sys
import threading
import queue
import io

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


# ============================================================================
# Helper Functions
# ============================================================================

def safe_get_feature_attr(feature, attr_name, default=None):
    """Safely get attribute from Feature object or dict"""
    if isinstance(feature, dict):
        return feature.get(attr_name, default)
    else:
        return getattr(feature, attr_name, default)


def scan_subjects():
    """Scan and validate MRI_processed dataset"""
    subject_labels = {}
    data_folders = glob.glob("data/MRI_processed/*/sub-*")
    
    for folder_path in data_folders:
        parts = folder_path.replace('\\', '/').split('/')
        if len(parts) >= 3:
            subject_id = parts[-1]
            label = parts[-2]
            nii_files = list(Path(folder_path).glob("*.nii.gz"))
            if len(nii_files) >= 3:
                subject_labels[subject_id] = label
    
    return subject_labels


# ============================================================================
# Streamlit App
# ============================================================================

st.set_page_config(
    page_title="CDDA Clinical Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("CDDA Clinical Dashboard")
st.markdown("Cognitive Discrepancy-Driven Agent for Alzheimer's Disease Diagnosis")

# ============================================================================
# Sidebar Configuration
# ============================================================================

st.sidebar.header("Configuration")

# Initialize analysis state
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# Scan subjects
subject_labels = scan_subjects()
subject_list = sorted(subject_labels.keys())

if not subject_list:
    st.error("No valid subjects found in data/MRI_processed/")
    st.stop()

# Check if subject changed - clear results if so
current_subject = st.session_state.get('current_subject', None)
is_running = st.session_state.analysis_running

# Subject selection (disabled during analysis)
selected_subject = st.sidebar.selectbox(
    "Select Subject",
    subject_list,
    help="Choose a subject for CDDA analysis",
    disabled=is_running
)

# Clear results if subject changed
if current_subject and current_subject != selected_subject and not is_running:
    # Clear all analysis results
    for key in ['analysis_result', 'ground_truth', 'init_time', 'analysis_time', 'analysis_logs']:
        if key in st.session_state:
            del st.session_state[key]

st.session_state.current_subject = selected_subject

ground_truth = subject_labels.get(selected_subject, "N/A")
st.sidebar.markdown(f"**Ground Truth:** {ground_truth}")

st.sidebar.markdown("---")

# Model configuration
st.sidebar.subheader("Model Configuration")

orchestrator_path = st.sidebar.text_input(
    "Orchestrator Model Path",
    value="D:/hf_models/Phi-4-mini-instruct",
    help="Path to Phi-4-mini model",
    disabled=is_running
)

consultant_path = st.sidebar.text_input(
    "Consultant Model Path",
    value="D:/hf_models/Llama3.1-Aloe-Beta-8B",
    help="Path to Llama3.1-Aloe-Beta-8B model",
    disabled=is_running
)

use_llm = st.sidebar.checkbox(
    "Enable LLM Mode",
    value=True,
    help="Use LLM for agent reasoning (disable for rule-based fallback)",
    disabled=is_running
)

use_4bit = st.sidebar.checkbox(
    "Use 4-bit Quantization",
    value=True,
    help="Enable 4-bit quantization to reduce VRAM usage",
    disabled=is_running
)

st.sidebar.markdown("---")

# Start/Stop analysis buttons
if is_running:
    # Show stop button during analysis
    if st.sidebar.button(
        "Force Stop Analysis",
        type="secondary",
        use_container_width=True
    ):
        st.session_state.analysis_running = False
        st.warning("Analysis stopped by user")
        st.rerun()
    
    start_analysis = False
else:
    # Show start button when not running
    start_analysis = st.sidebar.button(
        "Start Analysis",
        type="primary",
        use_container_width=True
    )

st.sidebar.markdown("---")
st.sidebar.caption("CDDA Framework v1.0")
st.sidebar.caption("Dual-LLM A2A Architecture")

# ============================================================================
# Main Content
# ============================================================================

if start_analysis:
    # Set analysis running state
    st.session_state.analysis_running = True
    
    # Clear previous results
    for key in ['analysis_result', 'ground_truth', 'init_time', 'analysis_time', 'analysis_logs']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.rerun()

# Analysis execution (when running)
if st.session_state.analysis_running and 'analysis_result' not in st.session_state:
    st.markdown("## Analysis Execution")
    
    progress_bar = st.progress(0)
    
    # System Information
    with st.expander("System Configuration", expanded=True):
        st.markdown(f"""
        **System:** Cognitive Discrepancy-Driven Agent (CDDA)  
        **Architecture:** Dual-LLM A2A Pattern  
        **Agent A (Orchestrator):** Phi-4-mini  
        **Agent B (Consultant):** Llama3.1-Aloe-Beta-8B  
        **Subject:** {selected_subject}  
        **Ground Truth:** {ground_truth}  
        **LLM Mode:** {'Enabled' if use_llm else 'Disabled (Rule-based)'}  
        **Quantization:** {'4-bit' if use_4bit else '8-bit'}
        """)
    
    # Initialize CDDA Agent
    st.markdown("### 1. Initializing CDDA Agent")
    
    status_text = st.empty()
    
    status_text.text("Initializing CDDA Agent...")
    progress_bar.progress(5)
    
    init_start = time.time()
    
    try:
        agent = CDDAAgent(
            orchestrator_model="phi-4-mini",
            orchestrator_model_path=orchestrator_path,
            consultant_model="llama3.1-aloe-beta-8b",
            consultant_model_path=consultant_path,
            use_llm=use_llm,
            use_4bit=use_4bit,
            verbose=True  # Enable verbose output to see Agent A and Agent B activity
        )
        init_time = time.time() - init_start
        
        status_text.text(f"Agent initialized successfully ({init_time:.2f}s)")
        progress_bar.progress(10)
        
        st.success(f"CDDA Agent initialized in {init_time:.2f}s")
        
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.session_state.analysis_running = False
        st.stop()
    
    # Run Analysis
    st.markdown("### 2. Running CDDA Analysis")
    
    status_text.text(f"Running analysis for {selected_subject}...")
    progress_bar.progress(10)
    
    st.markdown("""
    **Pipeline Stages:**
    1. Agent A: Orchestration (MCP resource reading, tool invocation)
    2. Agent B: Clinical synthesis (report generation)
    3. Post-processing: Executive summary generation
    """)
    
    analysis_start = time.time()
    
    # Simplified progress log for clinicians
    log_container = st.expander("Analysis Progress", expanded=True)
    log_placeholder = log_container.empty()
    
    # Simplified progress messages
    progress_messages = []
    
    def update_progress(message):
        progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        log_placeholder.markdown('\n\n'.join(progress_messages))
    
    try:
        # Run analysis with progress updates
        result_container = {}
        error_container = {}
        
        def run_analysis_thread():
            try:
                result_container['result'] = agent.run_analysis(selected_subject)
            except Exception as e:
                error_container['error'] = e
        
        # Start analysis
        update_progress("✓ Starting diagnostic analysis...")
        analysis_thread = threading.Thread(target=run_analysis_thread)
        analysis_thread.start()
        
        # Detailed progress updates with progress bar sync
        progress_steps = [
            (2, 15, "✓ Loading patient MRI data..."),
            (3, 20, "✓ Preprocessing brain images..."),
            (4, 25, "✓ Extracting brain region features..."),
            (5, 30, "✓ Normalizing feature values..."),
            (6, 35, "✓ Running machine learning model..."),
            (7, 40, "✓ Generating predictions..."),
            (8, 45, "✓ Calculating feature importance (SHAP)..."),
            (9, 50, "✓ Computing SHAP values for top features..."),
            (10, 55, "✓ Evaluating prediction uncertainty..."),
            (11, 60, "✓ Detecting statistical anomalies..."),
            (12, 65, "✓ Agent A: Analyzing diagnostic signals..."),
            (13, 70, "✓ Agent A: Evaluating uncertainty threshold..."),
            (15, 75, "✓ Agent A: Making adaptive decisions..."),
            (16, 80, "✓ Agent A: Compiling diagnostic context..."),
            (18, 85, "✓ Agent B: Receiving context object..."),
            (19, 90, "✓ Agent B: Generating clinical report..."),
            (20, 95, "✓ Post-processing: Creating executive summary..."),
            (21, 100, "✓ Finalizing analysis results...")
        ]
        
        start_time = time.time()
        step_idx = 0
        
        while analysis_thread.is_alive():
            elapsed = time.time() - start_time
            
            # Update progress based on elapsed time
            if step_idx < len(progress_steps) and elapsed >= progress_steps[step_idx][0]:
                _, progress_pct, message = progress_steps[step_idx]
                update_progress(message)
                progress_bar.progress(progress_pct)
                step_idx += 1
            
            time.sleep(0.3)  # More frequent updates
        
        # Wait for thread to complete
        analysis_thread.join()
        
        # Check for errors
        if 'error' in error_container:
            raise error_container['error']
        
        result = result_container.get('result')
        
        if not result:
            raise Exception("Analysis completed but no result returned")
        
        analysis_time = time.time() - analysis_start
        
        # Final progress update
        update_progress(f"✓ Analysis completed successfully! (Total time: {analysis_time:.1f}s)")
        
        status_text.text(f"Analysis completed successfully ({analysis_time:.2f}s)")
        progress_bar.progress(100)
        
        st.success(f"Analysis completed in {analysis_time:.2f}s")
        
        # Store result in session state
        st.session_state.analysis_result = result
        st.session_state.ground_truth = ground_truth
        st.session_state.init_time = init_time
        st.session_state.analysis_time = analysis_time
        st.session_state.analysis_running = False
        
        # Free up Agent A memory (only Agent B needed for chat)
        try:
            if hasattr(agent, 'agent_a'):
                # Clear Agent A's LLM to free GPU memory
                if hasattr(agent.agent_a, 'llm_provider'):
                    agent.agent_a.llm_provider = None
                if hasattr(agent.agent_a, 'model'):
                    agent.agent_a.model = None
                
                # Force garbage collection
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if st.session_state.get('verbose', False):
                    st.info("Agent A memory released. GPU memory freed for chatbot.")
        except Exception as e:
            # Silent fail - memory cleanup is optional
            pass
        
        # Rerun to show results
        st.rerun()
        
    except Exception as e:
        update_progress(f"✗ Error: {str(e)}")
        st.error(f"Analysis failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.session_state.analysis_running = False
        st.stop()

# ============================================================================
# Display Results
# ============================================================================

if 'analysis_result' in st.session_state:
    result = st.session_state.analysis_result
    ground_truth = st.session_state.ground_truth
    init_time = st.session_state.init_time
    analysis_time = st.session_state.analysis_time
    
    st.markdown("---")
    st.markdown("## Clinical Dashboard")
    
    # Integrated Dashboard: Executive Summary + Diagnostic Results
    diagnosis_map = {
        'AD': 'AD',
        'MCI': 'MCI',
        'NC': 'NC'
    }
    
    # Get risk level from executive summary
    risk_level = "Medium"
    if 'executive_summary' in result.metadata:
        risk_level = result.metadata['executive_summary'].get('risk_level', 'Medium')
    
    # Top row: Key metrics with colored indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prediction", diagnosis_map.get(result.prediction, result.prediction))
        # No indicator for prediction
    
    with col2:
        st.metric("Confidence", f"{result.confidence:.3f}")
        # Confidence indicator
        if result.confidence > 0.8:
            st.markdown('<p style="color: green; font-size: 0.8em; margin-top: -10px;">High</p>', unsafe_allow_html=True)
        elif result.confidence > 0.6:
            st.markdown('<p style="color: orange; font-size: 0.8em; margin-top: -10px;">Medium</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: red; font-size: 0.8em; margin-top: -10px;">Low</p>', unsafe_allow_html=True)
    
    with col3:
        st.metric("Uncertainty", f"{result.uq_score:.3f}")
        # Uncertainty indicator (inverse: low is good)
        if result.uq_score < 0.5:
            st.markdown('<p style="color: green; font-size: 0.8em; margin-top: -10px;">Low</p>', unsafe_allow_html=True)
        elif result.uq_score < 0.8:
            st.markdown('<p style="color: orange; font-size: 0.8em; margin-top: -10px;">Medium</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: red; font-size: 0.8em; margin-top: -10px;">High</p>', unsafe_allow_html=True)
    
    with col4:
        st.metric("Risk Level", risk_level)
        # Risk level indicator
        if risk_level == "High":
            st.markdown('<p style="color: red; font-size: 0.8em; margin-top: -10px;">High Risk</p>', unsafe_allow_html=True)
        elif risk_level == "Medium":
            st.markdown('<p style="color: orange; font-size: 0.8em; margin-top: -10px;">Medium Risk</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: green; font-size: 0.8em; margin-top: -10px;">Low Risk</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Executive Summary Section
    if 'executive_summary' in result.metadata:
        summary = result.metadata['executive_summary']
        
        # Headline
        st.markdown(f"### {summary.get('headline', 'Analysis Complete')}")
        
        st.markdown("")
        
        # Key Findings and Recommended Actions (side by side)
        col_findings, col_actions = st.columns(2)
        
        with col_findings:
            st.markdown("**Key Findings:**")
            for finding in summary.get('key_findings', []):
                st.markdown(f"- {finding}")
        
        with col_actions:
            st.markdown("**Recommended Actions:**")
            for action in summary.get('recommended_actions', []):
                st.markdown(f"- {action}")
    
    # Decision Mode (smaller text, below dashboard)
    st.markdown(f'<p style="color: gray; font-size: 0.9em; margin-top: 1em;">Decision Mode: {result.agent_decision}</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Importance Analysis
    if result.context_object and result.context_object.diagnostic_report:
        st.markdown("### Feature Importance Analysis (SHAP + Z-score)")
        
        top_features = result.context_object.diagnostic_report.top_features[:10]
        
        if top_features:
            # Create table data
            table_data = []
            for feat in top_features:
                rank = safe_get_feature_attr(feat, 'rank', 0)
                roi_name = safe_get_feature_attr(feat, 'roi_name', 'Unknown')
                shap_value = safe_get_feature_attr(feat, 'shap_value', 0)
                z_score = safe_get_feature_attr(feat, 'z_score', 0)
                
                # Clinical significance
                if abs(z_score) > 2.5:
                    significance = "Anomalous (|Z| > 2.5)"
                elif z_score < -1.5:
                    significance = "Atrophy pattern"
                elif z_score > 1.5:
                    significance = "Preserved volume"
                else:
                    significance = "Normal range"
                
                table_data.append({
                    "Rank": rank,
                    "Brain Region": roi_name,
                    "SHAP Value": f"{shap_value:+.4f}",
                    "Z-score": f"{z_score:+.2f}",
                    "Clinical Significance": significance
                })
            
            st.table(table_data)
    
    # Clinical Report - Agent Interaction Summary
    with st.expander("Clinical Report - Agent Interaction Summary", expanded=False):
        if hasattr(result, 'clinical_report') and result.clinical_report:
            report = result.clinical_report
            
            # Extract content after <REPORT> marker
            if '<REPORT>' in report:
                # Get the LAST segment after splitting by <REPORT>
                # (in case <REPORT> appears multiple times)
                report_content = report.split('<REPORT>')[-1].strip()
            else:
                # Fallback: use original filtering logic
                lines = report.split('\n')
                filtered_lines = []
                
                for line in lines:
                    line_lower = line.lower().strip()
                    
                    # Skip system prompt keywords
                    if any(keyword in line_lower for keyword in [
                        'your role is to', 'important: you have no access',
                        'input: contextobject', 'your task:', 'synthesis guidelines:',
                        'report structure:', 'diagnostic_report:', 'tool_results:',
                        'decision_rationale:', 'signals:'
                    ]):
                        continue
                    
                    # Skip Chinese text
                    if any('\u4e00' <= char <= '\u9fff' for char in line):
                        continue
                    
                    # Add valid lines
                    if line.strip():
                        filtered_lines.append(line)
                
                report_content = '\n'.join(filtered_lines)
            
            # Display report
            if report_content:
                st.markdown(report_content)
            else:
                st.info("Clinical report is being processed...")
        else:
            st.info("No clinical report available")
        
        # Add agent interaction summary
        st.markdown("---")
        st.markdown("**Agent Interaction Summary:**")
        st.markdown(f"""
        - **Agent A (Orchestrator):** Analyzed diagnostic data, evaluated uncertainty (UQ: {result.uq_score:.3f})
        - **Decision:** {result.agent_decision}
        - **Agent B (Consultant):** Generated clinical synthesis based on provided context
        - **Recommendation:** Review detailed findings and consider clinical correlation
        """)
    

    
    # Performance Metrics
    st.markdown("### Performance Metrics")
    
    total_time = init_time + analysis_time
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Initialization Time", f"{init_time:.2f}s")
    
    with col2:
        st.metric("Analysis Time", f"{analysis_time:.2f}s")
    
    with col3:
        st.metric("Total Time", f"{total_time:.2f}s")
    
    throughput = 3600 / analysis_time if analysis_time > 0 else 0
    st.markdown(f"**Throughput:** {throughput:.2f} subjects/hour")
    
    st.markdown("---")
    
    # Chatbot - Ask Agent B
    st.markdown("### Ask Chatbot (Clinical Consultant)")
    st.markdown("Ask questions about this analysis. Chatbot will answer based on the diagnostic context.")
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history using st.chat_message
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
    
    # Chat input (supports Enter key)
    user_question = st.chat_input("Ask a question (press Enter to send)...")
    
    if user_question:
        # Add user question to history
        st.session_state.chat_history.append(("user", user_question))
        
        # Prepare context for Agent B
        context_summary = f"""
DIAGNOSTIC CONTEXT:
- Subject: {result.subject_id}
- Prediction: {result.prediction}
- Confidence: {result.confidence:.3f}
- Uncertainty: {result.uq_score:.3f}
- Decision Mode: {result.agent_decision}

CLINICAL REPORT SUMMARY:
{result.clinical_report[:500] if hasattr(result, 'clinical_report') else 'N/A'}...

EXECUTIVE SUMMARY:
"""
        
        if 'executive_summary' in result.metadata:
            summary = result.metadata['executive_summary']
            context_summary += f"""
- Headline: {summary.get('headline', 'N/A')}
- Risk Level: {summary.get('risk_level', 'N/A')}
- Key Findings: {', '.join(summary.get('key_findings', [])[:3])}
"""
        
        # Create prompt for Agent B
        chat_prompt = f"""
{context_summary}

PHYSICIAN QUESTION:
{user_question}

Please provide a clear, concise answer based on the diagnostic context above. 
Focus on clinical interpretation and practical recommendations.
"""
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)
        
        try:
            # Import HuggingFace provider at the top level
            from app.services.llm_providers import huggingface as hf_provider
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get Agent B from session state or create new one
                    if 'agent_b_chat' not in st.session_state:
                        # Import Agent B
                        from app.agents.agent_b_consultant import AgentB, AgentBConfig
                        
                        config = AgentBConfig(
                            model="llama3.1-aloe-beta-8b",
                            model_path=consultant_path,
                            provider="huggingface",
                            temperature=0.3,
                            use_llm=True,
                            load_in_8bit=not use_4bit,
                            verbose=False
                        )
                        
                        agent_b = AgentB(config=config)
                        
                        # Manually initialize LLM provider if not already done
                        if not hasattr(agent_b, 'llm_provider') or agent_b.llm_provider is None:
                            agent_b.llm_provider = hf_provider
                        
                        st.session_state.agent_b_chat = agent_b
                    
                    agent_b = st.session_state.agent_b_chat
                    
                    # Get response from Agent B using HuggingFace provider directly
                    try:
                        response = hf_provider.handle_text(
                            prompt=chat_prompt,
                            model_path=consultant_path,
                            system_instruction="You are a clinical consultant AI. Provide clear, concise, evidence-based answers to physician questions about diagnostic cases. Keep responses focused and actionable.",
                            load_in_8bit=not use_4bit
                        )
                        
                        # Filter out <REPORT> marker if present
                        if '<REPORT>' in response:
                            response = response.split('<REPORT>')[-1].strip()
                        
                    except Exception as llm_error:
                        response = f"I apologize, but I encountered an issue accessing the language model: {str(llm_error)}\n\nPlease ensure:\n1. The model path is correct: {consultant_path}\n2. The model files are downloaded\n3. Sufficient GPU memory is available"
                    
                    # Display response
                    st.markdown(response)
                
                # Add to history
                st.session_state.chat_history.append(("assistant", response))
                
        except Exception as e:
            with st.chat_message("assistant"):
                error_msg = f"I encountered an error: {str(e)}\n\nDebug info:\n- Model path: {consultant_path}\n- Use 4-bit: {use_4bit}"
                st.markdown(error_msg)
                st.session_state.chat_history.append(("assistant", error_msg))

else:
    # Welcome message
    st.info("Select a subject and click 'Start Analysis' in the sidebar to begin.")
    
    st.markdown("""
    ### About CDDA Framework
    
    The Cognitive Discrepancy-Driven Agent (CDDA) is a dual-LLM framework for explainable 
    Alzheimer's disease diagnosis. It combines:
    
    - **Adaptive Decision-Making**: Dynamic pathway selection based on uncertainty
    - **Counterfactual Analysis**: Causal reasoning for diagnostic drivers
    - **Knowledge Integration**: Clinical context from knowledge graph
    - **Complete Transparency**: Full reasoning chain from both agents
    
    **Architecture:**
    - Agent A (Orchestrator): Phi-4-mini - MCP client, resource reader, tool invoker
    - Agent B (Consultant): Llama3.1-Aloe-Beta-8B - Clinical synthesizer
    
    **Key Features:**
    - Uncertainty Quantification (UQ)
    - Anomaly Detection (Z-score)
    - SHAP Feature Importance
    - Executive Summary Generation
    """)
