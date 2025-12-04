"""
CNN-RF Inference Agent (LOOCV-Aware Version)

This agent performs ML-based inference on structural MRI scans.
It dynamically loads subject-specific LOOCV models to ensure 
strict separation between training and testing data.
"""

import numpy as np
import os
from typing import Dict, Optional
from pathlib import Path
from app.graph.state import AgentState

# Import end-to-end predictor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.core.cnn_rf.end_to_end_inference import EndToEndPredictor
from app.core.cnn_rf.config import MODELS, DEFAULT_MODEL

# [設定] LOOCV 模型存放的資料夾
LOOCV_MODEL_DIR = Path("model/loocv_models_binary_opt")
# [設定] 通用二分類模型路徑 (給 MCI/OOD 使用)
GENERAL_MODEL_PATH = Path("model/cnn_rf/rf_model_NC_vs_AD.joblib")

def get_model_path_for_subject(subject_id: str, default_model_name: str) -> Path:
    """
    Helper function to resolve the correct model path.
    
    Strategy:
    1. NC/AD Subjects: Use specific LOOCV binary model (Strict Separation).
    2. MCI Subjects: Use General Binary Model (OOD/Uncertainty Testing).
    """
    # 1. 嘗試尋找 LOOCV 專屬模型 (針對 NC/AD)
    specific_model_name = f"rf_model_{subject_id}.joblib"
    specific_model_path = LOOCV_MODEL_DIR / specific_model_name
    
    if specific_model_path.exists():
        print(f"   [INFO] Strategy: LOOCV Binary (Target Subject: {subject_id})")
        print(f"   [PATH] {specific_model_path}")
        return specific_model_path
    
    # 2. 如果找不到 (代表他是 MCI，或是資料集外的新病人)
    # [修正 2] 強制使用通用二分類模型，觀察 Agent 的不確定性反應
    if GENERAL_MODEL_PATH.exists():
        print(f"   [WARN] LOOCV model not found for {subject_id} (likely MCI/OOD).")
        print(f"   [INFO] Strategy: General Binary Model (OOD Testing)")
        return GENERAL_MODEL_PATH
        
    # 3. 如果連通用模型都沒有，才退回到 Config 設定 (最後防線)
    print(f"   [WARN] General model not found. Falling back to config: {default_model_name}")
    model_config = MODELS.get(default_model_name)
    if not model_config:
        raise ValueError(f"Unknown model config: {default_model_name}")
    return model_config['path']

def run_cnn_rf_inference(state: AgentState) -> dict:
    """
    Execute end-to-end CNN-RF inference with LOOCV support.
    Returns a complete diagnostic report compatible with DiagnosticReport.from_toolkit_report()
    
    This function now uses CDDAToolKit to generate a complete diagnostic report
    including UQ scoring and anomaly detection.
    """
    print("\n" + "="*80)
    print("AGENT: CNN-RF End-to-End Inference (LOOCV-Enabled)")
    print("="*80)
    
    subject_id = state.get('subject_id', 'unknown')
    model_name = state.get('model_name', DEFAULT_MODEL)
    data_root = state.get('data_root', 'data/MRI_processed')
    
    print(f"\n[Subject] {subject_id}")
    print(f"[Model Config] {model_name}")
    print(f"[Data Root] {data_root}")
    
    try:
        # Determine the correct model path (LOOCV-specific or global fallback)
        model_path = get_model_path_for_subject(subject_id, model_name)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing: {model_path}")
        
        # Initialize CDDAToolKit with the correct model
        from app.core.ml_processing.cdda_tools import CDDAToolKit
        
        print(f"\n[1/2] Initializing CDDAToolKit with LOOCV model...")
        toolkit = CDDAToolKit(
            model_path=str(model_path),
            data_root=data_root
        )
        
        # Get complete diagnostic report (includes UQ, anomaly detection, etc.)
        print(f"\n[2/2] Generating complete diagnostic report...")
        report = toolkit.get_diagnostic_report(subject_id, verbose=True)
        
        # Add trace log for model verification
        trace_msg = (
            f"Inference complete for {subject_id} using {model_path.name}: "
            f"{report['prediction_result']} ({report['confidence']:.1%})"
        )
        
        report['trace_log'] = state.get("trace_log", []) + [trace_msg]
        report['model_name'] = model_path.name
        
        print(f"\n[SUCCESS] Complete diagnostic report generated")
        print("="*80 + "\n")
        
        return report
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {
            "error_log": state.get("error_log", []) + [error_msg],
            "classification_result": "ERROR"
        }


def run_cnn_rf_inference_with_visualization(state: AgentState) -> dict:
    """
    Extended version that also generates brain region visualization.
    Ensures visualization uses the SAME LOOCV model as the inference.
    """
    # 1. Run standard inference
    result = run_cnn_rf_inference(state)
    
    if "ERROR" in result.get("classification_result", ""):
        return result
    
    subject_id = state.get('subject_id', 'unknown')
    model_name = state.get('model_name', DEFAULT_MODEL)
    
    try:
        print(f"\n[VISUALIZATION] Generating brain map...")
        
        from scripts.cnn_rf.inference import CNNRF_Predictor
        
        # [!! 修改點 !!] Visualization 也要用同樣的邏輯取模型
        # 否則畫出來的 Feature Importance 會跟預測用的模型不一致
        model_path = get_model_path_for_subject(subject_id, model_name)
        
        predictor = CNNRF_Predictor(model_path=str(model_path))
        
        # Get top important ROIs
        feature_importances = result.get('feature_importances', {})
        if feature_importances:
            important_rois = predictor.extract_important_rois(top_n=10)
            
            output_path = f"output/cnn_rf/{subject_id}_brain_map.nii.gz"
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            brain_map_path = predictor.create_brain_map(
                important_rois=important_rois,
                output_path=output_path
            )
            
            print(f"   [OK] Brain map saved: {brain_map_path}")
            result['brain_map_path'] = brain_map_path
        else:
            print("   [WARN] No feature importances available for visualization")
        
    except Exception as e:
        error_msg = f"Brain visualization failed: {e}"
        print(f"   [WARN] {error_msg}")
        result['error_log'] = result.get('error_log', []) + [error_msg]
    
    return result