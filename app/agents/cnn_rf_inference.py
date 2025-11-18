"""
CNN-RF Inference Agent

This agent performs ML-based inference on structural MRI scans
using the CNN-RF Random Forest model trained on AAL3 ROI features.

Features:
- End-to-end inference from raw MRI images
- Multi-modal support (GM, FA, MD)
- AAL3 atlas-based ROI extraction
- Feature importance analysis
- Brain region visualization
"""

import numpy as np
from typing import Dict, Optional
from pathlib import Path
from app.graph.state import AgentState

# Import end-to-end predictor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.cnn_rf.end_to_end_inference import EndToEndPredictor
from scripts.cnn_rf.config import MODELS, DEFAULT_MODEL


def run_cnn_rf_inference(state: AgentState) -> dict:
    """
    Execute end-to-end CNN-RF inference from raw MRI images
    
    This agent:
    1. Locates subject's MRI images in data/MRI_processed
    2. Extracts ROI features from raw images (GM, FA, MD)
    3. Loads CNN-RF Random Forest model
    4. Performs prediction with confidence scores
    5. Analyzes feature importances
    
    Args:
        state: AgentState containing:
            - subject_id: Subject identifier (e.g., 'sub-0005')
            - model_name: Optional model name ('NC_vs_AD' or 'NC_MCI_AD')
            - data_root: Optional data root directory (default: 'data/MRI_processed')
    
    Returns:
        Updated state dict with:
        - classification_result: Predicted class (NC, MCI, or AD)
        - prediction_confidence: Confidence score (0-1)
        - prediction_probabilities: Dict of class probabilities
        - true_label: Ground truth label from directory structure
        - correct_prediction: Boolean indicating if prediction matches ground truth
        - roi_features: Dict of extracted ROI feature values
        - subject_directory: Path to subject's MRI data
        - trace_log: Updated with processing steps
        - error_log: Updated if errors occur
    """
    print("\n" + "="*80)
    print("AGENT: CNN-RF End-to-End Inference")
    print("="*80)
    
    subject_id = state.get('subject_id', 'unknown')
    model_name = state.get('model_name', DEFAULT_MODEL)
    data_root = state.get('data_root', 'data/MRI_processed')
    
    print(f"\n📊 Subject: {subject_id}")
    print(f"🤖 Model: {model_name}")
    print(f"📁 Data Root: {data_root}")
    
    try:
        # Step 1: Initialize end-to-end predictor
        print(f"\n[1/3] Initializing end-to-end predictor...")
        try:
            model_config = MODELS.get(model_name)
            if not model_config:
                raise ValueError(f"Unknown model: {model_name}")
            
            model_path = model_config['path']
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            predictor = EndToEndPredictor(
                model_path=str(model_path),
                data_root=data_root
            )
            print(f"   ✓ Predictor initialized")
            print(f"   ✓ Model: {model_path.name}")
            print(f"   ✓ Classes: {predictor.classes}")
            
        except Exception as e:
            error_msg = f"Predictor initialization failed: {e}"
            print(f"   ❌ {error_msg}")
            return {
                "error_log": state.get("error_log", []) + [error_msg],
                "classification_result": "ERROR: Predictor unavailable"
            }
        
        # Step 2: Perform end-to-end prediction
        print(f"\n[2/3] Performing end-to-end prediction from raw MRI images...")
        try:
            results = predictor.predict_subject(subject_id, verbose=True)
            
            prediction = results['predicted_label']
            confidence = results['confidence']
            probabilities = results['probabilities']
            true_label = results['true_label']
            correct = results['correct']
            roi_features = results['features']
            subject_dir = results['subject_dir']
            
            print(f"\n   🎯 Prediction Results:")
            print(f"      Classification: {prediction}")
            print(f"      Confidence: {confidence:.1%}")
            print(f"      Ground Truth: {true_label}")
            print(f"      Status: {'✓ CORRECT' if correct else '✗ INCORRECT'}")
            print(f"      Probabilities:")
            for cls, prob in probabilities.items():
                print(f"         {cls}: {prob:.1%}")
            
        except Exception as e:
            error_msg = f"End-to-end prediction failed: {e}"
            print(f"   ❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "error_log": state.get("error_log", []) + [error_msg],
                "classification_result": "ERROR: Prediction failed"
            }
        
        # Step 3: Extract SHAP features (local explainability)
        print(f"\n[3/3] Analyzing local feature importance (SHAP)...")
        try:
            shap_features = results.get('shap_features', [])
            
            if shap_features:
                print(f"   ✓ SHAP analysis complete")
                print(f"\n   🎯 Top 5 Features for This Subject (SHAP):")
                for i, feat in enumerate(shap_features[:5], 1):
                    direction_symbol = "→" if feat['direction'] == 'towards AD' else "←"
                    print(f"      {i}. {feat['name']}")
                    print(f"         SHAP: {feat['shap_value']:+.4f} {direction_symbol} {feat['direction']}")
                
                # Create feature importances dict for compatibility
                feature_importances_dict = {
                    feat['name']: feat['abs_shap_value'] 
                    for feat in shap_features
                }
            else:
                print(f"   ⚠️  SHAP not available, using global feature importances")
                # Fallback to global importances
                if hasattr(predictor.model, 'named_steps'):
                    rf_model = predictor.model.named_steps['model']
                    feature_importances = rf_model.feature_importances_
                    top_indices = np.argsort(feature_importances)[-5:][::-1]
                    
                    print(f"\n   📊 Top 5 Important Features (Global):")
                    for i, idx in enumerate(top_indices, 1):
                        print(f"      {i}. Feature {idx}: {feature_importances[idx]:.4f}")
                    
                    feature_importances_dict = {
                        int(idx): float(feature_importances[idx]) 
                        for idx in top_indices
                    }
                else:
                    feature_importances_dict = {}
            
        except Exception as e:
            error_msg = f"Feature importance extraction failed: {e}"
            print(f"   ⚠️  {error_msg}")
            feature_importances_dict = {}
            shap_features = []
        
        # Step 4: Prepare return state
        trace_msg = (
            f"End-to-end CNN-RF inference complete for {subject_id}: "
            f"{prediction} (confidence: {confidence:.1%}, ground truth: {true_label})"
        )
        
        print(f"\n✅ {trace_msg}")
        print("="*80 + "\n")
        
        return {
            "classification_result": prediction,
            "prediction_confidence": float(confidence),
            "prediction_probabilities": probabilities,
            "true_label": true_label,
            "correct_prediction": correct,
            "roi_features": roi_features,
            "feature_importances": feature_importances_dict,
            "shap_features": shap_features,
            "subject_directory": subject_dir,
            "model_name": model_name,
            "trace_log": state.get("trace_log", []) + [trace_msg]
        }
        
    except Exception as e:
        # Catch-all for unexpected errors
        error_msg = f"Unexpected error in CNN-RF inference: {type(e).__name__}: {e}"
        print(f"❌ {error_msg}")
        print("="*80 + "\n")
        
        import traceback
        traceback.print_exc()
        
        return {
            "error_log": state.get("error_log", []) + [error_msg],
            "classification_result": "ERROR"
        }


def run_cnn_rf_inference_with_visualization(state: AgentState) -> dict:
    """
    Extended version that also generates brain region visualization
    
    This version includes all features of run_cnn_rf_inference plus:
    - Generates 3D brain map of important regions
    - Saves visualization to output directory
    
    Args:
        state: AgentState (same as run_cnn_rf_inference)
    
    Returns:
        Updated state dict (same as run_cnn_rf_inference) plus:
        - brain_map_path: Path to generated brain visualization
    """
    # First run standard inference
    result = run_cnn_rf_inference(state)
    
    # If inference failed, return early
    if "ERROR" in result.get("classification_result", ""):
        return result
    
    # Generate brain visualization
    subject_id = state.get('subject_id', 'unknown')
    
    try:
        print(f"\n[VISUALIZATION] Generating brain map...")
        
        from scripts.cnn_rf.inference import CNNRF_Predictor
        from scripts.cnn_rf.config import MODELS, DEFAULT_MODEL
        
        model_name = state.get('model_name', DEFAULT_MODEL)
        model_config = MODELS[model_name]
        
        predictor = CNNRF_Predictor(model_path=str(model_config['path']))
        
        # Get top important ROIs from feature importances
        feature_importances = result.get('feature_importances', {})
        if feature_importances:
            important_rois = predictor.extract_important_rois(top_n=10)
            
            output_path = f"output/cnn_rf/{subject_id}_brain_map.nii.gz"
            brain_map_path = predictor.create_brain_map(
                important_rois=important_rois,
                output_path=output_path
            )
            
            print(f"   ✓ Brain map saved: {brain_map_path}")
            
            result['brain_map_path'] = brain_map_path
            result['trace_log'] = result.get('trace_log', []) + [
                f"Brain visualization generated: {brain_map_path}"
            ]
        else:
            print("   ⚠️  No feature importances available for visualization")
        
    except Exception as e:
        error_msg = f"Brain visualization failed: {e}"
        print(f"   ⚠️  {error_msg}")
        # Non-critical error, don't fail the whole inference
        result['error_log'] = result.get('error_log', []) + [error_msg]
    
    return result
