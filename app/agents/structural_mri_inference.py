"""
Structural MRI Inference Agent

This agent performs ML-based inference on structural T1 MRI scans
using a Random Forest model trained on 32 AAL ROI features.
"""

import numpy as np
from typing import Dict
from app.graph.state import AgentState
from app.core.ml_processing import (
    MLModelLoader,
    ROIFeatureExtractor,
    MLModelConfig,
    ModelLoadError,
    FeatureExtractionError,
    PredictionError
)


def run_structural_mri_inference(state: AgentState) -> dict:
    """
    Execute structural MRI inference using ML model
    
    This agent:
    1. Loads the Random Forest model and components
    2. Extracts 32 ROI features from T1 MRI
    3. Standardizes features using the trained scaler
    4. Performs prediction and calculates confidence
    5. Extracts feature importances
    
    Args:
        state: AgentState containing fmri_scan_path (actually T1 MRI path)
    
    Returns:
        Updated state dict with:
        - classification_result: "NC" or "AD"
        - prediction_confidence: float (0-1)
        - roi_features: Dict[str, float]
        - feature_importances: Dict[str, float]
        - trace_log: Updated with processing steps
        - error_log: Updated if errors occur
    """
    print("\n" + "="*60)
    print("AGENT: Structural MRI Inference")
    print("="*60)
    
    subject_id = state.get('subject_id', 'unknown')
    mri_path = state.get('fmri_scan_path')  # Note: variable name kept for compatibility
    
    if not mri_path:
        error_msg = "Missing MRI scan path in state"
        print(f"❌ ERROR: {error_msg}")
        return {
            "error_log": state.get("error_log", []) + [error_msg],
            "classification_result": "ERROR"
        }
    
    try:
        # Step 1: Load model components
        print(f"\n📊 Subject: {subject_id}")
        print(f"📁 MRI Path: {mri_path}")
        
        try:
            config = MLModelConfig.from_directory()
            loader = MLModelLoader(config)
            components = loader.get_all_components()
            
            model = components['model']
            scaler = components['scaler']
            roi_list = components['roi_list']
            feature_names = components['feature_names']
            
        except ModelLoadError as e:
            error_msg = f"Model loading failed: {e}"
            print(f"❌ {error_msg}")
            return {
                "error_log": state.get("error_log", []) + [error_msg],
                "classification_result": "ERROR: Model unavailable"
            }
        
        # Step 2: Extract ROI features
        try:
            extractor = ROIFeatureExtractor()
            features = extractor.extract_features(mri_path, roi_list)
            
            # Create feature dictionary
            roi_features = dict(zip(feature_names, features))
            
            print(f"✓ Extracted {len(features)} ROI features")
            
        except (FeatureExtractionError, Exception) as e:
            error_msg = f"Feature extraction failed: {e}"
            print(f"❌ {error_msg}")
            return {
                "error_log": state.get("error_log", []) + [error_msg],
                "classification_result": "ERROR: Feature extraction failed"
            }
        
        # Step 3: Standardize features
        try:
            features_scaled = scaler.transform(features.reshape(1, -1))
            print(f"✓ Features standardized")
            
        except Exception as e:
            error_msg = f"Feature standardization failed: {e}"
            print(f"❌ {error_msg}")
            return {
                "error_log": state.get("error_log", []) + [error_msg],
                "classification_result": "ERROR: Standardization failed"
            }
        
        # Step 4: Perform prediction
        try:
            # Get prediction
            prediction = model.predict(features_scaled)[0]
            
            # Get prediction probabilities
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Map prediction to class name
            class_names = model.classes_
            prediction_label = class_names[prediction]
            
            # Get confidence (probability of predicted class)
            confidence = probabilities[prediction]
            
            print(f"\n🎯 Prediction Results:")
            print(f"   Classification: {prediction_label}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Probabilities: NC={probabilities[0]:.1%}, AD={probabilities[1]:.1%}")
            
        except Exception as e:
            error_msg = f"Prediction failed: {e}"
            print(f"❌ {error_msg}")
            return {
                "error_log": state.get("error_log", []) + [error_msg],
                "classification_result": "ERROR: Prediction failed"
            }
        
        # Step 5: Extract feature importances
        try:
            importances = model.feature_importances_
            feature_importances = dict(zip(feature_names, importances))
            
            # Sort by importance
            sorted_importances = sorted(
                feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            print(f"\n📊 Top 5 Important Features:")
            for i, (feature, importance) in enumerate(sorted_importances[:5], 1):
                print(f"   {i}. {feature}: {importance:.4f} ({importance*100:.2f}%)")
            
        except Exception as e:
            error_msg = f"Feature importance extraction failed: {e}"
            print(f"⚠️  {error_msg}")
            # Non-critical error, continue with empty importances
            feature_importances = {}
        
        # Step 6: Prepare return state
        trace_msg = (
            f"Structural MRI inference complete for {subject_id}: "
            f"{prediction_label} (confidence: {confidence:.1%})"
        )
        
        print(f"\n✅ {trace_msg}")
        print("="*60 + "\n")
        
        return {
            "classification_result": prediction_label,
            "prediction_confidence": float(confidence),
            "roi_features": roi_features,
            "feature_importances": feature_importances,
            "trace_log": state.get("trace_log", []) + [trace_msg]
        }
        
    except Exception as e:
        # Catch-all for unexpected errors
        error_msg = f"Unexpected error in structural MRI inference: {type(e).__name__}: {e}"
        print(f"❌ {error_msg}")
        print("="*60 + "\n")
        
        return {
            "error_log": state.get("error_log", []) + [error_msg],
            "classification_result": "ERROR"
        }
