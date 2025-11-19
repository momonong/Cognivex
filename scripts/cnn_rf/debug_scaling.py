"""
Debug Scaling Pipeline

This script checks if StandardScaler is properly saved and loaded
during training and inference.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_model_pipeline(model_path="model/cnn_rf/rf_model_NC_vs_AD.joblib"):
    """Check if model has proper scaling pipeline"""
    
    print("="*80)
    print("Scaling Pipeline Debug")
    print("="*80)
    
    print(f"\n[1/4] Loading model from {model_path}...")
    model = joblib.load(model_path)
    print(f"[OK] Model loaded")
    print(f"[OK] Model type: {type(model)}")
    
    # Check if it's a pipeline
    print(f"\n[2/4] Checking pipeline structure...")
    if hasattr(model, 'named_steps'):
        print(f"[OK] Model is a Pipeline")
        print(f"\nPipeline steps:")
        for step_name, step_obj in model.named_steps.items():
            print(f"  - {step_name}: {type(step_obj).__name__}")
        
        # Check for scaler (could be named 'scaler' or 'scale')
        scaler_step = None
        if 'scaler' in model.named_steps:
            scaler_step = model.named_steps['scaler']
        elif 'scale' in model.named_steps:
            scaler_step = model.named_steps['scale']
        
        if scaler_step is not None:
            print(f"\n✓ Scaler found: {type(scaler_step).__name__}")
            
            if hasattr(scaler_step, 'mean_'):
                print(f"  Mean shape: {scaler_step.mean_.shape}")
                print(f"  Mean sample: {scaler_step.mean_[:5]}")
                print(f"  Mean range: [{scaler_step.mean_.min():.4f}, {scaler_step.mean_.max():.4f}]")
            if hasattr(scaler_step, 'scale_'):
                print(f"  Scale shape: {scaler_step.scale_.shape}")
                print(f"  Scale sample: {scaler_step.scale_[:5]}")
                print(f"  Scale range: [{scaler_step.scale_.min():.4f}, {scaler_step.scale_.max():.4f}]")
        else:
            print(f"\n⚠️  WARNING: No scaler step found in pipeline!")
            print(f"  Available steps: {list(model.named_steps.keys())}")
    else:
        print(f"⚠️  WARNING: Model is NOT a Pipeline!")
        print(f"  This means no preprocessing is applied during inference!")
    
    # Check for feature selector
    print(f"\n[3/4] Checking feature selection...")
    if hasattr(model, 'named_steps') and 'select' in model.named_steps:
        selector = model.named_steps['select']
        print(f"✓ Feature selector found: {type(selector).__name__}")
        
        if hasattr(selector, 'get_support'):
            selected_mask = selector.get_support()
            n_selected = selected_mask.sum()
            n_total = len(selected_mask)
            print(f"  Selected features: {n_selected} / {n_total} ({n_selected/n_total*100:.1f}%)")
    else:
        print(f"⚠️  No feature selector found")
    
    # Check model
    print(f"\n[4/4] Checking final model...")
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        rf_model = model.named_steps['model']
        print(f"✓ RandomForest found: {type(rf_model).__name__}")
        print(f"  n_estimators: {rf_model.n_estimators}")
        print(f"  n_features_in_: {rf_model.n_features_in_}")
        print(f"  n_classes_: {rf_model.n_classes_}")
        print(f"  classes_: {rf_model.classes_}")
    
    return model


def test_scaling_consistency(model_path="model/cnn_rf/rf_model_NC_vs_AD.joblib",
                             csv_path="data/roi_features.csv"):
    """Test if scaling is consistent between training and inference"""
    
    print("\n" + "="*80)
    print("Scaling Consistency Test")
    print("="*80)
    
    # Load model
    print(f"\n[1/3] Loading model...")
    model = joblib.load(model_path)
    
    # Load data
    print(f"\n[2/3] Loading test data...")
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col not in ['Subject_ID', 'Group']]
    
    # Get one sample
    sample = df[feature_cols].iloc[0:1]
    print(f"[OK] Sample shape: {sample.shape}")
    print(f"\nRaw feature values (first 5):")
    for col in feature_cols[:5]:
        print(f"  {col}: {sample[col].values[0]:.6f}")
    
    # Test prediction
    print(f"\n[3/3] Testing prediction with scaling...")
    try:
        prediction = model.predict(sample)
        proba = model.predict_proba(sample)
        
        print(f"✓ Prediction successful")
        print(f"  Predicted class: {prediction[0]}")
        print(f"  Probabilities: {proba[0]}")
        
        # Check if scaler was applied
        scaler_step = None
        if hasattr(model, 'named_steps'):
            if 'scaler' in model.named_steps:
                scaler_step = model.named_steps['scaler']
            elif 'scale' in model.named_steps:
                scaler_step = model.named_steps['scale']
        
        if scaler_step is not None:
            scaled_sample = scaler_step.transform(sample)
            
            print(f"\n✓ Scaler applied during prediction")
            print(f"\nScaled feature values (first 5):")
            for i, col in enumerate(feature_cols[:5]):
                print(f"  {col}: {scaled_sample[0, i]:.6f}")
            
            # Check if values are standardized (mean~0, std~1)
            print(f"\nScaled data statistics:")
            print(f"  Mean: {scaled_sample.mean():.6f} (should be ~0)")
            print(f"  Std:  {scaled_sample.std():.6f} (should be ~1)")
            print(f"  Min:  {scaled_sample.min():.6f}")
            print(f"  Max:  {scaled_sample.max():.6f}")
            
            if abs(scaled_sample.mean()) > 0.5:
                print(f"\n⚠️  WARNING: Scaled mean is far from 0!")
                print(f"  This suggests the scaler was not fit on the training data properly")
            else:
                print(f"\n✓ Scaling looks correct (mean ≈ 0)")
        else:
            print(f"\n⚠️  WARNING: No scaler in pipeline!")
            print(f"  Raw features are being used directly!")
            
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()


def check_inference_code():
    """Check if inference code properly uses the model pipeline"""
    
    print("\n" + "="*80)
    print("Inference Code Check")
    print("="*80)
    
    print("\n[INFO] Checking end_to_end_inference.py...")
    
    inference_file = Path("scripts/cnn_rf/end_to_end_inference.py")
    if inference_file.exists():
        with open(inference_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for scaling-related code
        checks = {
            'StandardScaler import': 'StandardScaler' in content,
            'scaler.transform() call': 'scaler.transform' in content or '.transform(' in content,
            'model.predict() call': 'model.predict' in content or '.predict(' in content,
        }
        
        print("\nCode checks:")
        for check_name, result in checks.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check_name}")
        
        if not checks['scaler.transform() call']:
            print(f"\n⚠️  WARNING: No explicit scaler.transform() found!")
            print(f"  If the model is a Pipeline, this is OK (automatic)")
            print(f"  If not, features are NOT being scaled!")
    else:
        print(f"✗ File not found: {inference_file}")


def main():
    """Main function"""
    
    # Check model pipeline
    model = check_model_pipeline()
    
    # Test scaling consistency
    test_scaling_consistency()
    
    # Check inference code
    check_inference_code()
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if hasattr(model, 'named_steps'):
        if 'scaler' in model.named_steps:
            print("\n✓ Model has proper scaling pipeline")
            print("  Scaling is automatically applied during inference")
        else:
            print("\n⚠️  Model pipeline missing scaler!")
            print("\nRecommended fix:")
            print("  1. Retrain model with StandardScaler in pipeline:")
            print("     pipeline = Pipeline([")
            print("         ('scaler', StandardScaler()),")
            print("         ('select', SelectKBest(...)),")
            print("         ('model', RandomForestClassifier(...))")
            print("     ])")
    else:
        print("\n⚠️  Model is not a Pipeline!")
        print("\nRecommended fix:")
        print("  1. Retrain model as a Pipeline with StandardScaler")
        print("  2. Or manually scale features during inference")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
