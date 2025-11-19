"""
Debug Inference Values - Step-by-Step Analysis

This script simulates the inference process and prints actual numbers
to identify where the data pipeline might be failing.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.cnn_rf.end_to_end_inference import EndToEndPredictor

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def debug_inference_pipeline(
    subject_ids=['sub-0005', 'sub-0010'],
    model_path="model/cnn_rf/rf_model_NC_vs_AD.joblib",
    csv_path="data/roi_features.csv"
):
    """
    Debug the inference pipeline step by step
    
    Args:
        subject_ids: List of subject IDs to compare
        model_path: Path to model
        csv_path: Path to CSV features
    """
    
    print("="*80)
    print("INFERENCE PIPELINE DEBUG")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load Model and Check Structure
    # ========================================================================
    print(f"\n[STEP 1] Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    print(f"[OK] Model type: {type(model)}")
    
    if hasattr(model, 'named_steps'):
        print(f"[OK] Pipeline steps: {list(model.named_steps.keys())}")
        
        # Get scaler
        scaler = None
        if 'scaler' in model.named_steps:
            scaler = model.named_steps['scaler']
        elif 'scale' in model.named_steps:
            scaler = model.named_steps['scale']
        
        if scaler:
            print(f"[OK] Scaler found: {type(scaler).__name__}")
            print(f"  - Mean shape: {scaler.mean_.shape}")
            print(f"  - Scale shape: {scaler.scale_.shape}")
        
        # Get selector
        selector = None
        if 'select' in model.named_steps:
            selector = model.named_steps['select']
            selected_mask = selector.get_support()
            n_selected = selected_mask.sum()
            print(f"[OK] Selector found: {type(selector).__name__}")
            print(f"  - Selected features: {n_selected} / {len(selected_mask)}")
        
        # Get RF model
        if 'model' in model.named_steps:
            rf_model = model.named_steps['model']
            print(f"[OK] RF model found: {type(rf_model).__name__}")
            print(f"  - n_features_in_: {rf_model.n_features_in_}")
            print(f"  - classes_: {rf_model.classes_}")
    
    # ========================================================================
    # STEP 2: Load CSV Data
    # ========================================================================
    print(f"\n[STEP 2] Loading CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    feature_cols = [col for col in df.columns if col not in ['Subject_ID', 'Group']]
    print(f"[OK] Loaded {len(df)} subjects")
    print(f"[OK] Feature columns: {len(feature_cols)}")
    print(f"[OK] First 5 features: {feature_cols[:5]}")
    
    # ========================================================================
    # STEP 3: Check Feature Order
    # ========================================================================
    print(f"\n[STEP 3] Checking feature order...")
    
    # Get expected feature names from model
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
        print(f"[OK] Model expects {len(model_features)} features")
        print(f"[OK] First 5 expected: {list(model_features[:5])}")
        
        # Check if order matches
        if list(feature_cols) == list(model_features):
            print(f"[OK] ✓ Feature order MATCHES!")
        else:
            print(f"[ERROR] ✗ Feature order MISMATCH!")
            
            # Find differences
            for i, (csv_feat, model_feat) in enumerate(zip(feature_cols[:10], model_features[:10])):
                if csv_feat != model_feat:
                    print(f"  Position {i}: CSV='{csv_feat}' vs Model='{model_feat}'")
    else:
        print(f"[WARN] Model does not have feature_names_in_")
    
    # ========================================================================
    # STEP 4: Compare Two Subjects
    # ========================================================================
    print(f"\n[STEP 4] Comparing subjects: {subject_ids}")
    
    # Key features to monitor
    key_features = [
        'Hippocampus_L_GM',
        'Hippocampus_R_GM',
        'Amygdala_L_GM',
        'Amygdala_R_GM',
        'Supp_Motor_Area_L_GM',
        'Supp_Motor_Area_L_FA'
    ]
    
    # Filter available features
    available_key_features = [f for f in key_features if f in feature_cols]
    
    print(f"\n[INFO] Monitoring {len(available_key_features)} key features:")
    for feat in available_key_features:
        print(f"  - {feat}")
    
    # Load subjects from CSV
    subjects_data = {}
    for subject_id in subject_ids:
        subject_row = df[df['Subject_ID'] == subject_id]
        if subject_row.empty:
            print(f"\n[ERROR] Subject {subject_id} not found in CSV!")
            continue
        
        subjects_data[subject_id] = {
            'group': subject_row['Group'].values[0],
            'features': subject_row[feature_cols].values[0]
        }
        print(f"\n[OK] Loaded {subject_id} (Group: {subjects_data[subject_id]['group']})")
    
    # ========================================================================
    # STEP 5: Print Raw Values Comparison
    # ========================================================================
    print(f"\n[STEP 5] Raw Feature Values Comparison")
    print("="*80)
    
    # Create comparison table
    print(f"\n{'Feature':<30}", end='')
    for subject_id in subject_ids:
        print(f"{subject_id:>15}", end='')
    print(f"{'Difference':>15}")
    print("-"*80)
    
    for feat in available_key_features:
        feat_idx = feature_cols.index(feat)
        print(f"{feat:<30}", end='')
        
        values = []
        for subject_id in subject_ids:
            if subject_id in subjects_data:
                value = subjects_data[subject_id]['features'][feat_idx]
                values.append(value)
                print(f"{value:>15.6f}", end='')
            else:
                print(f"{'N/A':>15}", end='')
        
        if len(values) == 2:
            diff = abs(values[0] - values[1])
            print(f"{diff:>15.6f}")
        else:
            print()
    
    # ========================================================================
    # STEP 6: Apply Scaling
    # ========================================================================
    print(f"\n[STEP 6] Scaled Feature Values (After StandardScaler)")
    print("="*80)
    
    if scaler:
        print(f"\n{'Feature':<30}", end='')
        for subject_id in subject_ids:
            print(f"{subject_id:>15}", end='')
        print(f"{'Difference':>15}")
        print("-"*80)
        
        for feat in available_key_features:
            feat_idx = feature_cols.index(feat)
            print(f"{feat:<30}", end='')
            
            scaled_values = []
            for subject_id in subject_ids:
                if subject_id in subjects_data:
                    raw_value = subjects_data[subject_id]['features'][feat_idx]
                    # Apply scaling manually
                    scaled_value = (raw_value - scaler.mean_[feat_idx]) / scaler.scale_[feat_idx]
                    scaled_values.append(scaled_value)
                    print(f"{scaled_value:>15.6f}", end='')
                else:
                    print(f"{'N/A':>15}", end='')
            
            if len(scaled_values) == 2:
                diff = abs(scaled_values[0] - scaled_values[1])
                print(f"{diff:>15.6f}")
            else:
                print()
    else:
        print("[WARN] No scaler found, skipping scaled values")
    
    # ========================================================================
    # STEP 7: Model Input (After Feature Selection)
    # ========================================================================
    print(f"\n[STEP 7] Model Input (After Feature Selection)")
    print("="*80)
    
    if selector:
        selected_mask = selector.get_support()
        selected_features = [feat for feat, selected in zip(feature_cols, selected_mask) if selected]
        
        print(f"\n[INFO] {len(selected_features)} features selected")
        print(f"[INFO] Checking if key features are selected:")
        
        for feat in available_key_features:
            if feat in selected_features:
                print(f"  ✓ {feat}")
            else:
                print(f"  ✗ {feat} (NOT SELECTED)")
        
        # Show selected key features
        selected_key_features = [f for f in available_key_features if f in selected_features]
        
        if selected_key_features:
            print(f"\n{'Feature':<30}", end='')
            for subject_id in subject_ids:
                print(f"{subject_id:>15}", end='')
            print(f"{'Difference':>15}")
            print("-"*80)
            
            for feat in selected_key_features:
                feat_idx = feature_cols.index(feat)
                print(f"{feat:<30}", end='')
                
                model_input_values = []
                for subject_id in subject_ids:
                    if subject_id in subjects_data:
                        # Full pipeline transform
                        X = pd.DataFrame([subjects_data[subject_id]['features']], columns=feature_cols)
                        X_transformed = model[:-1].transform(X)  # All steps except final model
                        
                        # Find this feature in transformed data
                        selected_feat_idx = selected_features.index(feat)
                        value = X_transformed[0, selected_feat_idx]
                        model_input_values.append(value)
                        print(f"{value:>15.6f}", end='')
                    else:
                        print(f"{'N/A':>15}", end='')
                
                if len(model_input_values) == 2:
                    diff = abs(model_input_values[0] - model_input_values[1])
                    print(f"{diff:>15.6f}")
                else:
                    print()
        else:
            print(f"\n[WARN] None of the key features were selected!")
    
    # ========================================================================
    # STEP 8: Run Predictions
    # ========================================================================
    print(f"\n[STEP 8] Running Predictions")
    print("="*80)
    
    for subject_id in subject_ids:
        if subject_id not in subjects_data:
            continue
        
        print(f"\n[{subject_id}] (True: {subjects_data[subject_id]['group']})")
        
        # Prepare input
        X = pd.DataFrame([subjects_data[subject_id]['features']], columns=feature_cols)
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Map prediction
        class_names = ['AD', 'NC'] if len(probabilities) == 2 else ['Class_0', 'Class_1']
        predicted_class = class_names[prediction]
        
        print(f"  Predicted: {predicted_class}")
        print(f"  Probabilities:")
        for cls, prob in zip(class_names, probabilities):
            print(f"    {cls}: {prob:.1%}")
    
    # ========================================================================
    # STEP 9: SHAP Values Comparison
    # ========================================================================
    if SHAP_AVAILABLE:
        print(f"\n[STEP 9] SHAP Values Comparison")
        print("="*80)
        
        try:
            # Get RF model
            if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                rf_model = model.named_steps['model']
                explainer = shap.TreeExplainer(rf_model)
                
                print(f"\n[INFO] Calculating SHAP values...")
                
                for subject_id in subject_ids:
                    if subject_id not in subjects_data:
                        continue
                    
                    print(f"\n[{subject_id}]")
                    
                    # Prepare input (full pipeline)
                    X = pd.DataFrame([subjects_data[subject_id]['features']], columns=feature_cols)
                    X_transformed = model[:-1].transform(X)  # All steps except final model
                    
                    # Calculate SHAP
                    shap_values = explainer.shap_values(X_transformed)
                    
                    # Get SHAP for AD class (class 0)
                    if isinstance(shap_values, list):
                        shap_ad = shap_values[0][0]
                    else:
                        shap_ad = shap_values[0]
                    
                    # Get top 5
                    abs_shap = np.abs(shap_ad)
                    top_indices = np.argsort(abs_shap)[-5:][::-1]
                    
                    print(f"  Top 5 SHAP features:")
                    top_indices_list = top_indices.tolist() if hasattr(top_indices, 'tolist') else list(top_indices)
                    
                    for i, idx in enumerate(top_indices_list, 1):
                        if selector:
                            feat_name = selected_features[idx]
                        else:
                            feat_name = f"Feature_{idx}"
                        
                        shap_val = float(shap_ad[idx])
                        direction = "→ AD" if shap_val > 0 else "← NC"
                        print(f"    {i}. {feat_name:<30} SHAP: {shap_val:+.4f} {direction}")
        
        except Exception as e:
            print(f"[ERROR] SHAP calculation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # STEP 10: Summary
    # ========================================================================
    print(f"\n[STEP 10] Summary")
    print("="*80)
    
    print(f"\n[Checklist]")
    print(f"  {'✓' if scaler else '✗'} Scaler found and applied")
    print(f"  {'✓' if selector else '✗'} Feature selector found")
    
    if selector:
        selected_key_count = sum(1 for f in available_key_features if f in selected_features)
        print(f"  {'✓' if selected_key_count > 0 else '✗'} Key AD features selected: {selected_key_count}/{len(available_key_features)}")
        
        # Show what WAS selected
        print(f"\n[INFO] Actually selected features (first 10):")
        for i, feat in enumerate(selected_features[:10], 1):
            print(f"  {i}. {feat}")
    
    # Check if values are different
    if len(subjects_data) == 2:
        subject_ids_list = list(subjects_data.keys())
        feat_idx = feature_cols.index(available_key_features[0])
        val1 = subjects_data[subject_ids_list[0]]['features'][feat_idx]
        val2 = subjects_data[subject_ids_list[1]]['features'][feat_idx]
        
        if abs(val1 - val2) > 0.001:
            print(f"  ✓ Raw values are DIFFERENT between subjects")
        else:
            print(f"  ✗ Raw values are IDENTICAL (possible bug!)")
    
    print("\n" + "="*80)
    
    # ========================================================================
    # DIAGNOSIS
    # ========================================================================
    print(f"\n[DIAGNOSIS]")
    print("="*80)
    
    if selector and selected_key_count == 0:
        print(f"\n⚠️  CRITICAL ISSUE FOUND!")
        print(f"\nThe feature selector (SelectFromModel) has excluded ALL key AD biomarkers!")
        print(f"\nThis explains why:")
        print(f"  1. SHAP values are similar across subjects")
        print(f"  2. Top features are motor areas instead of memory areas")
        print(f"  3. Model is not using biologically relevant features")
        print(f"\nRECOMMENDED FIX:")
        print(f"  1. Use the GM-only model: rf_model_NC_vs_AD_GM_only.joblib")
        print(f"  2. Or retrain with less aggressive feature selection")
        print(f"  3. Or manually select AD-relevant features")
    
    print("\n" + "="*80)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug inference values")
    parser.add_argument(
        '--subjects',
        nargs='+',
        default=['sub-0005', 'sub-0010'],
        help='Subject IDs to compare'
    )
    parser.add_argument(
        '--model',
        default='model/cnn_rf/rf_model_NC_vs_AD.joblib',
        help='Path to model'
    )
    parser.add_argument(
        '--csv',
        default='data/roi_features.csv',
        help='Path to CSV features'
    )
    
    args = parser.parse_args()
    
    debug_inference_pipeline(
        subject_ids=args.subjects,
        model_path=args.model,
        csv_path=args.csv
    )


if __name__ == "__main__":
    main()
