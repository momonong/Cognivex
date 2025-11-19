"""
Debug Biomarkers - Monitor AD-relevant regions

This script explicitly checks if the model is "seeing" key AD biomarkers
like Hippocampus and Amygdala, regardless of their SHAP ranking.
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


def monitor_biomarkers(subject_id, predictor=None):
    """
    Monitor key AD biomarkers for a specific subject
    
    Args:
        subject_id: Subject identifier
        predictor: EndToEndPredictor instance (optional)
    """
    
    print("="*80)
    print(f"Biomarker Monitor: {subject_id}")
    print("="*80)
    
    # Initialize predictor if not provided
    if predictor is None:
        predictor = EndToEndPredictor(
            model_path="model/cnn_rf/rf_model_NC_vs_AD.joblib",
            data_root="data/MRI_processed"
        )
    
    # Get prediction results
    print(f"\n[1/4] Running prediction...")
    results = predictor.predict_subject(subject_id, verbose=False)
    
    print(f"\n[Prediction Results]")
    print(f"  Predicted: {results['predicted_label']}")
    print(f"  Confidence: {results['confidence']:.1%}")
    print(f"  True Label: {results['true_label']}")
    print(f"  Correct: {results['correct']}")
    
    # Get features
    features = results['features']
    feature_names = list(features.keys())
    feature_values = np.array(list(features.values()))
    
    # Define AD-relevant biomarkers
    ad_biomarkers = {
        'Hippocampus': ['Hippocampus_L', 'Hippocampus_R'],
        'Amygdala': ['Amygdala_L', 'Amygdala_R'],
        'Olfactory': ['Olfactory_L', 'Olfactory_R'],
        'ParaHippocampal': ['ParaHippocampal_L', 'ParaHippocampal_R'],
        'Posterior Cingulate': ['Cingulate_Post_L', 'Cingulate_Post_R'],
        'Entorhinal': ['Entorhinal_L', 'Entorhinal_R']
    }
    
    # Check which biomarkers exist
    print(f"\n[2/4] Checking AD-relevant biomarkers...")
    
    available_biomarkers = {}
    for region_name, roi_list in ad_biomarkers.items():
        available = []
        for roi in roi_list:
            for modality in ['_GM', '_FA', '_MD']:
                feature_name = roi + modality
                if feature_name in feature_names:
                    available.append(feature_name)
        
        if available:
            available_biomarkers[region_name] = available
            print(f"  ✓ {region_name}: {len(available)} features")
        else:
            print(f"  ✗ {region_name}: NOT FOUND")
    
    # Calculate statistics
    print(f"\n[3/4] Calculating feature statistics...")
    
    # Load training data for z-score calculation
    try:
        train_df = pd.read_csv("data/roi_features.csv")
        train_features = train_df[[col for col in train_df.columns if col not in ['Subject_ID', 'Group']]]
        
        # Calculate mean and std from training data
        train_mean = train_features.mean()
        train_std = train_features.std()
        
        print(f"  ✓ Loaded training statistics from {len(train_df)} subjects")
    except Exception as e:
        print(f"  ⚠️  Could not load training data: {e}")
        train_mean = None
        train_std = None
    
    # Get SHAP values if available
    shap_features = results.get('shap_features', [])
    shap_dict = {feat['name']: feat['shap_value'] for feat in shap_features}
    
    # Print biomarker details
    print(f"\n[4/4] Biomarker Details:")
    print("="*80)
    
    for region_name, feature_list in available_biomarkers.items():
        print(f"\n{region_name}:")
        print("-" * 80)
        print(f"{'Feature':<40} {'Raw Value':>12} {'Z-Score':>10} {'SHAP':>10} {'Direction':>12}")
        print("-" * 80)
        
        for feature_name in sorted(feature_list):
            # Raw value
            raw_value = features[feature_name]
            
            # Z-score
            if train_mean is not None and feature_name in train_mean.index:
                z_score = (raw_value - train_mean[feature_name]) / train_std[feature_name]
                z_score_str = f"{z_score:+.4f}"
            else:
                z_score_str = "N/A"
            
            # SHAP value
            if feature_name in shap_dict:
                shap_value = shap_dict[feature_name]
                shap_str = f"{shap_value:+.4f}"
                direction = "→ AD" if shap_value > 0 else "← NC"
            else:
                shap_str = "N/A"
                direction = "N/A"
            
            print(f"{feature_name:<40} {raw_value:>12.6f} {z_score_str:>10} {shap_str:>10} {direction:>12}")
    
    # Compare with top SHAP features
    print("\n" + "="*80)
    print("Comparison with Top SHAP Features")
    print("="*80)
    
    if shap_features:
        print(f"\nTop 10 SHAP features:")
        for i, feat in enumerate(shap_features[:10], 1):
            # Check if it's a biomarker
            is_biomarker = False
            for region_name, feature_list in available_biomarkers.items():
                if feat['name'] in feature_list:
                    is_biomarker = True
                    marker = f"[{region_name}]"
                    break
            
            if not is_biomarker:
                marker = ""
            
            print(f"  {i:2d}. {feat['name']:<40} SHAP: {feat['shap_value']:+.4f} {marker}")
        
        # Count biomarkers in top 10
        biomarker_count = 0
        for feat in shap_features[:10]:
            for region_name, feature_list in available_biomarkers.items():
                if feat['name'] in feature_list:
                    biomarker_count += 1
                    break
        
        print(f"\n⚠️  AD biomarkers in top 10: {biomarker_count} / 10")
        
        if biomarker_count == 0:
            print(f"\n⚠️  WARNING: NO AD-relevant biomarkers in top 10!")
            print(f"  This suggests the model may not be learning biologically relevant patterns")
    
    return results


def compare_subjects(subject_ids):
    """Compare biomarkers across multiple subjects"""
    
    print("\n" + "="*80)
    print("Multi-Subject Biomarker Comparison")
    print("="*80)
    
    # Initialize predictor once
    predictor = EndToEndPredictor(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    
    all_results = []
    
    for subject_id in subject_ids:
        print(f"\n{'='*80}")
        results = monitor_biomarkers(subject_id, predictor)
        all_results.append(results)
    
    # Summary comparison
    print("\n" + "="*80)
    print("Summary Comparison")
    print("="*80)
    
    print(f"\n{'Subject':<15} {'True':<8} {'Predicted':<12} {'Confidence':>12} {'Correct':>10}")
    print("-" * 80)
    
    for subject_id, results in zip(subject_ids, all_results):
        print(f"{subject_id:<15} {results['true_label']:<8} {results['predicted_label']:<12} "
              f"{results['confidence']:>11.1%} {str(results['correct']):>10}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor AD biomarkers")
    parser.add_argument(
        '--subject',
        default='sub-0005',
        help='Subject ID to analyze'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Multiple subjects to compare'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_subjects(args.compare)
    else:
        monitor_biomarkers(args.subject)
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("\nIf AD biomarkers are NOT in top features:")
    print("  1. Check for data leakage (same features with opposite signs)")
    print("  2. Verify scaling is applied correctly")
    print("  3. Check for high collinearity (run debug_collinearity.py)")
    print("  4. Consider feature selection based on biological relevance")
    print("  5. Try training on GM features only")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
