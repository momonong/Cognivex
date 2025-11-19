"""Test GM-only model"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.cnn_rf.end_to_end_inference import EndToEndPredictor

print("="*80)
print("Testing GM-Only Model")
print("="*80)

# Initialize predictor with GM-only model
predictor = EndToEndPredictor(
    model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
    data_root="data/MRI_processed"
)

# Test on two subjects
subjects = ['sub-0005', 'sub-0010']

for subject_id in subjects:
    print(f"\n{'='*80}")
    print(f"Subject: {subject_id}")
    print("="*80)
    
    results = predictor.predict_subject(subject_id, verbose=True)
    
    print(f"\n[Results]")
    print(f"  Predicted: {results['predicted_label']}")
    print(f"  Confidence: {results['confidence']:.1%}")
    print(f"  True Label: {results['true_label']}")
    print(f"  Correct: {results['correct']}")
    
    # Check SHAP features
    shap_features = results.get('shap_features', [])
    if shap_features:
        print(f"\n[SHAP Features] Top 10:")
        for i, feat in enumerate(shap_features[:10], 1):
            print(f"  {i:2d}. {feat['name']:<40} SHAP: {feat['shap_value']:+.4f} ({feat['direction']})")
        
        # Check for AD biomarkers
        ad_biomarkers = ['Hippocampus', 'Amygdala', 'ParaHippocampal', 'Cingulate_Post']
        biomarker_count = 0
        for feat in shap_features[:10]:
            if any(marker in feat['name'] for marker in ad_biomarkers):
                biomarker_count += 1
        
        print(f"\n[INFO] AD biomarkers in top 10: {biomarker_count}/10")
    else:
        print(f"\n[WARN] No SHAP features available")

print("\n" + "="*80)
print("Test Complete!")
print("="*80)
