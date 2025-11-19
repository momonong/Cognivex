"""
Train CNN-RF Model with GM Features Only

This script trains a model using only Gray Matter (GM) features
to avoid the mirror effect between different modalities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib

def train_gm_only_model(
    csv_path="data/roi_features.csv",
    output_dir="model/cnn_rf",
    test_size=0.2,
    random_state=42
):
    """
    Train model using only GM features
    
    Args:
        csv_path: Path to ROI features CSV
        output_dir: Output directory for model
        test_size: Test set size
        random_state: Random seed
    """
    
    print("="*80)
    print("Training CNN-RF Model with GM Features Only")
    print("="*80)
    
    # Load data
    print(f"\n[1/6] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for NC vs AD only
    df_filtered = df[df['Group'].isin(['NC', 'AD'])].copy()
    
    print(f"[OK] Loaded {len(df)} subjects")
    print(f"[OK] Filtered to {len(df_filtered)} subjects (NC vs AD)")
    print(f"[OK] Class distribution:")
    print(df_filtered['Group'].value_counts())
    
    # Select only GM features
    print(f"\n[2/6] Selecting GM features only...")
    all_features = [col for col in df.columns if col not in ['Subject_ID', 'Group']]
    gm_features = [col for col in all_features if col.endswith('_GM')]
    
    print(f"[OK] Total features: {len(all_features)}")
    print(f"[OK] GM features: {len(gm_features)}")
    print(f"[OK] Reduction: {(1 - len(gm_features)/len(all_features))*100:.1f}%")
    
    # Check AD-relevant regions
    ad_regions = [
        'Hippocampus_L', 'Hippocampus_R',
        'Amygdala_L', 'Amygdala_R',
        'Olfactory_L', 'Olfactory_R',
        'ParaHippocampal_L', 'ParaHippocampal_R',
        'Cingulate_Post_L', 'Cingulate_Post_R'
    ]
    
    print(f"\n[INFO] Checking AD-relevant regions in GM features:")
    ad_gm_features = []
    for region in ad_regions:
        feature = region + '_GM'
        if feature in gm_features:
            ad_gm_features.append(feature)
            print(f"  ✓ {feature}")
        else:
            print(f"  ✗ {feature} NOT FOUND")
    
    print(f"\n[OK] AD-relevant GM features: {len(ad_gm_features)} / {len(ad_regions)}")
    
    # Prepare data
    X = df_filtered[gm_features]
    y = df_filtered['Group'].map({'AD': 0, 'NC': 1}).values
    
    # Split data
    print(f"\n[3/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"[OK] Train set: {len(X_train)} samples")
    print(f"[OK] Test set: {len(X_test)} samples")
    print(f"[OK] Train distribution: AD={sum(y_train==0)}, NC={sum(y_train==1)}")
    print(f"[OK] Test distribution: AD={sum(y_test==0)}, NC={sum(y_test==1)}")
    
    # Build pipeline
    print(f"\n[4/6] Building pipeline...")
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('select', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=random_state),
            threshold='median'
        )),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight='balanced'
        ))
    ])
    
    print(f"[OK] Pipeline created:")
    print(f"  1. StandardScaler")
    print(f"  2. SelectFromModel (threshold='median')")
    print(f"  3. RandomForestClassifier (n_estimators=200, balanced)")
    
    # Train model
    print(f"\n[5/6] Training model...")
    pipeline.fit(X_train, y_train)
    
    # Get selected features
    selector = pipeline.named_steps['select']
    selected_mask = selector.get_support()
    selected_features = [feat for feat, selected in zip(gm_features, selected_mask) if selected]
    
    print(f"[OK] Model trained")
    print(f"[OK] Selected features: {len(selected_features)} / {len(gm_features)}")
    
    # Check if AD regions are selected
    ad_selected = [feat for feat in selected_features if any(region in feat for region in ad_regions)]
    print(f"[OK] AD-relevant features selected: {len(ad_selected)} / {len(ad_gm_features)}")
    
    if ad_selected:
        print(f"\n[INFO] Selected AD-relevant features:")
        for feat in ad_selected:
            print(f"  ✓ {feat}")
    else:
        print(f"\n⚠️  WARNING: No AD-relevant features selected!")
    
    # Evaluate
    print(f"\n[6/6] Evaluating model...")
    
    # Training accuracy
    train_score = pipeline.score(X_train, y_train)
    print(f"[OK] Training accuracy: {train_score:.1%}")
    
    # Test accuracy
    test_score = pipeline.score(X_test, y_test)
    print(f"[OK] Test accuracy: {test_score:.1%}")
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"[OK] Cross-validation accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    
    # Detailed metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_pred = pipeline.predict(X_test)
    
    print(f"\n[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=['AD', 'NC']))
    
    print(f"\n[Confusion Matrix]")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              AD    NC")
    print(f"Actual  AD    {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"        NC    {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "rf_model_NC_vs_AD_GM_only.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\n[SAVED] Model: {model_path}")
    
    # Save selected features
    features_path = output_dir / "selected_features_GM_only.txt"
    with open(features_path, 'w') as f:
        for feat in selected_features:
            f.write(feat + '\n')
    print(f"[SAVED] Selected features: {features_path}")
    
    # Save metadata
    metadata = {
        'total_features': len(gm_features),
        'selected_features': len(selected_features),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'ad_features_selected': len(ad_selected),
        'selected_feature_names': selected_features
    }
    
    import json
    metadata_path = output_dir / "model_metadata_GM_only.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Metadata: {metadata_path}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
    return pipeline, selected_features


def main():
    """Main function"""
    pipeline, selected_features = train_gm_only_model()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Test the new model:")
    print("   python scripts/cnn_rf/debug_biomarkers.py --subject sub-0005")
    print("\n2. Compare with original model:")
    print("   python scripts/cnn_rf/compare_models.py")
    print("\n3. Update inference to use new model:")
    print("   model_path='model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib'")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
