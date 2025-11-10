"""
Train Final Model with Hybrid ROI Selection
使用混合 ROI 選擇策略訓練最終模型

Hybrid Strategy:
- 24 literature-based ROIs (domain knowledge)
- + 8 data-driven ROIs from Top 30 analysis
- = 32 ROIs total
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Use English for plots
plt.rcParams['font.family'] = 'sans-serif'

# Final 32 ROIs: Hybrid Selection
FINAL_32_ROIS = [
    # Original 24 ROIs (literature-based)
    'Hippocampus_L', 'Hippocampus_R',
    'Amygdala_L', 'Amygdala_R',
    'ParaHippocampal_L', 'ParaHippocampal_R',
    'Temporal_Sup_L', 'Temporal_Sup_R',
    'Temporal_Mid_L', 'Temporal_Mid_R',
    'Temporal_Inf_L', 'Temporal_Inf_R',
    'Parietal_Sup_L', 'Parietal_Sup_R',
    'Parietal_Inf_L', 'Parietal_Inf_R',
    'Cingulum_Ant_L', 'Cingulum_Ant_R',
    'Cingulum_Post_L', 'Cingulum_Post_R',
    'Frontal_Sup_L', 'Frontal_Sup_R',
    'Frontal_Mid_L', 'Frontal_Mid_R',
    
    # Additional 8 ROIs (data-driven from Top 30 analysis)
    'Cingulum_Mid_L', 'Cingulum_Mid_R',      # Mid cingulate (selected by both methods)
    'Fusiform_L', 'Fusiform_R',              # Object recognition (AD-relevant)
    'Lingual_L', 'Lingual_R',                # Visual processing (AD-relevant)
    'SupraMarginal_L', 'SupraMarginal_R'    # Language processing (AD-relevant)
]

# Rationale for additional ROIs
ROI_RATIONALE = {
    'Cingulum_Mid': 'Part of Default Mode Network, metabolic changes in AD',
    'Fusiform': 'Object and face recognition, impaired in AD',
    'Lingual': 'Visual processing, connected to memory systems',
    'SupraMarginal': 'Language and semantic processing, affected in AD'
}


def load_features(features_path, selected_rois):
    """載入並選擇特定的 ROI 特徵"""
    print(f"Loading features from: {features_path}")
    
    df = pd.read_csv(features_path)
    
    # Check which ROIs are available
    available_rois = [roi for roi in selected_rois if roi in df.columns]
    missing_rois = [roi for roi in selected_rois if roi not in df.columns]
    
    if missing_rois:
        print(f"\n⚠ Warning: {len(missing_rois)} ROIs not found in data:")
        for roi in missing_rois:
            print(f"  - {roi}")
    
    print(f"\n✓ Using {len(available_rois)}/{len(selected_rois)} ROIs")
    
    # Extract features and labels
    X = df[available_rois].values
    y = df['label_id'].values if 'label_id' in df.columns else df['label'].map({'NC': 0, 'AD': 1}).values
    subject_ids = df['subject_id'].values
    
    print(f"✓ Samples: {len(df)} (NC: {(y==0).sum()}, AD: {(y==1).sum()})")
    
    return X, y, available_rois, subject_ids


def train_and_evaluate(X, y, feature_names, n_folds=5):
    """訓練並評估模型"""
    print("\n" + "="*80)
    print("Training Final Model")
    print("="*80)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation with multiple metrics
    print(f"\nPerforming {n_folds}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring, 
                                return_train_score=True, n_jobs=-1)
    
    # Print CV results
    print("\nCross-Validation Results:")
    print("-"*80)
    print(f"Accuracy:  {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
    print(f"Precision: {cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}")
    print(f"Recall:    {cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}")
    print(f"F1 Score:  {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")
    print(f"ROC-AUC:   {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")
    
    # Train on full dataset
    print("\nTraining on full dataset...")
    model.fit(X_scaled, y)
    train_score = model.score(X_scaled, y)
    
    print(f"Train Accuracy: {train_score:.3f}")
    print(f"Overfitting Gap: {train_score - cv_results['test_accuracy'].mean():.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'ROI': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important ROIs:")
    print("-"*80)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['ROI']:30s} {row['Importance']:.4f}")
    
    return model, scaler, feature_importance, cv_results


def save_model(model, scaler, feature_names, output_dir):
    """儲存模型和相關檔案"""
    print("\nSaving model...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'final_model.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Saved model: {model_path}")
    
    # Save scaler
    scaler_path = output_dir / 'final_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved scaler: {scaler_path}")
    
    # Save feature names
    features_path = output_dir / 'final_feature_names.txt'
    with open(features_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print(f"✓ Saved feature names: {features_path}")
    
    # Save ROI list as CSV
    roi_df = pd.DataFrame({
        'ROI': feature_names,
        'Source': ['Original 24' if roi in FINAL_32_ROIS[:24] else 'Data-driven' 
                   for roi in feature_names]
    })
    roi_csv_path = output_dir / 'final_roi_list.csv'
    roi_df.to_csv(roi_csv_path, index=False)
    print(f"✓ Saved ROI list: {roi_csv_path}")


def visualize_results(feature_importance, cv_results, output_dir):
    """視覺化結果"""
    print("\nGenerating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Feature Importance
    ax = axes[0, 0]
    top20 = feature_importance.head(20)
    
    # Color by source
    colors = ['red' if roi in FINAL_32_ROIS[:24] else 'blue' 
              for roi in top20['ROI']]
    
    y_pos = np.arange(len(top20))
    ax.barh(y_pos, top20['Importance'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20['ROI'], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Top 20 Feature Importance\nRed=Original 24, Blue=Data-driven', 
                fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. CV Metrics
    ax = axes[0, 1]
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    means = [cv_results[f'test_{m}'].mean() for m in metrics]
    stds = [cv_results[f'test_{m}'].std() for m in metrics]
    
    x_pos = np.arange(len(metrics))
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.upper() for m in metrics], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Cross-Validation Performance', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. ROI Source Distribution
    ax = axes[1, 0]
    source_counts = {
        'Original 24\n(Literature)': 24,
        'Data-driven\n(Top 30)': 8
    }
    
    colors_pie = ['red', 'blue']
    ax.pie(source_counts.values(), labels=source_counts.keys(), autopct='%1.1f%%',
          colors=colors_pie, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax.set_title('ROI Source Distribution (32 Total)', fontsize=12, fontweight='bold')
    
    # 4. Importance by Source
    ax = axes[1, 1]
    
    original_importance = feature_importance[
        feature_importance['ROI'].isin(FINAL_32_ROIS[:24])
    ]['Importance'].sum()
    
    datadriven_importance = feature_importance[
        ~feature_importance['ROI'].isin(FINAL_32_ROIS[:24])
    ]['Importance'].sum()
    
    sources = ['Original 24', 'Data-driven 8']
    importances = [original_importance, datadriven_importance]
    
    bars = ax.bar(sources, importances, color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('Total Importance')
    ax.set_title('Total Importance by Source', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, imp in zip(bars, importances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{imp:.3f}\n({imp/sum(importances)*100:.1f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_model_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'final_model_analysis.png'}")
    plt.close()


def generate_report(feature_importance, cv_results, output_dir):
    """生成最終報告"""
    print("\nGenerating final report...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'final_model_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Final Model Training Report\n")
        f.write("Hybrid ROI Selection Strategy\n")
        f.write("="*80 + "\n\n")
        
        f.write("Model Configuration:\n")
        f.write("-"*80 + "\n")
        f.write("Algorithm: Random Forest Classifier\n")
        f.write("Number of ROIs: 32\n")
        f.write("  - Original 24 (literature-based)\n")
        f.write("  - Additional 8 (data-driven from Top 30 analysis)\n")
        f.write("Hyperparameters:\n")
        f.write("  - n_estimators: 500\n")
        f.write("  - max_depth: 10\n")
        f.write("  - min_samples_split: 5\n")
        f.write("  - class_weight: balanced\n\n")
        
        f.write("="*80 + "\n")
        f.write("Performance Metrics (5-Fold Cross-Validation)\n")
        f.write("="*80 + "\n\n")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in metrics:
            mean = cv_results[f'test_{metric}'].mean()
            std = cv_results[f'test_{metric}'].std()
            f.write(f"{metric.upper():12s}: {mean:.3f} ± {std:.3f}\n")
        
        train_mean = cv_results['train_accuracy'].mean()
        test_mean = cv_results['test_accuracy'].mean()
        gap = train_mean - test_mean
        
        f.write(f"\nTrain Accuracy: {train_mean:.3f}\n")
        f.write(f"Test Accuracy:  {test_mean:.3f}\n")
        f.write(f"Overfitting Gap: {gap:.3f}\n\n")
        
        if gap < 0.1:
            f.write("Assessment: ✓ Low overfitting risk\n")
        elif gap < 0.2:
            f.write("Assessment: ⚠ Moderate overfitting\n")
        else:
            f.write("Assessment: ✗ High overfitting risk\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Feature Importance Analysis\n")
        f.write("="*80 + "\n\n")
        
        f.write("Top 20 Most Important ROIs:\n")
        f.write("-"*80 + "\n")
        
        for idx, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
            roi = row['ROI']
            imp = row['Importance']
            source = "Original" if roi in FINAL_32_ROIS[:24] else "Data-driven"
            marker = "★" if source == "Original" else "●"
            f.write(f"{marker} {idx:2d}. {roi:30s} {imp:.4f} [{source}]\n")
        
        f.write("\n★ = Original 24 (literature-based)\n")
        f.write("● = Data-driven (from Top 30 analysis)\n\n")
        
        # Importance by source
        original_importance = feature_importance[
            feature_importance['ROI'].isin(FINAL_32_ROIS[:24])
        ]['Importance'].sum()
        
        datadriven_importance = feature_importance[
            ~feature_importance['ROI'].isin(FINAL_32_ROIS[:24])
        ]['Importance'].sum()
        
        total_importance = original_importance + datadriven_importance
        
        f.write("Importance by Source:\n")
        f.write("-"*80 + "\n")
        f.write(f"Original 24 ROIs:    {original_importance:.4f} ({original_importance/total_importance*100:.1f}%)\n")
        f.write(f"Data-driven 8 ROIs:  {datadriven_importance:.4f} ({datadriven_importance/total_importance*100:.1f}%)\n\n")
        
        f.write("="*80 + "\n")
        f.write("Additional ROIs and Their Rationale\n")
        f.write("="*80 + "\n\n")
        
        for roi_base, rationale in ROI_RATIONALE.items():
            f.write(f"{roi_base} (L/R):\n")
            f.write(f"  {rationale}\n\n")
        
        f.write("="*80 + "\n")
        f.write("Comparison with Previous Models\n")
        f.write("="*80 + "\n\n")
        
        f.write("Model Evolution:\n")
        f.write("  1. Original 24 ROIs:     CV Accuracy ~73.8%\n")
        f.write("  2. Top 30 (F-test):      CV Accuracy ~81.5%\n")
        f.write(f"  3. Final 32 (Hybrid):    CV Accuracy ~{test_mean:.1%}\n\n")
        
        f.write("="*80 + "\n")
        f.write("Conclusion\n")
        f.write("="*80 + "\n\n")
        
        if test_mean >= 0.80:
            f.write("✓ EXCELLENT: Model achieves high accuracy with good generalization\n")
            f.write("✓ The hybrid approach successfully combines domain knowledge\n")
            f.write("  with data-driven insights\n")
        elif test_mean >= 0.75:
            f.write("✓ GOOD: Model performs well with acceptable accuracy\n")
            f.write("✓ The hybrid approach shows promise\n")
        else:
            f.write("⚠ MODERATE: Model performance is acceptable but could be improved\n")
            f.write("  Consider collecting more data or refining feature selection\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved report: {report_path}")
    
    # Save feature importance as CSV
    csv_path = output_dir / 'final_feature_importance.csv'
    feature_importance.to_csv(csv_path, index=False)
    print(f"✓ Saved feature importance: {csv_path}")


def main():
    """主函數"""
    print("="*80)
    print("Training Final Model with Hybrid ROI Selection")
    print("="*80)
    
    print("\nHybrid Strategy:")
    print("  - 24 literature-based ROIs (domain knowledge)")
    print("  - + 8 data-driven ROIs (from Top 30 analysis)")
    print("  - = 32 ROIs total")
    
    print("\nAdditional ROIs:")
    for roi_base, rationale in ROI_RATIONALE.items():
        print(f"  • {roi_base}: {rationale}")
    
    try:
        # Load features
        features_path = Path('data/processed/all_aal_roi_features.csv')
        
        if not features_path.exists():
            print(f"\n⚠ Feature file not found: {features_path}")
            print("\nPlease run feature extraction first:")
            print("  python scripts/ml/extract_all_roi_features.py")
            return
        
        X, y, feature_names, subject_ids = load_features(features_path, FINAL_32_ROIS)
        
        # Train and evaluate
        model, scaler, feature_importance, cv_results = train_and_evaluate(
            X, y, feature_names
        )
        
        # Save model
        output_dir = Path('model/ml/final')
        save_model(model, scaler, feature_names, output_dir)
        
        # Visualize
        output_dir_viz = Path('output/ml/final_model')
        visualize_results(feature_importance, cv_results, output_dir_viz)
        
        # Generate report
        generate_report(feature_importance, cv_results, output_dir_viz)
        
        print("\n" + "="*80)
        print("Final Model Training Complete!")
        print("="*80)
        print(f"\nModel saved to: model/ml/final/")
        print(f"Results saved to: output/ml/final_model/")
        
        # Print summary
        print("\nQuick Summary:")
        print("-"*80)
        print(f"ROIs: {len(feature_names)}")
        print(f"CV Accuracy: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
        print(f"CV F1 Score: {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}")
        print(f"Overfitting Gap: {cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean():.3f}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
