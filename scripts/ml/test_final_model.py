"""
Test Final Model on All Data
測試最終模型並分析關注的腦區
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Use English for plots
plt.rcParams['font.family'] = 'sans-serif'

# AD 相關的關鍵腦區
AD_CRITICAL_REGIONS = {
    'Hippocampus & Amygdala': [
        'Hippocampus_L', 'Hippocampus_R',
        'Amygdala_L', 'Amygdala_R',
        'ParaHippocampal_L', 'ParaHippocampal_R'
    ],
    'Temporal Lobe': [
        'Temporal_Sup_L', 'Temporal_Sup_R',
        'Temporal_Mid_L', 'Temporal_Mid_R',
        'Temporal_Inf_L', 'Temporal_Inf_R'
    ],
    'Parietal Lobe': [
        'Parietal_Sup_L', 'Parietal_Sup_R',
        'Parietal_Inf_L', 'Parietal_Inf_R',
        'SupraMarginal_L', 'SupraMarginal_R'
    ],
    'Cingulate Cortex': [
        'Cingulum_Ant_L', 'Cingulum_Ant_R',
        'Cingulum_Mid_L', 'Cingulum_Mid_R',
        'Cingulum_Post_L', 'Cingulum_Post_R'
    ],
    'Frontal Lobe': [
        'Frontal_Sup_L', 'Frontal_Sup_R',
        'Frontal_Mid_L', 'Frontal_Mid_R'
    ],
    'Visual Processing': [
        'Fusiform_L', 'Fusiform_R',
        'Lingual_L', 'Lingual_R'
    ]
}


def categorize_roi(roi_name):
    """將 ROI 分類到腦區類別"""
    for category, rois in AD_CRITICAL_REGIONS.items():
        if roi_name in rois:
            return category
    return 'Other'


def load_model_and_data():
    """載入最終模型和數據"""
    print("Loading final model and data...")
    
    # Load model
    model_path = Path('model/ml/final/final_model.pkl')
    scaler_path = Path('model/ml/final/final_scaler.pkl')
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"✓ Loaded model: {model.__class__.__name__}")
    
    # Load feature names
    with open('model/ml/final/final_feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    
    print(f"✓ Features: {len(feature_names)}")
    
    # Load data
    data_path = Path('data/processed/all_aal_roi_features.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Extract features
    X = df[feature_names].values
    y = df['label_id'].values
    subject_ids = df['subject_id'].values
    labels = df['label'].values
    
    print(f"✓ Samples: {len(df)} (NC: {(y==0).sum()}, AD: {(y==1).sum()})")
    
    return model, scaler, X, y, subject_ids, labels, feature_names


def test_model(model, scaler, X, y, subject_ids, labels):
    """測試模型並生成預測"""
    print("\n" + "="*80)
    print("Testing Final Model on All Data")
    print("="*80)
    
    # Standardize features
    X_scaled = scaler.transform(X)
    
    # Predict
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    
    # Calculate metrics
    accuracy = (y_pred == y).mean()
    
    print(f"\nOverall Accuracy: {accuracy:.3f} ({(y_pred == y).sum()}/{len(y)})")
    
    # Per-class accuracy
    nc_mask = (y == 0)
    ad_mask = (y == 1)
    
    nc_accuracy = (y_pred[nc_mask] == y[nc_mask]).mean()
    ad_accuracy = (y_pred[ad_mask] == y[ad_mask]).mean()
    
    print(f"\nNC Accuracy: {nc_accuracy:.3f} ({(y_pred[nc_mask] == y[nc_mask]).sum()}/{nc_mask.sum()})")
    print(f"AD Accuracy: {ad_accuracy:.3f} ({(y_pred[ad_mask] == y[ad_mask]).sum()}/{ad_mask.sum()})")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted NC    Predicted AD")
    print(f"Actual NC          {cm[0,0]:3d}             {cm[0,1]:3d}")
    print(f"Actual AD          {cm[1,0]:3d}             {cm[1,1]:3d}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['NC', 'AD']))
    
    # ROC-AUC
    roc_auc = roc_auc_score(y, y_proba[:, 1])
    print(f"ROC-AUC Score: {roc_auc:.3f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'subject_id': subject_ids,
        'true_label': labels,
        'predicted_label': ['AD' if p == 1 else 'NC' for p in y_pred],
        'confidence_nc': y_proba[:, 0],
        'confidence_ad': y_proba[:, 1],
        'confidence_max': y_proba.max(axis=1),
        'correct': y_pred == y
    })
    
    return results_df, cm, roc_auc


def analyze_feature_importance(model, feature_names):
    """分析特徵重要性"""
    print("\n" + "="*80)
    print("Feature Importance Analysis")
    print("="*80)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'ROI': feature_names,
        'Importance': importance,
        'Category': [categorize_roi(roi) for roi in feature_names]
    }).sort_values('Importance', ascending=False)
    
    # Print top 20
    print("\nTop 20 Most Important ROIs:")
    print("-"*80)
    for idx, row in importance_df.head(20).iterrows():
        marker = "★" if row['Category'] != 'Other' else " "
        print(f"{marker} {row['ROI']:30s} | {row['Category']:25s} | {row['Importance']:.4f}")
    
    print("\n★ = Critical AD Region")
    
    # Category summary
    print("\n" + "="*80)
    print("Importance by Brain Region Category")
    print("="*80)
    
    category_importance = importance_df.groupby('Category')['Importance'].agg(['sum', 'mean', 'count'])
    category_importance = category_importance.sort_values('sum', ascending=False)
    
    total_importance = importance_df['Importance'].sum()
    
    print(f"\n{'Category':<30s} {'Total':>10s} {'Mean':>10s} {'Count':>8s} {'%':>8s}")
    print("-"*80)
    for cat, row in category_importance.iterrows():
        pct = row['sum'] / total_importance * 100
        marker = "★" if cat != 'Other' else " "
        print(f"{marker} {cat:<28s} {row['sum']:>10.4f} {row['mean']:>10.4f} {row['count']:>8.0f} {pct:>7.1f}%")
    
    # Critical regions percentage
    critical_importance = importance_df[importance_df['Category'] != 'Other']['Importance'].sum()
    critical_pct = critical_importance / total_importance * 100
    
    print(f"\n{'='*80}")
    print(f"Critical AD Regions: {critical_pct:.1f}% of total importance")
    print(f"{'='*80}")
    
    if critical_pct >= 80:
        print("✓ EXCELLENT: Model heavily relies on AD-relevant regions")
    elif critical_pct >= 60:
        print("✓ GOOD: Model primarily uses AD-relevant regions")
    elif critical_pct >= 40:
        print("⚠ MODERATE: Model uses some AD-relevant regions")
    else:
        print("✗ POOR: Model relies mostly on non-AD regions")
    
    return importance_df


def visualize_results(results_df, cm, roc_auc, importance_df, output_dir):
    """視覺化測試結果"""
    print("\nGenerating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['NC', 'AD'], yticklabels=['NC', 'AD'])
    ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Confidence Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    correct = results_df[results_df['correct']]
    incorrect = results_df[~results_df['correct']]
    
    ax2.hist([correct['confidence_max'], incorrect['confidence_max']], 
             bins=20, label=['Correct', 'Incorrect'], 
             alpha=0.7, color=['green', 'red'])
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Per-Class Confidence
    ax3 = fig.add_subplot(gs[0, 2])
    nc_conf = results_df[results_df['true_label'] == 'NC']['confidence_nc']
    ad_conf = results_df[results_df['true_label'] == 'AD']['confidence_ad']
    
    bp = ax3.boxplot([nc_conf, ad_conf], labels=['NC', 'AD'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    ax3.set_ylabel('Confidence')
    ax3.set_title('Prediction Confidence by True Class', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Top 20 Feature Importance
    ax4 = fig.add_subplot(gs[1, :])
    top20 = importance_df.head(20)
    
    colors = ['red' if cat != 'Other' else 'gray' for cat in top20['Category']]
    y_pos = np.arange(len(top20))
    
    ax4.barh(y_pos, top20['Importance'], color=colors, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top20['ROI'], fontsize=8)
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 20 Feature Importance\nRed = Critical AD Regions', 
                 fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Category Importance
    ax5 = fig.add_subplot(gs[2, 0])
    category_imp = importance_df.groupby('Category')['Importance'].sum().sort_values(ascending=True)
    
    colors_cat = ['red' if cat != 'Other' else 'gray' for cat in category_imp.index]
    ax5.barh(range(len(category_imp)), category_imp.values, color=colors_cat, alpha=0.7)
    ax5.set_yticks(range(len(category_imp)))
    ax5.set_yticklabels(category_imp.index, fontsize=9)
    ax5.set_xlabel('Total Importance')
    ax5.set_title('Importance by Brain Region', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. ROC Curve
    ax6 = fig.add_subplot(gs[2, 1])
    from sklearn.metrics import roc_curve
    
    y_true = results_df['true_label'].map({'NC': 0, 'AD': 1}).values
    y_score = results_df['confidence_ad'].values
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    ax6.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax6.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax6.set_xlabel('False Positive Rate')
    ax6.set_ylabel('True Positive Rate')
    ax6.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Error Analysis
    ax7 = fig.add_subplot(gs[2, 2])
    
    errors = results_df[~results_df['correct']]
    if len(errors) > 0:
        error_types = {
            'False Positive\n(NC→AD)': len(errors[(errors['true_label'] == 'NC')]),
            'False Negative\n(AD→NC)': len(errors[(errors['true_label'] == 'AD')])
        }
        
        ax7.bar(error_types.keys(), error_types.values(), 
               color=['orange', 'red'], alpha=0.7)
        ax7.set_ylabel('Number of Errors')
        ax7.set_title('Error Types', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (k, v) in enumerate(error_types.items()):
            ax7.text(i, v, str(v), ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
    else:
        ax7.text(0.5, 0.5, '🎉 Perfect!\nNo Errors', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax7.set_xlim([0, 1])
        ax7.set_ylim([0, 1])
        ax7.axis('off')
    
    plt.savefig(output_dir / 'final_model_test_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'final_model_test_results.png'}")
    plt.close()


def generate_report(results_df, cm, roc_auc, importance_df, output_dir):
    """生成測試報告"""
    print("\nGenerating test report...")
    
    output_dir = Path(output_dir)
    report_path = output_dir / 'final_model_test_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Final Model Test Report\n")
        f.write("="*80 + "\n\n")
        
        # Overall performance
        accuracy = (results_df['correct']).mean()
        f.write("Overall Performance:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Samples: {len(results_df)}\n")
        f.write(f"Accuracy: {accuracy:.3f} ({results_df['correct'].sum()}/{len(results_df)})\n")
        f.write(f"ROC-AUC: {roc_auc:.3f}\n\n")
        
        # Per-class performance
        nc_results = results_df[results_df['true_label'] == 'NC']
        ad_results = results_df[results_df['true_label'] == 'AD']
        
        nc_acc = nc_results['correct'].mean()
        ad_acc = ad_results['correct'].mean()
        
        f.write("Per-Class Performance:\n")
        f.write("-"*80 + "\n")
        f.write(f"NC: {nc_acc:.3f} ({nc_results['correct'].sum()}/{len(nc_results)})\n")
        f.write(f"AD: {ad_acc:.3f} ({ad_results['correct'].sum()}/{len(ad_results)})\n\n")
        
        # Confusion matrix
        f.write("Confusion Matrix:\n")
        f.write("-"*80 + "\n")
        f.write(f"              Predicted NC    Predicted AD\n")
        f.write(f"Actual NC          {cm[0,0]:3d}             {cm[0,1]:3d}\n")
        f.write(f"Actual AD          {cm[1,0]:3d}             {cm[1,1]:3d}\n\n")
        
        # Confidence analysis
        f.write("Confidence Analysis:\n")
        f.write("-"*80 + "\n")
        f.write(f"Average Confidence: {results_df['confidence_max'].mean():.3f}\n")
        f.write(f"  Correct predictions: {results_df[results_df['correct']]['confidence_max'].mean():.3f}\n")
        
        if (~results_df['correct']).any():
            f.write(f"  Incorrect predictions: {results_df[~results_df['correct']]['confidence_max'].mean():.3f}\n")
        
        f.write("\n")
        
        # Feature importance
        f.write("="*80 + "\n")
        f.write("Feature Importance Analysis\n")
        f.write("="*80 + "\n\n")
        
        f.write("Top 20 Most Important ROIs:\n")
        f.write("-"*80 + "\n")
        
        for idx, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
            marker = "★" if row['Category'] != 'Other' else " "
            f.write(f"{marker} {idx:2d}. {row['ROI']:30s} | {row['Category']:25s} | {row['Importance']:.4f}\n")
        
        f.write("\n★ = Critical AD Region\n\n")
        
        # Category summary
        category_imp = importance_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)
        total_imp = importance_df['Importance'].sum()
        
        f.write("Importance by Brain Region Category:\n")
        f.write("-"*80 + "\n")
        
        for cat, imp in category_imp.items():
            pct = imp / total_imp * 100
            marker = "★" if cat != 'Other' else " "
            f.write(f"{marker} {cat:30s} {imp:.4f} ({pct:5.1f}%)\n")
        
        critical_imp = importance_df[importance_df['Category'] != 'Other']['Importance'].sum()
        critical_pct = critical_imp / total_imp * 100
        
        f.write(f"\nCritical AD Regions: {critical_pct:.1f}% of total importance\n\n")
        
        # Error analysis
        if (~results_df['correct']).any():
            f.write("="*80 + "\n")
            f.write("Error Analysis\n")
            f.write("="*80 + "\n\n")
            
            errors = results_df[~results_df['correct']]
            
            fp = errors[errors['true_label'] == 'NC']
            fn = errors[errors['true_label'] == 'AD']
            
            f.write(f"False Positives (NC predicted as AD): {len(fp)}\n")
            if len(fp) > 0:
                for _, row in fp.iterrows():
                    f.write(f"  - {row['subject_id']}: Confidence {row['confidence_ad']:.3f}\n")
            
            f.write(f"\nFalse Negatives (AD predicted as NC): {len(fn)}\n")
            if len(fn) > 0:
                for _, row in fn.iterrows():
                    f.write(f"  - {row['subject_id']}: Confidence {row['confidence_nc']:.3f}\n")
        else:
            f.write("="*80 + "\n")
            f.write("🎉 Perfect! No Errors!\n")
            f.write("="*80 + "\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved report: {report_path}")
    
    # Save results CSV
    csv_path = output_dir / 'final_model_test_predictions.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved predictions: {csv_path}")
    
    # Save importance CSV
    imp_csv_path = output_dir / 'final_model_test_importance.csv'
    importance_df.to_csv(imp_csv_path, index=False)
    print(f"✓ Saved importance: {imp_csv_path}")


def main():
    """主函數"""
    print("="*80)
    print("Testing Final Model")
    print("="*80)
    
    try:
        # Load model and data
        model, scaler, X, y, subject_ids, labels, feature_names = load_model_and_data()
        
        # Test model
        results_df, cm, roc_auc = test_model(model, scaler, X, y, subject_ids, labels)
        
        # Analyze feature importance
        importance_df = analyze_feature_importance(model, feature_names)
        
        # Visualize
        output_dir = Path('output/ml/final_model_test')
        visualize_results(results_df, cm, roc_auc, importance_df, output_dir)
        
        # Generate report
        generate_report(results_df, cm, roc_auc, importance_df, output_dir)
        
        print("\n" + "="*80)
        print("Testing Complete!")
        print("="*80)
        print(f"\nResults saved to: {output_dir}")
        
        # Print summary
        accuracy = results_df['correct'].mean()
        print("\nQuick Summary:")
        print("-"*80)
        print(f"Accuracy: {accuracy:.3f} ({results_df['correct'].sum()}/{len(results_df)})")
        print(f"ROC-AUC: {roc_auc:.3f}")
        
        critical_imp = importance_df[importance_df['Category'] != 'Other']['Importance'].sum()
        total_imp = importance_df['Importance'].sum()
        critical_pct = critical_imp / total_imp * 100
        print(f"Critical AD Regions: {critical_pct:.1f}% importance")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
