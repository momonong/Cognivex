"""
Compare 24 selected ROIs vs All 116 ROIs using REAL extracted features
使用真實提取的特徵比較 24 個精選腦區 vs 全部 116 個腦區
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Use English for plots
plt.rcParams['font.family'] = 'sans-serif'

# 24 selected ROIs (current model)
SELECTED_24_ROIS = [
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
    'Frontal_Mid_L', 'Frontal_Mid_R'
]


def load_features(features_path):
    """載入特徵數據"""
    print(f"Loading features from: {features_path}")
    
    if not Path(features_path).exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    
    df = pd.read_csv(features_path)
    
    # Separate features and labels
    X = df.drop(['subject_id', 'label', 'label_id'], axis=1, errors='ignore')
    y = df['label_id'].values if 'label_id' in df.columns else df['label'].map({'NC': 0, 'AD': 1}).values
    feature_names = X.columns.tolist()
    
    print(f"✓ Loaded {len(df)} subjects")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ NC: {(y == 0).sum()}, AD: {(y == 1).sum()}")
    
    return X.values, y, feature_names, df


def train_and_evaluate(X, y, feature_names, model_name, n_folds=5):
    """
    訓練並評估模型
    
    Returns:
    --------
    results : dict
        包含各種評估指標的字典
    """
    print(f"\nTraining {model_name}...")
    
    # Create model
    if 'L1' in model_name or 'Logistic' in model_name:
        # Standardize for logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegressionCV(
            penalty='l1',
            solver='saga',
            cv=n_folds,
            max_iter=10000,
            random_state=42,
            n_jobs=-1
        )
        X_train = X_scaled
    else:
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        X_train = X
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Get multiple metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(model, X_train, y, cv=cv, scoring=scoring, return_train_score=True)
    
    # Train on full dataset
    model.fit(X_train, y)
    train_score = model.score(X_train, y)
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_[0])
        n_selected = np.sum(model.coef_[0] != 0)
    else:
        feature_importance = None
    
    # Compile results
    results = {
        'model_name': model_name,
        'n_features': X.shape[1],
        'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
        'cv_accuracy_std': cv_results['test_accuracy'].std(),
        'cv_precision_mean': cv_results['test_precision'].mean(),
        'cv_recall_mean': cv_results['test_recall'].mean(),
        'cv_f1_mean': cv_results['test_f1'].mean(),
        'cv_roc_auc_mean': cv_results['test_roc_auc'].mean(),
        'train_accuracy_mean': cv_results['train_accuracy'].mean(),
        'train_score': train_score,
        'overfitting_gap': train_score - cv_results['test_accuracy'].mean(),
        'feature_importance': feature_importance,
        'feature_names': feature_names,
        'model': model
    }
    
    if 'L1' in model_name or 'Logistic' in model_name:
        results['n_selected'] = n_selected
    
    # Print summary
    print(f"  CV Accuracy: {results['cv_accuracy_mean']:.3f} ± {results['cv_accuracy_std']:.3f}")
    print(f"  CV Precision: {results['cv_precision_mean']:.3f}")
    print(f"  CV Recall: {results['cv_recall_mean']:.3f}")
    print(f"  CV F1: {results['cv_f1_mean']:.3f}")
    print(f"  CV ROC-AUC: {results['cv_roc_auc_mean']:.3f}")
    print(f"  Train Accuracy: {results['train_score']:.3f}")
    print(f"  Overfitting Gap: {results['overfitting_gap']:.3f}")
    
    if 'n_selected' in results:
        print(f"  Features Selected: {results['n_selected']}/{X.shape[1]}")
    
    return results


def compare_approaches(X_all, y, all_feature_names):
    """比較不同的特徵選擇方法"""
    print("\n" + "="*80)
    print("Comparing Different Approaches")
    print("="*80)
    
    results = {}
    
    # 1. 24 Selected ROIs (if available)
    available_24 = [roi for roi in SELECTED_24_ROIS if roi in all_feature_names]
    
    if len(available_24) >= 20:  # At least 20 of the 24 ROIs available
        print(f"\n1. Using {len(available_24)} Selected ROIs (from original 24)")
        indices_24 = [all_feature_names.index(roi) for roi in available_24]
        X_24 = X_all[:, indices_24]
        results['Selected_ROIs'] = train_and_evaluate(
            X_24, y, available_24, f'Selected_{len(available_24)}_ROIs'
        )
    else:
        print(f"\n⚠ Only {len(available_24)}/24 selected ROIs available, skipping...")
    
    # 2. All 116 ROIs with Random Forest
    print(f"\n2. Using All {X_all.shape[1]} ROIs (Random Forest)")
    results['All_ROIs_RF'] = train_and_evaluate(
        X_all, y, all_feature_names, f'All_{X_all.shape[1]}_ROIs_RF'
    )
    
    # 3. All ROIs with L1 Regularization
    print(f"\n3. Using All {X_all.shape[1]} ROIs (L1 Regularization)")
    results['All_ROIs_L1'] = train_and_evaluate(
        X_all, y, all_feature_names, f'All_{X_all.shape[1]}_ROIs_L1'
    )
    
    # 4. Top 30 by Univariate Selection
    print(f"\n4. Selecting Top 30 ROIs (Univariate F-test)")
    selector = SelectKBest(f_classif, k=min(30, X_all.shape[1]))
    X_top30 = selector.fit_transform(X_all, y)
    selected_indices = selector.get_support()
    selected_names = [all_feature_names[i] for i, sel in enumerate(selected_indices) if sel]
    
    results['Top_30_Univariate'] = train_and_evaluate(
        X_top30, y, selected_names, 'Top_30_Univariate'
    )
    
    # 5. Top 30 by Mutual Information
    print(f"\n5. Selecting Top 30 ROIs (Mutual Information)")
    selector_mi = SelectKBest(mutual_info_classif, k=min(30, X_all.shape[1]))
    X_top30_mi = selector_mi.fit_transform(X_all, y)
    selected_indices_mi = selector_mi.get_support()
    selected_names_mi = [all_feature_names[i] for i, sel in enumerate(selected_indices_mi) if sel]
    
    results['Top_30_MI'] = train_and_evaluate(
        X_top30_mi, y, selected_names_mi, 'Top_30_MI'
    )
    
    return results


def visualize_comparison(results, output_dir):
    """視覺化比較結果"""
    print("\nGenerating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    approaches = list(results.keys())
    cv_acc = [results[k]['cv_accuracy_mean'] for k in approaches]
    cv_std = [results[k]['cv_accuracy_std'] for k in approaches]
    train_acc = [results[k]['train_score'] for k in approaches]
    gaps = [results[k]['overfitting_gap'] for k in approaches]
    n_features = [results[k]['n_features'] for k in approaches]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. CV Accuracy with error bars
    ax1 = fig.add_subplot(gs[0, :2])
    x_pos = np.arange(len(approaches))
    bars = ax1.bar(x_pos, cv_acc, yerr=cv_std, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(approaches, rotation=45, ha='right')
    ax1.set_ylabel('Cross-Validation Accuracy')
    ax1.set_title('CV Accuracy Comparison (with std dev)', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% threshold')
    ax1.legend()
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(cv_acc, cv_std)):
        ax1.text(i, mean + std + 0.02, f'{mean:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Multiple metrics comparison
    ax2 = fig.add_subplot(gs[0, 2])
    metrics_data = []
    for approach in approaches:
        metrics_data.append([
            results[approach]['cv_accuracy_mean'],
            results[approach]['cv_precision_mean'],
            results[approach]['cv_recall_mean'],
            results[approach]['cv_f1_mean']
        ])
    
    metrics_df = pd.DataFrame(
        metrics_data,
        columns=['Accuracy', 'Precision', 'Recall', 'F1'],
        index=approaches
    )
    
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlGnBu', 
                ax=ax2, cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
    ax2.set_title('Multiple Metrics Heatmap', fontsize=12, fontweight='bold')
    ax2.set_ylabel('')
    
    # 3. Train vs CV Accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    width = 0.35
    ax3.bar(x_pos - width/2, train_acc, width, label='Train', alpha=0.7, color='lightcoral')
    ax3.bar(x_pos + width/2, cv_acc, width, label='CV', alpha=0.7, color='steelblue')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(approaches, rotation=45, ha='right')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Train vs CV Accuracy', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.05])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Overfitting Gap
    ax4 = fig.add_subplot(gs[1, 1])
    colors = ['green' if gap < 0.1 else 'orange' if gap < 0.2 else 'red' for gap in gaps]
    bars = ax4.bar(x_pos, gaps, alpha=0.7, color=colors)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(approaches, rotation=45, ha='right')
    ax4.set_ylabel('Overfitting Gap (Train - CV)')
    ax4.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
    ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, gap in enumerate(gaps):
        ax4.text(i, gap + 0.01, f'{gap:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Feature Count vs Performance
    ax5 = fig.add_subplot(gs[1, 2])
    scatter = ax5.scatter(n_features, cv_acc, s=300, alpha=0.6, c=gaps, 
                         cmap='RdYlGn_r', edgecolors='black', linewidth=2)
    
    for i, approach in enumerate(approaches):
        ax5.annotate(approach, (n_features[i], cv_acc[i]), 
                    fontsize=8, ha='center', va='top', 
                    xytext=(0, -10), textcoords='offset points')
    
    ax5.set_xlabel('Number of Features')
    ax5.set_ylabel('CV Accuracy')
    ax5.set_title('Features vs Performance', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Overfitting Gap')
    
    # 6. Top 10 Features for each approach
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Create text summary of top features
    text_y = 0.95
    for approach in approaches:
        if 'feature_importance' in results[approach] and results[approach]['feature_importance'] is not None:
            importances = results[approach]['feature_importance']
            names = results[approach]['feature_names']
            
            # Get top 5
            top_idx = np.argsort(importances)[::-1][:5]
            top_features = [f"{names[i]} ({importances[i]:.3f})" for i in top_idx]
            
            ax6.text(0.02, text_y, f"{approach}:", 
                    fontsize=10, fontweight='bold', transform=ax6.transAxes)
            text_y -= 0.05
            
            for feat in top_features:
                ax6.text(0.05, text_y, f"• {feat}", 
                        fontsize=8, transform=ax6.transAxes)
                text_y -= 0.04
            
            text_y -= 0.02
    
    ax6.set_title('Top 5 Important Features by Approach', 
                 fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'real_feature_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'real_feature_comparison.png'}")
    plt.close()


def generate_report(results, output_dir):
    """生成詳細報告"""
    print("\nGenerating detailed report...")
    
    output_dir = Path(output_dir)
    report_path = output_dir / 'real_feature_comparison_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Real Feature Comparison Report\n")
        f.write("="*80 + "\n\n")
        
        f.write("Objective: Compare different feature selection strategies using REAL MRI data\n\n")
        
        # Summary table
        f.write("Performance Summary:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Approach':<25} {'CV Acc':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        f.write("-"*80 + "\n")
        
        for approach, res in results.items():
            f.write(f"{approach:<25} "
                   f"{res['cv_accuracy_mean']:.3f}±{res['cv_accuracy_std']:.3f}  "
                   f"{res['cv_precision_mean']:.3f}      "
                   f"{res['cv_recall_mean']:.3f}      "
                   f"{res['cv_f1_mean']:.3f}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Overfitting analysis
        f.write("Overfitting Analysis:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Approach':<25} {'Train Acc':<12} {'CV Acc':<12} {'Gap':<10} {'Status':<20}\n")
        f.write("-"*80 + "\n")
        
        for approach, res in results.items():
            gap = res['overfitting_gap']
            if gap < 0.1:
                status = "✓ Good"
            elif gap < 0.2:
                status = "⚠ Moderate"
            else:
                status = "✗ High"
            
            f.write(f"{approach:<25} "
                   f"{res['train_score']:.3f}      "
                   f"{res['cv_accuracy_mean']:.3f}      "
                   f"{gap:.3f}    "
                   f"{status}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Find best approach
        best_cv = max(results.items(), key=lambda x: x[1]['cv_accuracy_mean'])
        best_f1 = max(results.items(), key=lambda x: x[1]['cv_f1_mean'])
        lowest_gap = min(results.items(), key=lambda x: x[1]['overfitting_gap'])
        
        f.write("Key Findings:\n\n")
        f.write(f"1. Best CV Accuracy: {best_cv[0]} ({best_cv[1]['cv_accuracy_mean']:.3f})\n")
        f.write(f"2. Best F1 Score: {best_f1[0]} ({best_f1[1]['cv_f1_mean']:.3f})\n")
        f.write(f"3. Lowest Overfitting: {lowest_gap[0]} (gap={lowest_gap[1]['overfitting_gap']:.3f})\n\n")
        
        # Recommendations
        f.write("Recommendations:\n\n")
        
        # Compare selected vs all
        if 'Selected_ROIs' in results and 'All_ROIs_RF' in results:
            sel_acc = results['Selected_ROIs']['cv_accuracy_mean']
            all_acc = results['All_ROIs_RF']['cv_accuracy_mean']
            sel_gap = results['Selected_ROIs']['overfitting_gap']
            all_gap = results['All_ROIs_RF']['overfitting_gap']
            
            if sel_acc >= all_acc * 0.95 and sel_gap < all_gap:
                f.write("✓ RECOMMENDATION: Use Selected ROIs\n\n")
                f.write("Reasons:\n")
                f.write(f"  - Comparable accuracy ({sel_acc:.3f} vs {all_acc:.3f})\n")
                f.write(f"  - Lower overfitting risk (gap: {sel_gap:.3f} vs {all_gap:.3f})\n")
                f.write("  - Better interpretability\n")
                f.write("  - More clinically relevant\n")
            elif all_acc > sel_acc * 1.05:
                f.write("⚠ RECOMMENDATION: Consider using All ROIs with regularization\n\n")
                f.write("Reasons:\n")
                f.write(f"  - Significantly better accuracy ({all_acc:.3f} vs {sel_acc:.3f})\n")
                f.write("  - May capture additional information\n")
                f.write("  - Use L1 regularization to control overfitting\n")
            else:
                f.write("⚠ RECOMMENDATION: Both approaches are comparable\n\n")
                f.write("  - Consider ensemble of both models\n")
                f.write("  - Or use feature selection to find optimal subset\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved report: {report_path}")
    
    # Save CSV summary
    csv_path = output_dir / 'real_feature_comparison_summary.csv'
    summary_data = []
    
    for approach, res in results.items():
        summary_data.append({
            'Approach': approach,
            'N_Features': res['n_features'],
            'CV_Accuracy_Mean': res['cv_accuracy_mean'],
            'CV_Accuracy_Std': res['cv_accuracy_std'],
            'CV_Precision': res['cv_precision_mean'],
            'CV_Recall': res['cv_recall_mean'],
            'CV_F1': res['cv_f1_mean'],
            'CV_ROC_AUC': res['cv_roc_auc_mean'],
            'Train_Accuracy': res['train_score'],
            'Overfitting_Gap': res['overfitting_gap']
        })
    
    pd.DataFrame(summary_data).to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")


def main():
    """主函數"""
    print("="*80)
    print("Real Feature Comparison Experiment")
    print("="*80)
    
    try:
        # Check if features are extracted
        features_path = Path('data/processed/all_aal_roi_features.csv')
        
        if not features_path.exists():
            print(f"\n⚠ Feature file not found: {features_path}")
            print("\nPlease run feature extraction first:")
            print("  python scripts/ml/extract_all_roi_features.py")
            print("\nThis will extract features from all AAL ROIs in your MRI scans.")
            return
        
        # Load features
        X, y, feature_names, df = load_features(features_path)
        
        # Compare approaches
        results = compare_approaches(X, y, feature_names)
        
        # Visualize
        output_dir = Path('output/ml/real_feature_comparison')
        visualize_comparison(results, output_dir)
        
        # Generate report
        generate_report(results, output_dir)
        
        print("\n" + "="*80)
        print("Comparison Complete!")
        print("="*80)
        print(f"\nResults saved to: {output_dir}")
        
        # Print quick summary
        print("\nQuick Summary:")
        print("-"*80)
        for approach, res in results.items():
            print(f"{approach:25s}: CV={res['cv_accuracy_mean']:.3f}±{res['cv_accuracy_std']:.3f}, "
                  f"F1={res['cv_f1_mean']:.3f}, Gap={res['overfitting_gap']:.3f}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
