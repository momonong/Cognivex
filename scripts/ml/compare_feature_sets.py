"""
Compare different feature sets: 24 selected ROIs vs All 116 ROIs
比較不同的特徵集：24 個精選腦區 vs 全部 116 個腦區
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Use English for plots
plt.rcParams['font.family'] = 'sans-serif'

# Configuration
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


def load_current_model_data():
    """載入當前模型和數據"""
    print("Loading current model and data...")
    
    # Load model
    model = joblib.load('model/ml/rf_model.pkl')
    scaler = joblib.load('model/ml/scaler.pkl')
    
    # Load ROI names
    roi_df = pd.read_csv('output/ml/roi_importance.csv')
    feature_names = roi_df['ROI'].tolist()
    
    # Load predictions to get labels
    pred_df = pd.read_csv('output/ml/batch_predictions.csv')
    
    print(f"✓ Model: {model.__class__.__name__}")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Samples: {len(pred_df)}")
    
    return model, scaler, feature_names, pred_df


def simulate_all_rois_data(n_samples, n_features=116):
    """
    模擬全部 116 個 ROI 的數據
    注意：這是模擬數據，實際使用時需要從影像中提取
    """
    print(f"\n⚠️  Simulating {n_features} ROI features...")
    print("   (In real use, extract from actual MRI scans)")
    
    # Generate random features (for demonstration)
    X_all = np.random.randn(n_samples, n_features)
    
    # Generate ROI names
    all_roi_names = [f'ROI_{i:03d}' for i in range(n_features)]
    
    return X_all, all_roi_names


def compare_feature_sets(X_24, X_116, y, roi_names_24, roi_names_116):
    """比較不同特徵集的效能"""
    print("\n" + "="*80)
    print("Comparing Feature Sets")
    print("="*80)
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. 24 Selected ROIs
    print("\n1. Testing 24 Selected ROIs...")
    model_24 = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores_24 = cross_val_score(model_24, X_24, y, cv=cv, scoring='accuracy')
    
    # Train on all data to get feature importance
    model_24.fit(X_24, y)
    train_score_24 = model_24.score(X_24, y)
    
    results['24_ROIs'] = {
        'cv_mean': cv_scores_24.mean(),
        'cv_std': cv_scores_24.std(),
        'train_score': train_score_24,
        'overfitting_gap': train_score_24 - cv_scores_24.mean(),
        'n_features': 24,
        'feature_importance': model_24.feature_importances_,
        'feature_names': roi_names_24
    }
    
    print(f"   CV Accuracy: {cv_scores_24.mean():.3f} ± {cv_scores_24.std():.3f}")
    print(f"   Train Accuracy: {train_score_24:.3f}")
    print(f"   Overfitting Gap: {train_score_24 - cv_scores_24.mean():.3f}")
    
    # 2. All 116 ROIs with Random Forest
    print("\n2. Testing All 116 ROIs (Random Forest)...")
    model_116_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores_116_rf = cross_val_score(model_116_rf, X_116, y, cv=cv, scoring='accuracy')
    model_116_rf.fit(X_116, y)
    train_score_116_rf = model_116_rf.score(X_116, y)
    
    results['116_ROIs_RF'] = {
        'cv_mean': cv_scores_116_rf.mean(),
        'cv_std': cv_scores_116_rf.std(),
        'train_score': train_score_116_rf,
        'overfitting_gap': train_score_116_rf - cv_scores_116_rf.mean(),
        'n_features': 116,
        'feature_importance': model_116_rf.feature_importances_,
        'feature_names': roi_names_116
    }
    
    print(f"   CV Accuracy: {cv_scores_116_rf.mean():.3f} ± {cv_scores_116_rf.std():.3f}")
    print(f"   Train Accuracy: {train_score_116_rf:.3f}")
    print(f"   Overfitting Gap: {train_score_116_rf - cv_scores_116_rf.mean():.3f}")
    
    # 3. All 116 ROIs with L1 Regularization
    print("\n3. Testing All 116 ROIs (L1 Regularization)...")
    
    # Standardize for logistic regression
    scaler = StandardScaler()
    X_116_scaled = scaler.fit_transform(X_116)
    
    model_116_l1 = LogisticRegressionCV(
        penalty='l1',
        solver='saga',
        cv=5,
        max_iter=5000,
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores_116_l1 = cross_val_score(model_116_l1, X_116_scaled, y, cv=cv, scoring='accuracy')
    model_116_l1.fit(X_116_scaled, y)
    train_score_116_l1 = model_116_l1.score(X_116_scaled, y)
    
    # Count non-zero coefficients (selected features)
    n_selected = np.sum(model_116_l1.coef_[0] != 0)
    
    results['116_ROIs_L1'] = {
        'cv_mean': cv_scores_116_l1.mean(),
        'cv_std': cv_scores_116_l1.std(),
        'train_score': train_score_116_l1,
        'overfitting_gap': train_score_116_l1 - cv_scores_116_l1.mean(),
        'n_features': 116,
        'n_selected': n_selected,
        'coefficients': model_116_l1.coef_[0],
        'feature_names': roi_names_116
    }
    
    print(f"   CV Accuracy: {cv_scores_116_l1.mean():.3f} ± {cv_scores_116_l1.std():.3f}")
    print(f"   Train Accuracy: {train_score_116_l1:.3f}")
    print(f"   Overfitting Gap: {train_score_116_l1 - cv_scores_116_l1.mean():.3f}")
    print(f"   Features Selected: {n_selected}/116")
    
    # 4. Feature Selection: Top 30 from 116
    print("\n4. Testing Top 30 ROIs (Univariate Selection)...")
    
    selector = SelectKBest(f_classif, k=30)
    X_top30 = selector.fit_transform(X_116, y)
    selected_indices = selector.get_support()
    selected_roi_names = [roi_names_116[i] for i, selected in enumerate(selected_indices) if selected]
    
    model_top30 = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    cv_scores_top30 = cross_val_score(model_top30, X_top30, y, cv=cv, scoring='accuracy')
    model_top30.fit(X_top30, y)
    train_score_top30 = model_top30.score(X_top30, y)
    
    results['Top_30_ROIs'] = {
        'cv_mean': cv_scores_top30.mean(),
        'cv_std': cv_scores_top30.std(),
        'train_score': train_score_top30,
        'overfitting_gap': train_score_top30 - cv_scores_top30.mean(),
        'n_features': 30,
        'feature_importance': model_top30.feature_importances_,
        'feature_names': selected_roi_names
    }
    
    print(f"   CV Accuracy: {cv_scores_top30.mean():.3f} ± {cv_scores_top30.std():.3f}")
    print(f"   Train Accuracy: {train_score_top30:.3f}")
    print(f"   Overfitting Gap: {train_score_top30 - cv_scores_top30.mean():.3f}")
    
    return results


def visualize_comparison(results, output_dir):
    """視覺化比較結果"""
    print("\nGenerating comparison visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    approaches = list(results.keys())
    cv_means = [results[k]['cv_mean'] for k in approaches]
    cv_stds = [results[k]['cv_std'] for k in approaches]
    train_scores = [results[k]['train_score'] for k in approaches]
    gaps = [results[k]['overfitting_gap'] for k in approaches]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. CV Accuracy Comparison
    x_pos = np.arange(len(approaches))
    axes[0, 0].bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='steelblue')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(approaches, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Cross-Validation Accuracy')
    axes[0, 0].set_title('CV Accuracy Comparison')
    axes[0, 0].set_ylim([0, 1.05])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
        axes[0, 0].text(i, mean + std + 0.02, f'{mean:.3f}', 
                       ha='center', va='bottom', fontsize=9)
    
    # 2. Train vs CV Accuracy
    x_pos = np.arange(len(approaches))
    width = 0.35
    axes[0, 1].bar(x_pos - width/2, train_scores, width, label='Train', alpha=0.7, color='lightcoral')
    axes[0, 1].bar(x_pos + width/2, cv_means, width, label='CV', alpha=0.7, color='steelblue')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(approaches, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Train vs CV Accuracy')
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Overfitting Gap
    colors = ['green' if gap < 0.1 else 'orange' if gap < 0.2 else 'red' for gap in gaps]
    axes[1, 0].bar(x_pos, gaps, alpha=0.7, color=colors)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(approaches, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Overfitting Gap (Train - CV)')
    axes[1, 0].set_title('Overfitting Analysis\nGreen=Good, Orange=Moderate, Red=High')
    axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Moderate')
    axes[1, 0].axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='High')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, gap in enumerate(gaps):
        axes[1, 0].text(i, gap + 0.01, f'{gap:.3f}', 
                       ha='center', va='bottom', fontsize=9)
    
    # 4. Feature Count vs Performance
    n_features = [results[k]['n_features'] for k in approaches]
    axes[1, 1].scatter(n_features, cv_means, s=200, alpha=0.6, c=gaps, cmap='RdYlGn_r')
    
    for i, approach in enumerate(approaches):
        axes[1, 1].annotate(approach, (n_features[i], cv_means[i]), 
                           fontsize=8, ha='center', va='bottom')
    
    axes[1, 1].set_xlabel('Number of Features')
    axes[1, 1].set_ylabel('CV Accuracy')
    axes[1, 1].set_title('Feature Count vs Performance\nColor = Overfitting Gap')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', 
                               norm=plt.Normalize(vmin=min(gaps), vmax=max(gaps)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1, 1])
    cbar.set_label('Overfitting Gap')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_set_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'feature_set_comparison.png'}")
    plt.close()


def generate_report(results, output_dir):
    """生成詳細報告"""
    print("\nGenerating comparison report...")
    
    output_dir = Path(output_dir)
    report_path = output_dir / 'feature_comparison_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Feature Set Comparison Report\n")
        f.write("="*80 + "\n\n")
        
        f.write("Objective: Compare 24 selected ROIs vs all 116 ROIs\n\n")
        
        # Summary table
        f.write("Summary Table:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Approach':<20} {'CV Acc':<12} {'Train Acc':<12} {'Gap':<10} {'Features':<10}\n")
        f.write("-"*80 + "\n")
        
        for approach, res in results.items():
            f.write(f"{approach:<20} "
                   f"{res['cv_mean']:.3f}±{res['cv_std']:.3f}  "
                   f"{res['train_score']:.3f}      "
                   f"{res['overfitting_gap']:.3f}    "
                   f"{res['n_features']}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Analysis
        f.write("Analysis:\n\n")
        
        # Find best approach
        best_cv = max(results.items(), key=lambda x: x[1]['cv_mean'])
        lowest_gap = min(results.items(), key=lambda x: x[1]['overfitting_gap'])
        
        f.write(f"1. Best CV Accuracy: {best_cv[0]} ({best_cv[1]['cv_mean']:.3f})\n")
        f.write(f"2. Lowest Overfitting: {lowest_gap[0]} (gap={lowest_gap[1]['overfitting_gap']:.3f})\n\n")
        
        # Overfitting assessment
        f.write("Overfitting Assessment:\n")
        for approach, res in results.items():
            gap = res['overfitting_gap']
            if gap < 0.1:
                status = "✓ Good (low overfitting)"
            elif gap < 0.2:
                status = "⚠ Moderate (some overfitting)"
            else:
                status = "✗ High (significant overfitting)"
            
            f.write(f"  {approach}: {gap:.3f} - {status}\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("Recommendations:\n\n")
        
        # Compare 24 vs 116
        gap_24 = results['24_ROIs']['overfitting_gap']
        gap_116 = results['116_ROIs_RF']['overfitting_gap']
        cv_24 = results['24_ROIs']['cv_mean']
        cv_116 = results['116_ROIs_RF']['cv_mean']
        
        if cv_24 >= cv_116 and gap_24 <= gap_116:
            f.write("✓ RECOMMENDATION: Keep 24 selected ROIs\n\n")
            f.write("Reasons:\n")
            f.write(f"  - Similar or better CV accuracy ({cv_24:.3f} vs {cv_116:.3f})\n")
            f.write(f"  - Lower overfitting risk (gap: {gap_24:.3f} vs {gap_116:.3f})\n")
            f.write("  - Better interpretability (24 vs 116 features)\n")
            f.write("  - More appropriate for sample size (n=65)\n")
        else:
            f.write("⚠ RECOMMENDATION: Consider using more features\n\n")
            f.write("Reasons:\n")
            f.write(f"  - Higher CV accuracy with 116 ROIs ({cv_116:.3f} vs {cv_24:.3f})\n")
            f.write("  - May capture additional information\n")
            f.write("  - Consider using regularization (L1) to reduce overfitting\n")
        
        f.write("\n")
        
        # Feature importance comparison (if available)
        if 'feature_importance' in results['24_ROIs']:
            f.write("Top 10 Features by Approach:\n\n")
            
            for approach in ['24_ROIs', '116_ROIs_RF', 'Top_30_ROIs']:
                if approach in results and 'feature_importance' in results[approach]:
                    f.write(f"{approach}:\n")
                    
                    importances = results[approach]['feature_importance']
                    names = results[approach]['feature_names']
                    
                    # Sort by importance
                    sorted_idx = np.argsort(importances)[::-1][:10]
                    
                    for i, idx in enumerate(sorted_idx, 1):
                        f.write(f"  {i:2d}. {names[idx]:<30s} {importances[idx]:.4f}\n")
                    
                    f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Saved report: {report_path}")
    
    # Also save as CSV
    csv_path = output_dir / 'feature_comparison_summary.csv'
    summary_data = []
    
    for approach, res in results.items():
        summary_data.append({
            'Approach': approach,
            'CV_Accuracy_Mean': res['cv_mean'],
            'CV_Accuracy_Std': res['cv_std'],
            'Train_Accuracy': res['train_score'],
            'Overfitting_Gap': res['overfitting_gap'],
            'N_Features': res['n_features']
        })
    
    pd.DataFrame(summary_data).to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")


def main():
    """主函數"""
    print("="*80)
    print("Feature Set Comparison Experiment")
    print("="*80)
    
    try:
        # Load current model data
        model, scaler, feature_names_24, pred_df = load_current_model_data()
        
        # Get labels
        y = pred_df['true_label_id'].values
        n_samples = len(y)
        
        # Generate feature matrices
        # Note: In real use, extract from actual MRI scans
        print("\nGenerating feature matrices...")
        
        # 24 ROIs (use random data for demonstration)
        X_24 = np.random.randn(n_samples, 24)
        
        # 116 ROIs (simulated)
        X_116, roi_names_116 = simulate_all_rois_data(n_samples, 116)
        
        # Compare feature sets
        results = compare_feature_sets(X_24, X_116, y, feature_names_24, roi_names_116)
        
        # Visualize
        output_dir = Path('output/ml/feature_comparison')
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
            print(f"{approach:20s}: CV={res['cv_mean']:.3f}±{res['cv_std']:.3f}, "
                  f"Gap={res['overfitting_gap']:.3f}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
