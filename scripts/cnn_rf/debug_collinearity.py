"""
Debug Collinearity in ROI Features

This script checks for high correlation between features,
especially between different modalities (GM, FA, MD) of the same ROI.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_collinearity(csv_path="data/roi_features.csv", threshold=0.90):
    """
    Analyze collinearity in ROI features
    
    Args:
        csv_path: Path to ROI features CSV
        threshold: Correlation threshold to flag (default: 0.90)
    """
    print("="*80)
    print("Collinearity Analysis for ROI Features")
    print("="*80)
    
    # Load data
    print(f"\n[1/5] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Remove non-feature columns
    feature_cols = [col for col in df.columns if col not in ['Subject_ID', 'Group']]
    X = df[feature_cols]
    
    print(f"[OK] Loaded {len(df)} subjects with {len(feature_cols)} features")
    print(f"[OK] Groups: {df['Group'].value_counts().to_dict()}")
    
    # Calculate correlation matrix
    print(f"\n[2/5] Calculating correlation matrix...")
    corr_matrix = X.corr()
    print(f"[OK] Correlation matrix shape: {corr_matrix.shape}")
    
    # Find high correlations
    print(f"\n[3/5] Finding high correlations (threshold: {threshold})...")
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_value
                })
    
    print(f"[OK] Found {len(high_corr_pairs)} high correlation pairs")
    
    # Analyze same-ROI correlations
    print(f"\n[4/5] Analyzing same-ROI correlations (GM vs FA vs MD)...")
    same_roi_corr = []
    
    # Extract ROI names
    roi_names = set()
    for col in feature_cols:
        # Remove modality suffix (_GM, _FA, _MD)
        if col.endswith('_GM') or col.endswith('_FA') or col.endswith('_MD'):
            roi_name = col.rsplit('_', 1)[0]
            roi_names.add(roi_name)
    
    print(f"[OK] Found {len(roi_names)} unique ROIs")
    
    # Check correlations within each ROI
    for roi in sorted(roi_names):
        gm_col = f"{roi}_GM"
        fa_col = f"{roi}_FA"
        md_col = f"{roi}_MD"
        
        # Check if all modalities exist
        available = []
        if gm_col in feature_cols:
            available.append('GM')
        if fa_col in feature_cols:
            available.append('FA')
        if md_col in feature_cols:
            available.append('MD')
        
        if len(available) >= 2:
            # Calculate correlations between modalities
            if 'GM' in available and 'FA' in available:
                corr_gm_fa = corr_matrix.loc[gm_col, fa_col]
                same_roi_corr.append({
                    'ROI': roi,
                    'Pair': 'GM vs FA',
                    'Correlation': corr_gm_fa
                })
            
            if 'GM' in available and 'MD' in available:
                corr_gm_md = corr_matrix.loc[gm_col, md_col]
                same_roi_corr.append({
                    'ROI': roi,
                    'Pair': 'GM vs MD',
                    'Correlation': corr_gm_md
                })
            
            if 'FA' in available and 'MD' in available:
                corr_fa_md = corr_matrix.loc[fa_col, md_col]
                same_roi_corr.append({
                    'ROI': roi,
                    'Pair': 'FA vs MD',
                    'Correlation': corr_fa_md
                })
    
    same_roi_df = pd.DataFrame(same_roi_corr)
    
    # Print results
    print(f"\n[5/5] Results:")
    print("\n" + "="*80)
    print("HIGH CORRELATION PAIRS (|r| >= {:.2f})".format(threshold))
    print("="*80)
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        print(f"\nTotal: {len(high_corr_df)} pairs")
        print("\nTop 20:")
        for idx, row in high_corr_df.head(20).iterrows():
            print(f"  {row['Feature_1']:<40} <-> {row['Feature_2']:<40} r={row['Correlation']:+.4f}")
    else:
        print("\n✓ No high correlations found!")
    
    # Same-ROI analysis
    print("\n" + "="*80)
    print("SAME-ROI CORRELATIONS (GM vs FA vs MD)")
    print("="*80)
    
    if not same_roi_df.empty:
        # High correlations within same ROI
        high_same_roi = same_roi_df[abs(same_roi_df['Correlation']) >= threshold]
        
        if not high_same_roi.empty:
            print(f"\n⚠️  HIGH CORRELATIONS WITHIN SAME ROI (|r| >= {threshold}):")
            print(f"Total: {len(high_same_roi)} pairs\n")
            for idx, row in high_same_roi.iterrows():
                print(f"  {row['ROI']:<40} {row['Pair']:<15} r={row['Correlation']:+.4f}")
        else:
            print(f"\n✓ No high correlations within same ROI")
        
        # Statistics by pair type
        print(f"\n" + "-"*80)
        print("STATISTICS BY MODALITY PAIR:")
        print("-"*80)
        for pair_type in ['GM vs FA', 'GM vs MD', 'FA vs MD']:
            pair_data = same_roi_df[same_roi_df['Pair'] == pair_type]['Correlation']
            if len(pair_data) > 0:
                print(f"\n{pair_type}:")
                print(f"  Mean:   {pair_data.mean():+.4f}")
                print(f"  Median: {pair_data.median():+.4f}")
                print(f"  Std:    {pair_data.std():.4f}")
                print(f"  Min:    {pair_data.min():+.4f}")
                print(f"  Max:    {pair_data.max():+.4f}")
                print(f"  |r| >= {threshold}: {(abs(pair_data) >= threshold).sum()} / {len(pair_data)}")
    
    # Check specific AD-relevant regions
    print("\n" + "="*80)
    print("AD-RELEVANT REGIONS CHECK")
    print("="*80)
    
    ad_regions = [
        'Hippocampus_L', 'Hippocampus_R',
        'Amygdala_L', 'Amygdala_R',
        'Olfactory_L', 'Olfactory_R',
        'ParaHippocampal_L', 'ParaHippocampal_R',
        'Cingulate_Post_L', 'Cingulate_Post_R'
    ]
    
    print("\nChecking if AD-relevant regions exist in features:")
    for region in ad_regions:
        gm_exists = f"{region}_GM" in feature_cols
        fa_exists = f"{region}_FA" in feature_cols
        md_exists = f"{region}_MD" in feature_cols
        
        status = []
        if gm_exists:
            status.append('GM')
        if fa_exists:
            status.append('FA')
        if md_exists:
            status.append('MD')
        
        if status:
            print(f"  ✓ {region:<30} Available: {', '.join(status)}")
        else:
            print(f"  ✗ {region:<30} NOT FOUND")
    
    # Save results
    output_dir = Path("output/cnn_rf/collinearity_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if high_corr_pairs:
        high_corr_df.to_csv(output_dir / "high_correlations.csv", index=False)
        print(f"\n[SAVED] High correlations: {output_dir / 'high_correlations.csv'}")
    
    if not same_roi_df.empty:
        same_roi_df.to_csv(output_dir / "same_roi_correlations.csv", index=False)
        print(f"[SAVED] Same-ROI correlations: {output_dir / 'same_roi_correlations.csv'}")
    
    # Create correlation heatmap for top features
    print(f"\n[PLOT] Creating correlation heatmap...")
    
    # Get top 30 features by variance
    feature_vars = X.var().sort_values(ascending=False)
    top_features = feature_vars.head(30).index.tolist()
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_matrix.loc[top_features, top_features],
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Correlation Heatmap (Top 30 Features by Variance)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"[SAVED] Heatmap: {output_dir / 'correlation_heatmap.png'}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    
    return {
        'high_corr_pairs': high_corr_pairs,
        'same_roi_corr': same_roi_df,
        'corr_matrix': corr_matrix
    }


def main():
    """Main function"""
    results = analyze_collinearity(
        csv_path="data/roi_features.csv",
        threshold=0.90
    )
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if results['high_corr_pairs']:
        print("\n⚠️  HIGH COLLINEARITY DETECTED!")
        print("\nRecommended actions:")
        print("  1. Remove one feature from each highly correlated pair")
        print("  2. Apply PCA to reduce dimensionality")
        print("  3. Use regularization (L1/L2) in the model")
        print("  4. Consider training on GM features only")
    else:
        print("\n✓ No severe collinearity detected")
    
    # Check if AD regions are in top features
    print("\n" + "-"*80)
    print("Next steps:")
    print("  1. Run: python scripts/cnn_rf/debug_scaling.py")
    print("  2. Run: python scripts/cnn_rf/debug_biomarkers.py")
    print("="*80)


if __name__ == "__main__":
    main()
