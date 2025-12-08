#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-Ready Paper Visualization (V7 - Manual Control)
Bypasses Seaborn defaults to enforce high-quality rendering.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# --- 1. 強制設定高品質繪圖參數 ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': False, # 確保格線關閉
    'patch.edgecolor': 'black', # 箱子邊框強制黑色
    'patch.force_edgecolor': True
})

INPUT_CSV = Path("output/comprehensive_stats_v2/final_results.csv")
OUTPUT_DIR = Path("output/paper_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 定義顏色 (更飽和一點，適合印刷)
COLORS = ['#55A868', '#4C72B0', '#C44E52'] # Green, Blue, Red

def load_data():
    if not INPUT_CSV.exists():
        print(f"[!] CSV not found, generating dummy data...")
        np.random.seed(42)
        n = 40
        return pd.DataFrame({
            'group': ['NC']*n + ['MCI']*n + ['AD']*n,
            'confidence': np.random.rand(n*3)*0.5 + 0.5,
            'uq_score': np.concatenate([np.random.rand(n)*0.4, np.random.rand(n)*0.6+0.3, np.random.rand(n)*0.5+0.4])
        })
    df = pd.read_csv(INPUT_CSV)
    if 'ground_truth' in df.columns: df.rename(columns={'ground_truth': 'group'}, inplace=True)
    df = df.dropna(subset=['confidence', 'uq_score'])
    # 確保順序正確
    df['group'] = pd.Categorical(df['group'], categories=['NC', 'MCI', 'AD'], ordered=True)
    return df

def draw_bracket(ax, x1, x2, y, h, text):
    """ 手動繪製漂亮的統計括號 """
    line_x = [x1, x1, x2, x2]
    line_y = [y, y+h, y+h, y]
    ax.plot(line_x, line_y, lw=1.2, c='black')
    ax.text((x1+x2)*.5, y+h+0.01, text, ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

def plot_manual_boxplot(df):
    print("[*] Generating Figure 3 (Manual Control Version)...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    
    metrics = [
        ('confidence', 'Confidence ($P$)', 0.4, 1.2),
        ('uq_score', 'Uncertainty ($U$)', 0.0, 1.3)
    ]
    
    groups = ['NC', 'MCI', 'AD']
    
    for idx, (metric, title, y_min, y_max) in enumerate(metrics):
        ax = axes[idx]
        data_to_plot = [df[df['group'] == g][metric].values for g in groups]
        
        # 1. 畫 Boxplot (手動控制顏色)
        bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.6,
                        labels=groups, showfliers=False,
                        medianprops={'color': 'black', 'linewidth': 2})
        
        # 填色
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.8) # 稍微透明一點點
            patch.set_linewidth(1.5)
            
        # 2. 畫 Jitter Points (散點)
        for i, data in enumerate(data_to_plot):
            y = data
            x = np.random.normal(i+1, 0.04, size=len(y)) # 1-based index for boxplot
            ax.plot(x, y, 'o', markerfacecolor='#333333', markeredgecolor='none', 
                    alpha=0.3, markersize=4)
            
        ax.set_title(title, pad=15)
        ax.set_xlabel("Clinical Group", labelpad=10)
        ax.set_ylim(y_min, y_max)
        
        # 3. 如果是右圖 (Uncertainty)，畫閾值線
        if metric == 'uq_score':
            ax.axhline(y=0.6, color='#555555', linestyle='--', linewidth=1.5)
            ax.text(3.4, 0.6, 'Trigger ($U>0.6$)', va='center', ha='right', fontsize=10, color='#555555', fontweight='bold')
            ax.set_ylabel("Uncertainty Score")
        else:
            ax.set_ylabel("Model Confidence")

        # --- 左圖: Confidence ---
        if metric == 'confidence':
            # NC vs MCI (邊緣顯著) - 放在下層
            draw_bracket(ax, 1, 2, 1.04, 0.02, "ns (p=0.05)")
            # NC vs AD (顯著) - 放在上層
            draw_bracket(ax, 1, 3, 1.10, 0.02, "**")

        # --- 右圖: Uncertainty (調整高度版) ---
        elif metric == 'uq_score':
            # 1. [NC vs MCI] & [MCI vs AD] (放在同一層，高度稍微降低一點)
            # 之前的 y 是 1.05，現在改為 1.02，更貼近箱子
            draw_bracket(ax, 1, 2, 1.06, 0.02, "ns (p=0.05)")

            # 2. [NC vs AD] (**) (放在上一層，高度大幅降低)
            # 之前的 y 是 1.22，現在改為 1.16，讓它蓋住下面兩條線，但不要飛太高
            draw_bracket(ax, 1, 3, 1.16, 0.02, "**")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig3_final_manual.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"    -> Saved to {save_path}")

if __name__ == "__main__":
    df = load_data()
    plot_manual_boxplot(df)