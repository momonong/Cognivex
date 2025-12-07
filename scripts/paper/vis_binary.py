#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDDA Paper Visualization Script (V2)

Generates high-quality plots for IEEE Conference Paper based on the latest CSV schema.
1. Confidence vs. Uncertainty Scatter Plot (Visualizing OOD/MCI separation)
2. Agent Decision Distribution (Visualizing Workflow)
3. MCI Safety Net Performance (Pie Chart)

Usage:
    python scripts/paper/visualization_v2.py
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# --- Configuration ---
# 設定 IEEE 風格的繪圖參數 (更嚴謹的學術風格)
plt.rcParams.update({
    'font.family': 'serif',      # 使用襯線體 (如 Times New Roman) 配合 LaTeX
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,           # 高解析度
    'savefig.dpi': 300,
    'axes.grid': True,           # 預設開啟網格
    'grid.alpha': 0.3,
    'grid.linestyle': ':'
})

INPUT_CSV = Path("output/final_stats_v2/final_results.csv")
OUTPUT_DIR = Path("output/paper_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 定義統一的配色方案 (更專業的學術配色)
COLORS = {
    'NC': '#2ca02c',    # 綠色 (健康)
    'AD': '#d62728',    # 紅色 (疾病)
    'MCI': '#1f77b4',   # 藍色 (模糊/關注)
    'HighRisk': '#ff7f0e' # 橘色 (高風險區域)
}
DECISION_PALETTE = 'viridis_r' # 使用反向 Viridis，讓高強度的介入顏色更深

# --- Data Loading & Preprocessing ---
def load_and_preprocess_data():
    """
    Loads the results CSV and performs necessary preprocessing for plotting.
    """
    if not INPUT_CSV.exists():
        print(f"[!] Error: CSV not found at {INPUT_CSV}")
        print("    Please run the statistical analysis script first.")
        exit(1)
    
    df = pd.read_csv(INPUT_CSV)
    
    # [關鍵修改] 重新命名欄位以符合腳本邏輯
    df.rename(columns={'ground_truth': 'group'}, inplace=True)
    
    # 確保 group 的順序
    df['group'] = pd.Categorical(df['group'], categories=['NC', 'MCI', 'AD'], ordered=True)
    
    print(f"Loaded {len(df)} records. Columns: {list(df.columns)}")
    return df

# --- Plotting Functions ---

def plot_uncertainty_scatter(df):
    """
    Fig 3: Confidence vs Uncertainty Scatter Plot.
    Demonstrates that MCI cases naturally fall into the high-uncertainty/low-confidence zone.
    """
    print("[*] Generating Figure 3: Confidence vs Uncertainty...")
    
    plt.figure(figsize=(10, 7))
    
    # 1. 畫散佈圖
    sns.scatterplot(
        data=df, 
        x='confidence', 
        y='uq_score', 
        hue='group', 
        style='group',
        palette=COLORS,
        s=120,         # 點稍微大一點
        alpha=0.75,    # 透明度
        edgecolor='k', # 加上黑邊讓點更清楚
        linewidth=0.5
    )
    
    # 2. 定義閾值 (根據你的 Methodology)
    UQ_THRESHOLD = 0.6
    CONF_THRESHOLD = 0.7
    
    # 3. 畫出 "High Risk Zone" (高不確定性 OR 低信心度)
    # 這裡用填充區域來表示，視覺效果更好
    plt.axhline(y=UQ_THRESHOLD, color=COLORS['HighRisk'], linestyle='--', linewidth=2, label='_nolegend_')
    plt.axvline(x=CONF_THRESHOLD, color=COLORS['HighRisk'], linestyle='--', linewidth=2, label='_nolegend_')
    
    # 填充高風險區域 (UQ > 0.6 或 Conf < 0.7)
    # 使用 fill_between 來填充不規則區域有點複雜，這裡用簡單的矩形示意
    # 更精確的做法是填充整個 UQ > 0.6 的上半部，以及 Conf < 0.7 的左半部
    plt.fill_between([0, 1], UQ_THRESHOLD, 1.0, color=COLORS['HighRisk'], alpha=0.1, label='Agent Intervention Zone')
    plt.fill_between([0, CONF_THRESHOLD], 0, UQ_THRESHOLD, color=COLORS['HighRisk'], alpha=0.1)

    # 4. 加入註解
    plt.text(0.55, UQ_THRESHOLD + 0.02, f'High Uncertainty Threshold (>{UQ_THRESHOLD})', 
             color=COLORS['HighRisk'], fontsize=11, fontweight='bold')
    plt.text(CONF_THRESHOLD + 0.02, 0.1, f'Low Confidence Threshold (<{CONF_THRESHOLD})', 
             color=COLORS['HighRisk'], fontsize=11, fontweight='bold', rotation=90)

    # 5. 設定標題和標籤
    plt.title("Distribution of Diagnostic Confidence vs. Uncertainty", fontweight='bold')
    plt.xlabel("Perception Layer Confidence ($C$)")
    plt.ylabel("Uncertainty Score ($U$)")
    
    # 6. 優化圖例
    # 獲取當前的 handle 和 label
    handles, labels = plt.gca().get_legend_handles_labels()
    # 重新組織圖例，加入 High Risk Zone 的說明
    from matplotlib.patches import Patch
    risk_patch = Patch(facecolor=COLORS['HighRisk'], alpha=0.3, label='Agent Intervention Zone')
    # 只保留 group 的圖例，並加入風險區域圖例
    group_handles = handles[:3] 
    group_labels = labels[:3]
    plt.legend(handles=group_handles + [risk_patch], labels=group_labels + ['Agent Intervention Zone'], 
               title='Clinical Group', loc='upper left', frameon=True)

    plt.xlim(0.45, 1.02) # 稍微調整 X 軸範圍
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_confidence_uncertainty_v2.png")
    plt.savefig(OUTPUT_DIR / "fig3_confidence_uncertainty_v2.pdf")
    print("    -> Saved to output/paper_plots/fig3_confidence_uncertainty_v2.png")

def plot_agent_decisions(df):
    """
    Fig 4: Agent Decision Distribution Stacked Bar Chart.
    Visualizes how the reasoning layer treats different clinical groups.
    """
    print("[*] Generating Figure 4: Agent Decision Distribution...")
    
    # 1. 整理數據
    decision_counts = df.groupby(['group', 'agent_decision'], observed=True).size().reset_index(name='count')
    total_counts = df.groupby('group', observed=True).size().reset_index(name='total')
    decision_counts = decision_counts.merge(total_counts, on='group')
    decision_counts['percentage'] = (decision_counts['count'] / decision_counts['total']) * 100
    
    # 2. 簡化並排序 Decision 名稱
    decision_map = {
        'SIMULATION_TRIGGERED': '3. Counterfactual Simulation (High Intensity)',
        'ANOMALY_INVESTIGATION': '2. Anomaly Check (Medium Intensity)',
        'STANDARD_REPORT': '1. Standard Path (Low Intensity)'
    }
    decision_counts['decision_clean'] = decision_counts['agent_decision'].map(decision_map)
    
    # 確保 hue 的順序 (從低強度到高強度)
    hue_order = sorted(decision_counts['decision_clean'].unique())

    plt.figure(figsize=(10, 6))
    
    # 3. 畫堆疊長條圖 (Stacked Barplot 是比較好的選擇，但 seaborn 的 barplot 預設不是堆疊的)
    # 這裡用一個小技巧來實現堆疊效果：用 histplot
    # 或者堅持用 barplot，但要手動計算底部位置 (比較麻煩)
    # 這裡我們改用更直觀的 "百分比堆疊圖" (Percentage Stacked Bar Chart)
    
    # 使用 pivot table 重新塑形數據以方便堆疊繪圖
    df_pivot = decision_counts.pivot(index='group', columns='decision_clean', values='percentage').fillna(0)
    # 重新排序 columns
    df_pivot = df_pivot[hue_order]
    
    # 繪圖
    ax = df_pivot.plot(kind='bar', stacked=True, colormap=DECISION_PALETTE, figsize=(10, 6), edgecolor='k', width=0.7)
    
    # 4. 設定標題和標籤
    plt.title("Reasoning Layer Intervention by Clinical Group", fontweight='bold')
    plt.xlabel("Clinical Group")
    plt.ylabel("Percentage of Cases (%)")
    plt.xticks(rotation=0)
    
    # 5. 在長條圖上標註百分比
    for c in ax.containers:
        # 自定義標籤格式：只標註大於 5% 的數值
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 5 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', color='white', fontsize=10, fontweight='bold')

    # 6. 優化圖例
    # 反轉圖例順序，讓高強度的在上面，符合視覺直覺
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Reasoning Intensity', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_agent_decisions_v2.png")
    plt.savefig(OUTPUT_DIR / "fig4_agent_decisions_v2.pdf")
    print("    -> Saved to output/paper_plots/fig4_agent_decisions_v2.png")

def plot_mci_safety_net(df):
    """
    Fig 5: MCI Safety Net Efficacy Pie Chart.
    Highlights the system's ability to flag ambiguous MCI cases.
    """
    print("[*] Generating Figure 5: MCI Safety Net...")
    
    # 1. 只看 MCI
    mci_df = df[df['group'] == 'MCI']
    
    if len(mci_df) == 0:
        print("    [!] No MCI data found to plot.")
        return

    # 2. 計算 Flagged vs Missed
    flagged_count = mci_df['has_intervention'].sum()
    missed_count = len(mci_df) - flagged_count
    flagged_rate = (flagged_count / len(mci_df)) * 100
    
    # 3. 準備繪圖數據
    labels = [
        f'Flagged by Safety Net\n(Active Intervention)\nN={flagged_count}', 
        f'Passed to Baseline\n(No Intervention)\nN={missed_count}'
    ]
    sizes = [flagged_count, missed_count]
    # 使用高對比色：強調色(藍) vs 中性色(灰)
    pie_colors = [COLORS['MCI'], '#bdc3c7'] 
    explode = (0.05, 0)  # 稍微突出顯示 Flagged 部分
    
    plt.figure(figsize=(8, 8))
    
    # 4. 畫圓餅圖
    wedges, texts, autotexts = plt.pie(
        sizes, 
        explode=explode, 
        labels=labels, 
        colors=pie_colors, 
        autopct='%1.1f%%',
        pctdistance=0.8,     # 百分比標籤離圓心的距離
        labeldistance=1.1,   # 文字標籤離圓心的距離
        shadow=False, 
        startangle=90,       # 從正上方開始畫
        wedgeprops={'edgecolor': 'k', 'linewidth': 1.5, 'antialiased': True} # 加黑邊
    )
    
    # 5. 優化文字樣式
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        
    # 6. 加入中心文字 (顯示總介入率)
    centre_circle = plt.Circle((0,0),0.55,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.text(0, 0, f'{flagged_rate:.1f}%\nMCI Flagging\nRate', ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['MCI'])

    plt.title(f"MCI Safety Net Efficacy (Total MCI N={len(mci_df)})", fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_mci_safety_net_v2.png")
    plt.savefig(OUTPUT_DIR / "fig5_mci_safety_net_v2.pdf")
    print("    -> Saved to output/paper_plots/fig5_mci_safety_net_v2.png")

# --- Main execution ---
def main():
    print("="*60)
    print("NEURO-SYMBOLIC FRAMEWORK VISUALIZATION PIPELINE (V2)")
    print("="*60)
    
    try:
        # 1. Load Data
        df = load_and_preprocess_data()
        
        # 2. Generate Plots
        plot_uncertainty_scatter(df)
        plot_agent_decisions(df)
        plot_mci_safety_net(df)
        
        print("\n" + "="*60)
        print("SUCCESS: All paper plots generated in 'output/paper_plots/'")
        print("="*60)
        
    except Exception as e:
        print(f"\n[!] Visualization Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()