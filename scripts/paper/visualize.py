import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os

# 設定頂級期刊風格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

# 專業配色方案
colors_map = {
    'Standard': '#2ecc71',      # 綠色 (安全/標準)
    'Anomaly': '#f39c12',       # 橘色 (警告)
    'Simulation': '#e74c3c'     # 紅色 (高風險/模擬)
}

# 設定輸入輸出路徑 (Windows 風格)
INPUT_DIR = os.path.join('output', 'comprehensive_statistics')
OUTPUT_DIR = os.path.join('output', 'comprehensive_statistics')

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filename):
    filepath = os.path.join(INPUT_DIR, filename)
    print(f"Loading data from: {filepath}")
    return pd.read_csv(filepath)

def plot_confidence_uncertainty(df):
    """
    Figure 3: Confidence vs. Uncertainty Scatter Plot (獨立圖)
    """
    plt.figure(figsize=(10, 8))
    
    # 繪製散點
    sns.scatterplot(
        data=df, 
        x='confidence', 
        y='uq_score', 
        hue='correct', 
        style='agent_decision',
        palette={True: '#27ae60', False: '#c0392b'}, 
        s=100,
        alpha=0.7,
        legend='brief'
    )
    
    # 標記高風險區域 (High Risk Zone)
    # Low Conf < 0.6, High UQ > 0.8
    rect = Rectangle((0.0, 0.8), 0.6, 0.2, linewidth=2, edgecolor='#e74c3c', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect)
    # 文字移到右側
    plt.text(0.62, 0.85, 'High Risk Zone\n(Agent Intercepted)', 
             fontsize=12, color='#c0392b', ha='left', weight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1))

    plt.title('The "Trap": High Uncertainty in Low-Resource Data', fontweight='bold', fontsize=16)
    plt.xlabel('Model Confidence', fontweight='bold', fontsize=14)
    plt.ylabel('Uncertainty Score (UQ)', fontweight='bold', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 調整 Legend 位置
    plt.legend(loc='lower left', borderaxespad=1.)
    
    output_path = os.path.join(OUTPUT_DIR, 'fig_confidence_uncertainty.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated: {output_path}")

def plot_agent_analysis_composite(df):
    """
    Figure 4: Agent Analysis Composite (Pie + Bar)
    整合決策分佈與準確率分析
    """
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2]) # Bar chart 稍微寬一點
    
    # ==========================================
    # Left Panel: Agent Decision Distribution (Donut)
    # ==========================================
    ax1 = plt.subplot(gs[0])
    
    decision_counts = df['agent_decision'].value_counts()
    labels_map_pie = {
        'SIMULATION_TRIGGERED': 'Counterfactual\nSimulation',
        'ANOMALY_INVESTIGATION': 'Anomaly\nInvestigation',
        'STANDARD_REPORT': 'Standard\nReport'
    }
    
    sizes = [
        decision_counts.get('SIMULATION_TRIGGERED', 0),
        decision_counts.get('ANOMALY_INVESTIGATION', 0),
        decision_counts.get('STANDARD_REPORT', 0)
    ]
    labels = [labels_map_pie['SIMULATION_TRIGGERED'], labels_map_pie['ANOMALY_INVESTIGATION'], labels_map_pie['STANDARD_REPORT']]
    colors_pie = [colors_map['Simulation'], colors_map['Anomaly'], colors_map['Standard']]
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                       startangle=140, pctdistance=0.85, explode=(0.05, 0, 0))
    
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax1.add_artist(centre_circle)
    
    plt.setp(texts, size=11, weight="bold")
    plt.setp(autotexts, size=10, weight="bold", color="white")
    
    ax1.set_title('(A) Agent A Decision Efficiency', fontweight='bold', loc='left', fontsize=14)
    
    # ==========================================
    # Right Panel: Accuracy by Decision Path (Bar)
    # ==========================================
    ax2 = plt.subplot(gs[1])
    
    decision_acc = df.groupby('agent_decision')['correct'].mean() * 100
    categories = ['SIMULATION_TRIGGERED', 'ANOMALY_INVESTIGATION', 'STANDARD_REPORT']
    acc_values = [decision_acc.get(cat, 0) for cat in categories]
    counts = [decision_counts.get(cat, 0) for cat in categories]
    
    x_pos = range(len(categories))
    display_labels = ['Simulation\n(High Risk)', 'Anomaly\n(Medium Risk)', 'Standard\n(Low Risk)']
    
    bars = ax2.bar(x_pos, acc_values, color=[colors_map['Simulation'], colors_map['Anomaly'], colors_map['Standard']], width=0.6)
    
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Diagnostic Accuracy (%)', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(display_labels, fontweight='bold', fontsize=11)
    ax2.set_title('(B) Robustness across Risk Levels', fontweight='bold', loc='left', fontsize=14)
    
    for i, rect in enumerate(bars):
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height + 2, 
                 f'{height:.1f}%\n(N={counts[i]})',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    output_path = os.path.join(OUTPUT_DIR, 'fig_agent_analysis.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    try:
        # 使用 os.path.join 構建正確的讀取路徑
        csv_filename = 'comprehensive_statistics.csv'
        
        # 為了容錯，檢查檔案是否存在
        if not os.path.exists(os.path.join(INPUT_DIR, csv_filename)):
             # 如果不在 output/comprehensive_statistics 下，試試看當前目錄
             if os.path.exists(csv_filename):
                 INPUT_DIR = '.' # 切換到當前目錄
             else:
                 raise FileNotFoundError(f"Cannot find {csv_filename}")

        df = load_data(csv_filename)
        
        plot_confidence_uncertainty(df)
        plot_agent_analysis_composite(df)
        
        print("\nAll figures generated successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")