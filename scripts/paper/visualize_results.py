#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize CDDA Results for Paper

生成論文用的可視化圖表。

使用方法:
    python scripts/visualize_results.py --input output/paper_results
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("WARNING: matplotlib or pandas not installed.")
    print("Install with: pip install matplotlib pandas")


def load_metrics(metrics_dir: Path) -> List[Dict]:
    """加載所有性能指標文件"""
    metrics_files = list(metrics_dir.glob("metrics_*.json"))
    
    all_metrics = []
    for file in metrics_files:
        with open(file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            all_metrics.append(metrics)
    
    return all_metrics


def plot_prediction_distribution(metrics: List[Dict], output_dir: Path):
    """繪製預測分布圖"""
    predictions = {}
    for m in metrics:
        pred = m['prediction']
        predictions[pred] = predictions.get(pred, 0) + 1
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = list(predictions.keys())
    values = list(predictions.values())
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    bars = ax.bar(labels, values, color=colors[:len(labels)])
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Prediction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: prediction_distribution.png")


def plot_confidence_vs_uncertainty(metrics: List[Dict], output_dir: Path):
    """繪製信心度 vs 不確定性散點圖"""
    confidences = [m['confidence'] for m in metrics]
    uq_scores = [m['uq_score'] for m in metrics]
    predictions = [m['prediction'] for m in metrics]
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按預測類別著色
    color_map = {'AD': '#ff6b6b', 'MCI': '#4ecdc4', 'NC': '#45b7d1'}
    colors = [color_map.get(p, 'gray') for p in predictions]
    
    scatter = ax.scatter(confidences, uq_scores, c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    # 添加閾值線
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='UQ Threshold (0.8)')
    ax.axvline(x=0.6, color='orange', linestyle='--', alpha=0.5, label='Low Confidence (0.6)')
    
    ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Uncertainty Score (UQ)', fontsize=12, fontweight='bold')
    ax.set_title('Confidence vs Uncertainty', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # 添加圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6b6b', label='AD'),
        Patch(facecolor='#4ecdc4', label='MCI'),
        Patch(facecolor='#45b7d1', label='NC')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_vs_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: confidence_vs_uncertainty.png")


def plot_agent_decision_distribution(metrics: List[Dict], output_dir: Path):
    """繪製 Agent 決策分布圖"""
    decisions = {}
    for m in metrics:
        decision = m['agent_decision']
        decisions[decision] = decisions.get(decision, 0) + 1
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = [d.replace('_', '\n') for d in decisions.keys()]
    values = list(decisions.values())
    colors = ['#ff6b6b', '#feca57', '#48dbfb']
    
    bars = ax.bar(labels, values, color=colors[:len(labels)])
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Agent Decision', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Agent Decision Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'agent_decision_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: agent_decision_distribution.png")


def plot_performance_metrics(metrics: List[Dict], output_dir: Path):
    """繪製性能指標圖"""
    init_times = [m['performance']['init_time'] for m in metrics]
    analysis_times = [m['performance']['analysis_time'] for m in metrics]
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子圖 1: 時間分布箱型圖
    data = [init_times, analysis_times]
    labels = ['Initialization', 'Analysis']
    
    bp = ax1.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#4ecdc4', '#ff6b6b']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Time Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 子圖 2: 平均時間條形圖
    avg_init = np.mean(init_times)
    avg_analysis = np.mean(analysis_times)
    avg_total = avg_init + avg_analysis
    
    categories = ['Init', 'Analysis', 'Total']
    values = [avg_init, avg_analysis, avg_total]
    colors_bar = ['#4ecdc4', '#ff6b6b', '#45b7d1']
    
    bars = ax2.bar(categories, values, color=colors_bar)
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Time', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: performance_metrics.png")


def plot_top_features_heatmap(metrics: List[Dict], output_dir: Path):
    """繪製前 N 個特徵的熱圖"""
    # 收集所有受試者的前 5 個特徵
    all_features = {}
    
    for m in metrics:
        subject_id = m['subject_id']
        top_features = m['top_features']
        
        for feat in top_features:
            roi_name = feat['roi_name']
            shap_value = feat['shap_value']
            
            if roi_name not in all_features:
                all_features[roi_name] = []
            
            all_features[roi_name].append(shap_value)
    
    # 計算每個特徵的平均 SHAP 值
    avg_shap = {roi: np.mean(values) for roi, values in all_features.items()}
    
    # 選擇前 10 個最重要的特徵
    top_10 = sorted(avg_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    roi_names = [roi for roi, _ in top_10]
    shap_values = [shap for _, shap in top_10]
    
    colors = ['#ff6b6b' if s < 0 else '#4ecdc4' for s in shap_values]
    
    bars = ax.barh(roi_names, shap_values, color=colors)
    
    ax.set_xlabel('Average SHAP Value', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Most Important Features (Average SHAP)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_features_shap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: top_features_shap.png")


def plot_confusion_matrix(metrics: List[Dict], output_dir: Path):
    """繪製混淆矩陣 (如果有真實標籤)"""
    # 檢查是否有真實標籤
    has_ground_truth = any(m.get('ground_truth') for m in metrics)
    
    if not has_ground_truth:
        print("⚠ Skipped: confusion_matrix.png (no ground truth labels)")
        return
    
    # 收集預測和真實標籤
    y_true = []
    y_pred = []
    
    for m in metrics:
        if m.get('ground_truth'):
            y_true.append(m['ground_truth'])
            y_pred.append(m['prediction'])
    
    # 創建混淆矩陣
    from sklearn.metrics import confusion_matrix
    
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')
    
    # 添加數值標籤
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: confusion_matrix.png")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="Visualize CDDA Results for Paper"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='output/paper_results',
        help='輸入目錄 (paper_analysis.py 的輸出目錄)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='輸出目錄 (默認: input/visualizations)'
    )
    
    args = parser.parse_args()
    
    if not VISUALIZATION_AVAILABLE:
        print("\nERROR: Visualization libraries not available.")
        print("Install with: pip install matplotlib pandas scikit-learn")
        sys.exit(1)
    
    # 設置輸入輸出目錄
    input_dir = Path(args.input)
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir / 'visualizations'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 檢查輸入目錄
    metrics_dir = input_dir / 'metrics'
    
    if not metrics_dir.exists():
        print(f"ERROR: Metrics directory not found: {metrics_dir}")
        print("Please run paper_analysis.py first.")
        sys.exit(1)
    
    print("=" * 80)
    print("CDDA Results Visualization")
    print("=" * 80)
    print()
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 加載所有指標
    print("Loading metrics...")
    metrics = load_metrics(metrics_dir)
    print(f"Loaded {len(metrics)} subjects")
    print()
    
    if len(metrics) == 0:
        print("ERROR: No metrics files found.")
        sys.exit(1)
    
    # 生成圖表
    print("Generating visualizations...")
    print()
    
    plot_prediction_distribution(metrics, output_dir)
    plot_confidence_vs_uncertainty(metrics, output_dir)
    plot_agent_decision_distribution(metrics, output_dir)
    plot_performance_metrics(metrics, output_dir)
    plot_top_features_heatmap(metrics, output_dir)
    plot_confusion_matrix(metrics, output_dir)
    
    print()
    print("=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print()
    print(f"All figures saved to: {output_dir}")
    print()
    print("Generated figures:")
    print("  1. prediction_distribution.png")
    print("  2. confidence_vs_uncertainty.png")
    print("  3. agent_decision_distribution.png")
    print("  4. performance_metrics.png")
    print("  5. top_features_shap.png")
    print("  6. confusion_matrix.png (if ground truth available)")
    print()


if __name__ == "__main__":
    main()
