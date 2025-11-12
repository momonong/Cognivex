"""
Structural Visualizer Agent

This agent generates visualizations for structural MRI analysis:
1. Feature importance bar chart
2. 3D brain visualization with important ROIs highlighted
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
from app.graph.state import AgentState


def plot_feature_importance(
    importances: Dict[str, float],
    output_path: str,
    top_n: int = 10
) -> str:
    """
    Generate feature importance bar chart with Chinese labels
    
    Args:
        importances: Dictionary of ROI names to importance values
        output_path: Path to save the plot
        top_n: Number of top features to display
    
    Returns:
        Path to saved plot
    """
    # Import ROI name mapping
    try:
        from app.core.ml_processing.roi_names_zh import get_roi_display_name
    except:
        def get_roi_display_name(name, lang="zh"):
            return name
    
    # Sort and select top N
    sorted_features = sorted(
        importances.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    # Prepare data with Chinese names
    features_zh = [get_roi_display_name(f[0], "zh") for f in sorted_features]
    values = [f[1] * 100 for f in sorted_features]  # Convert to percentage
    
    # Create figure with Chinese font support
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    colors = sns.color_palette("RdYlBu_r", n_colors=top_n)
    bars = ax.barh(range(len(features_zh)), values, color=colors, height=0.7)
    
    # Customize plot
    ax.set_yticks(range(len(features_zh)))
    ax.set_yticklabels(features_zh, fontsize=11)
    ax.set_xlabel('重要性 (%)', fontsize=13, fontweight='bold')
    ax.set_title(
        f'前 {top_n} 個最重要腦區\nTop {top_n} Most Important Brain Regions',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            value + 0.3,
            i,
            f'{value:.2f}%',
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set x-axis limits
    ax.set_xlim(0, max(values) * 1.15)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_roi_on_brain(
    roi_importances: Dict[str, float],
    output_path: str,
    top_n: int = 10
) -> str:
    """
    Generate 3D brain visualization with important ROIs highlighted
    
    Note: This is a simplified version. Full implementation would use
    nilearn.plotting.plot_roi to overlay ROIs on MNI152 template.
    
    Args:
        roi_importances: Dictionary of ROI names to importance values
        output_path: Path to save the plot
        top_n: Number of top ROIs to highlight
    
    Returns:
        Path to saved plot
    """
    try:
        from nilearn import plotting, datasets
        from nilearn.image import new_img_like
        
        # Get top N ROIs
        sorted_rois = sorted(
            roi_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Load AAL atlas
        atlas = datasets.fetch_atlas_aal(version='SPM12')
        atlas_img = atlas['maps']
        atlas_labels = atlas['labels']
        
        # Create ROI mapping
        roi_mapping = {label: idx + 1 for idx, label in enumerate(atlas_labels)}
        
        # Create a mask image with importance values
        atlas_data = atlas_img.get_fdata()
        importance_map = np.zeros_like(atlas_data)
        
        for roi_name, importance in sorted_rois:
            if roi_name in roi_mapping:
                roi_idx = roi_mapping[roi_name]
                importance_map[atlas_data == roi_idx] = importance
        
        # Create new image
        importance_img = new_img_like(atlas_img, importance_map)
        
        # Create figure with multiple views
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f'Top {top_n} Important Brain Regions\n(Highlighted on MNI152 Template)',
            fontsize=14,
            fontweight='bold'
        )
        
        # Plot different views
        plotting.plot_stat_map(
            importance_img,
            title='Sagittal View',
            cut_coords=1,
            display_mode='x',
            colorbar=True,
            cmap='RdYlBu_r',
            axes=axes[0, 0],
            threshold=0.001
        )
        
        plotting.plot_stat_map(
            importance_img,
            title='Coronal View',
            cut_coords=1,
            display_mode='y',
            colorbar=True,
            cmap='RdYlBu_r',
            axes=axes[0, 1],
            threshold=0.001
        )
        
        plotting.plot_stat_map(
            importance_img,
            title='Axial View',
            cut_coords=1,
            display_mode='z',
            colorbar=True,
            cmap='RdYlBu_r',
            axes=axes[1, 0],
            threshold=0.001
        )
        
        # Add legend in the fourth subplot
        axes[1, 1].axis('off')
        legend_text = "Top ROIs:\n\n"
        for i, (roi, imp) in enumerate(sorted_rois[:5], 1):
            legend_text += f"{i}. {roi}\n   ({imp*100:.2f}%)\n"
        axes[1, 1].text(
            0.1, 0.5, legend_text,
            fontsize=10,
            verticalalignment='center',
            family='monospace'
        )
        
        # Save figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"⚠️  3D brain visualization failed: {e}")
        print("   Creating simplified visualization instead...")
        
        # Fallback: Create a simple text-based visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        sorted_rois = sorted(
            roi_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        text = f"Top {top_n} Important Brain Regions\n\n"
        for i, (roi, imp) in enumerate(sorted_rois, 1):
            text += f"{i:2d}. {roi:30s} {imp*100:6.2f}%\n"
        
        ax.text(
            0.5, 0.5, text,
            fontsize=12,
            verticalalignment='center',
            horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path


def generate_structural_visualizations(state: AgentState) -> dict:
    """
    Generate all visualizations for structural MRI analysis
    
    This agent:
    1. Creates feature importance bar chart
    2. Creates 3D brain visualization with ROIs
    3. Saves visualizations to output directory
    4. Records paths in state
    
    Args:
        state: AgentState containing feature_importances and subject_id
    
    Returns:
        Updated state dict with:
        - visualization_paths: List of paths to generated visualizations
        - feature_importance_plot_path: Path to importance chart
        - roi_visualization_path: Path to brain visualization
        - trace_log: Updated with processing steps
    """
    print("\n" + "="*60)
    print("AGENT: Structural Visualizer")
    print("="*60)
    
    subject_id = state.get('subject_id', 'unknown')
    feature_importances = state.get('feature_importances', {})
    
    if not feature_importances:
        error_msg = "No feature importances found for visualization"
        print(f"⚠️  {error_msg}")
        return {
            "visualization_paths": [],
            "trace_log": state.get("trace_log", []) + [error_msg]
        }
    
    try:
        # Create output directory
        output_dir = Path(f"output/ml_analysis/{subject_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Output directory: {output_dir}")
        
        visualization_paths = []
        
        # Step 1: Generate feature importance plot
        print("\n📊 Generating feature importance chart...")
        importance_plot_path = str(output_dir / "feature_importance.png")
        plot_feature_importance(
            feature_importances,
            importance_plot_path,
            top_n=10
        )
        visualization_paths.append(importance_plot_path)
        print(f"✓ Saved to: {importance_plot_path}")
        
        # Step 2: Generate brain ROI visualization
        print("\n🧠 Generating brain ROI visualization...")
        roi_viz_path = str(output_dir / "roi_visualization.png")
        plot_roi_on_brain(
            feature_importances,
            roi_viz_path,
            top_n=10
        )
        visualization_paths.append(roi_viz_path)
        print(f"✓ Saved to: {roi_viz_path}")
        
        trace_msg = f"Generated {len(visualization_paths)} visualizations for {subject_id}"
        print(f"\n✅ {trace_msg}")
        print("="*60 + "\n")
        
        return {
            "visualization_paths": visualization_paths,
            "feature_importance_plot_path": importance_plot_path,
            "roi_visualization_path": roi_viz_path,
            "trace_log": state.get("trace_log", []) + [trace_msg]
        }
        
    except Exception as e:
        error_msg = f"Visualization generation failed: {type(e).__name__}: {e}"
        print(f"❌ {error_msg}")
        print("="*60 + "\n")
        
        return {
            "visualization_paths": [],
            "error_log": state.get("error_log", []) + [error_msg],
            "trace_log": state.get("trace_log", []) + ["Visualization generation failed"]
        }
