"""
UI components for the application
"""

from .structural_mri_components import (
    render_analysis_mode_selector,
    render_ml_model_info,
    render_structural_results,
    render_progress_indicator,
    render_error_message
)

from .brain_visualization import (
    render_brain_visualization,
    generate_brain_visualization
)

__all__ = [
    'render_analysis_mode_selector',
    'render_ml_model_info',
    'render_structural_results',
    'render_progress_indicator',
    'render_error_message',
    'render_brain_visualization',
    'generate_brain_visualization'
]
