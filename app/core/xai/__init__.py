"""
XAI (Explainable AI) module for 3D CNN brain region visualization.

This module provides tools for:
- Activation extraction from 3D CNN models
- Grad-CAM heatmap generation
- Brain region mapping using anatomical atlases
- Quantitative analysis and visualization
"""

from .config_manager import ConfigManager
from .activation_extractor import ActivationExtractor
from .gradcam_generator import GradCAMGenerator

__all__ = ['ConfigManager', 'ActivationExtractor', 'GradCAMGenerator']
