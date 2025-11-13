"""
Multi-modal ROI Feature Extraction Pipeline
多模態 ROI 特徵提取 Pipeline

完整的端到端 Pipeline，從多模態 MRI 影像到可解釋的分類結果。

主要組件:
- AAL116PatchExtractor: 3D Patch 提取器
- ResNet3D_Mini: 3D ResNet-10 Mini-CNN
- MultiModalFeatureExtractor: 多模態特徵提取器
- MultiModalROIDataset: PyTorch Dataset
- FeatureExtractionTrainer: 訓練器
- MultiModalROIPredictor: 推理器

快速開始:
    >>> from scripts.multimodal_roi import MultiModalROIPredictor
    >>> predictor = MultiModalROIPredictor(
    ...     feature_extractor_path='model/multimodal_roi/best_feature_extractor.pth',
    ...     xgboost_path='model/multimodal_roi/xgboost_classifier.pkl'
    ... )
    >>> result = predictor.predict(t1_path, t2_path, dwi_path)
"""

__version__ = "1.0.0"
__author__ = "Cognivex Team"

from .config import *
from .resnet3d_mini import ResNet3D_Mini, MultiModalFeatureExtractor
from .patch_extractor import AAL116PatchExtractor
from .dataset import MultiModalROIDataset, create_dataloaders
from .train import FeatureExtractionTrainer, extract_features_for_xgboost, train_xgboost_classifier
from .inference import MultiModalROIPredictor

__all__ = [
    # Configuration
    'DATA_ROOT',
    'MODEL_DIR',
    'OUTPUT_DIR',
    'DEVICE',
    
    # Models
    'ResNet3D_Mini',
    'MultiModalFeatureExtractor',
    
    # Data Processing
    'AAL116PatchExtractor',
    'MultiModalROIDataset',
    'create_dataloaders',
    
    # Training
    'FeatureExtractionTrainer',
    'extract_features_for_xgboost',
    'train_xgboost_classifier',
    
    # Inference
    'MultiModalROIPredictor',
]
