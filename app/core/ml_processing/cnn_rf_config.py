"""
CNN-RF Model Configuration

Configuration for CNN-RF Random Forest models trained on AAL3 ROI features.
"""

from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class CNNRFModelConfig:
    """Configuration for CNN-RF model"""
    name: str
    model_path: Path
    classes: List[str]
    description: str
    data_root: Path
    roi_features_path: Path
    
    @property
    def is_available(self) -> bool:
        """Check if model file exists"""
        return self.model_path.exists()
    
    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)


# Data paths
DATA_ROOT = Path("data/MRI_processed")
ROI_FEATURES_PATH = Path("data/roi_features.csv")
MODEL_DIR = Path("model/cnn_rf")

# Available CNN-RF models
CNN_RF_MODELS: Dict[str, CNNRFModelConfig] = {
    "NC_vs_AD": CNNRFModelConfig(
        name="NC_vs_AD",
        model_path=MODEL_DIR / "rf_model_NC_vs_AD.joblib",
        classes=['AD', 'NC'],
        description="Binary classification: Normal Control vs Alzheimer's Disease (All features)",
        data_root=DATA_ROOT,
        roi_features_path=ROI_FEATURES_PATH
    ),
    "NC_vs_AD_GM": CNNRFModelConfig(
        name="NC_vs_AD_GM",
        model_path=MODEL_DIR / "rf_model_NC_vs_AD_GM_only.joblib",
        classes=['AD', 'NC'],
        description="Binary classification: Normal Control vs Alzheimer's Disease (GM only) - Recommended",
        data_root=DATA_ROOT,
        roi_features_path=ROI_FEATURES_PATH
    ),
    "NC_MCI_AD": CNNRFModelConfig(
        name="NC_MCI_AD",
        model_path=MODEL_DIR / "rf_model_NC_MCI_AD.joblib",
        classes=['AD', 'MCI', 'NC'],
        description="Three-way classification: NC vs MCI vs AD",
        data_root=DATA_ROOT,
        roi_features_path=ROI_FEATURES_PATH
    )
}

# Default model - Use GM-only model (recommended)
DEFAULT_CNN_RF_MODEL = "NC_vs_AD_GM"


def get_cnn_rf_config(model_name: str = None) -> CNNRFModelConfig:
    """
    Get CNN-RF model configuration
    
    Args:
        model_name: Model name ('NC_vs_AD' or 'NC_MCI_AD')
                   If None, returns default model
    
    Returns:
        CNNRFModelConfig object
    
    Raises:
        ValueError: If model name is unknown
    """
    if model_name is None:
        model_name = DEFAULT_CNN_RF_MODEL
    
    if model_name not in CNN_RF_MODELS:
        available = list(CNN_RF_MODELS.keys())
        raise ValueError(
            f"Unknown CNN-RF model: {model_name}. "
            f"Available models: {available}"
        )
    
    return CNN_RF_MODELS[model_name]


def list_available_models() -> List[str]:
    """List all available CNN-RF models"""
    return [
        name for name, config in CNN_RF_MODELS.items()
        if config.is_available
    ]


def print_model_info(model_name: str = None):
    """Print information about a CNN-RF model"""
    config = get_cnn_rf_config(model_name)
    
    print("="*80)
    print(f"CNN-RF Model: {config.name}")
    print("="*80)
    print(f"Description: {config.description}")
    print(f"Classes: {', '.join(config.classes)}")
    print(f"Model path: {config.model_path}")
    print(f"Available: {'✓' if config.is_available else '✗'}")
    print(f"Data root: {config.data_root}")
    print(f"ROI features: {config.roi_features_path}")
    print("="*80)


if __name__ == "__main__":
    # Print info for all models
    for model_name in CNN_RF_MODELS.keys():
        print_model_info(model_name)
        print()
