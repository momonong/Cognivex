"""
Configuration classes for ML model integration
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MLModelConfig:
    """Configuration for ML model and feature extraction"""
    
    # Model paths
    model_type: str  # "random_forest"
    model_path: str
    scaler_path: str
    roi_list_path: str
    feature_names_path: str
    
    # ROI extraction configuration
    atlas_name: str = "AAL"
    num_features: int = 32
    
    # Visualization configuration
    top_n_features: int = 10
    colormap: str = "RdYlBu_r"
    
    # Output configuration
    output_dir: Optional[str] = None
    
    @classmethod
    def from_directory(cls, model_dir: str = "model/ml/final") -> "MLModelConfig":
        """
        Create configuration from model directory
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            MLModelConfig instance
        """
        model_dir_path = Path(model_dir)
        
        return cls(
            model_type="random_forest",
            model_path=str(model_dir_path / "final_model.pkl"),
            scaler_path=str(model_dir_path / "final_scaler.pkl"),
            roi_list_path=str(model_dir_path / "final_roi_list.csv"),
            feature_names_path=str(model_dir_path / "final_feature_names.txt"),
            output_dir=f"output/ml_analysis"
        )
    
    def validate(self) -> bool:
        """
        Validate that all required files exist
        
        Returns:
            True if all files exist
            
        Raises:
            FileNotFoundError: If any required file is missing
        """
        required_files = [
            self.model_path,
            self.scaler_path,
            self.roi_list_path,
            self.feature_names_path
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        return True


# Default configuration instance
ML_MODEL_CONFIG = {
    "model_dir": "model/ml/final",
    "model_type": "random_forest",
    "atlas_name": "AAL",
    "num_features": 32,
    "top_n_features": 10,
    "colormap": "RdYlBu_r",
    "output_dir": "output/ml_analysis"
}


def get_default_config() -> MLModelConfig:
    """
    Get default ML model configuration
    
    Returns:
        MLModelConfig instance with default settings
    """
    return MLModelConfig.from_directory(ML_MODEL_CONFIG["model_dir"])
