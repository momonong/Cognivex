"""
Model Configuration System for fMRI Processing Pipeline

This module provides a unified configuration interface for different model types,
allowing the pipeline to work with various neural network architectures without
hardcoded model-specific logic.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

class ModelType(Enum):
    """Supported model types for fMRI processing"""
    CNN_3D = "3d"
    CNN_2D = "2d" 
    CAPSULE_3D = "capsule_3d"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"

@dataclass
class ModelConfig:
    """Configuration class for model-specific parameters"""
    model_type: ModelType
    input_shape: Tuple[int, ...]
    window_size: int
    stride: int
    device: str = "auto"
    preprocessing_params: Dict[str, Any] = None
    inference_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        
        if self.preprocessing_params is None:
            self.preprocessing_params = {}
            
        if self.inference_params is None:
            self.inference_params = {}

class BaseModelAdapter(ABC):
    """Abstract base class for model adapters"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        """Create and return the model instance"""
        pass
    
    @abstractmethod
    def preprocess_data(self, data_path: str) -> torch.Tensor:
        """Preprocess input data for this model type"""
        pass
    
    @abstractmethod
    def get_layer_selection_strategy(self) -> str:
        """Return strategy for layer selection specific to this model type"""
        pass
    
    @abstractmethod
    def postprocess_prediction(self, model_output: torch.Tensor) -> Union[str, int, float]:
        """Convert model output to human-readable prediction"""
        pass

class CapsNet3DAdapter(BaseModelAdapter):
    """Adapter for 3D Capsule Network models"""
    
    def create_model(self) -> torch.nn.Module:
        from scripts.capsnet.model import CapsNetRNN
        model = CapsNetRNN()
        return model.to(self.config.device)
    
    def preprocess_data(self, data_path: str) -> torch.Tensor:
        """Preprocess fMRI data for 3D CapsNet"""
        import nibabel as nib
        import numpy as np
        
        nii = nib.load(data_path)
        data = nii.get_fdata()  # [X, Y, Z, T]
        data = np.transpose(data, (3, 2, 0, 1))  # → [T, Z, H, W]
        
        # Normalize to [0, 1]
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        
        # Sliding window over time
        clips = []
        window, stride = self.config.window_size, self.config.stride
        for i in range(0, data.shape[0] - window + 1, stride):
            clip = data[i:i + window]
            clips.append(clip)
        
        arr = np.stack(clips)  # [B, T, D, H, W]
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)  # → [B, 1, T, D, H, W]
        return tensor.to(self.config.device)
    
    def get_layer_selection_strategy(self) -> str:
        return "improved_capsule"  # Use improved capsule layer selection strategy
    
    def postprocess_prediction(self, model_output: torch.Tensor) -> str:
        """Convert sigmoid output to AD/CN classification"""
        preds = (model_output > 0.5).float().cpu().numpy()
        final_pred = int(np.round(preds.mean()))
        return "AD" if final_pred == 1 else "CN"

class MCADNNetAdapter(BaseModelAdapter):
    """Adapter for 2D MCADN Network models"""
    
    def create_model(self) -> torch.nn.Module:
        from scripts.macadnnet.model import MCADNNet
        # For MCADNNet, we need to pass the actual input shape without batch dimension
        # The model expects (channels, height, width) format
        model_input_shape = self.config.input_shape[1:]  # Remove batch dimension
        model = MCADNNet(input_shape=model_input_shape)
        return model.to(self.config.device)
    
    def preprocess_data(self, data_path: str) -> torch.Tensor:
        """Preprocess fMRI data for 2D MCADN (slice-by-slice processing)"""
        import nibabel as nib
        import numpy as np
        
        nii = nib.load(data_path)
        data = nii.get_fdata()  # [X, Y, Z, T]
        
        # Process as 2D slices - take middle slices or average across time
        # This is a simplified version - you might want more sophisticated preprocessing
        if len(data.shape) == 4:  # 4D fMRI
            data = np.mean(data, axis=3)  # Average across time → [X, Y, Z]
        
        # Take middle slice or process all slices
        middle_slice = data.shape[2] // 2
        slice_data = data[:, :, middle_slice]  # [X, Y]
        
        # Normalize and resize to expected input shape
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data) + 1e-8)
        
        # Resize to model input shape if needed
        # You might want to add proper resizing logic here
        tensor = torch.tensor(slice_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return tensor.to(self.config.device)
    
    def get_layer_selection_strategy(self) -> str:
        return "improved_conv"  # Use improved convolutional layer selection strategy
    
    def postprocess_prediction(self, model_output: torch.Tensor) -> str:
        """Convert logits to classification"""
        probabilities = torch.softmax(model_output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        return "AD" if predicted_class == 1 else "CN"

class ModelFactory:
    """Factory class for creating model adapters"""
    
    _adapters = {
        ModelType.CAPSULE_3D: CapsNet3DAdapter,
        ModelType.CNN_2D: MCADNNetAdapter,
        # Add more adapters here as you create them
    }
    
    @classmethod
    def create_adapter(cls, config: ModelConfig) -> BaseModelAdapter:
        """Create appropriate model adapter based on config"""
        if config.model_type not in cls._adapters:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        adapter_class = cls._adapters[config.model_type]
        return adapter_class(config)
    
    @classmethod
    def register_adapter(cls, model_type: ModelType, adapter_class: type):
        """Register a new model adapter"""
        cls._adapters[model_type] = adapter_class

# Predefined configurations for common models
CAPSNET_CONFIG = ModelConfig(
    model_type=ModelType.CAPSULE_3D,
    input_shape=(1, 1, 91, 91, 109),
    window_size=5,
    stride=3,
    preprocessing_params={
        "normalize_method": "min_max",
        "time_axis": 3
    },
    inference_params={
        "threshold": 0.5,
        "aggregation": "mean"
    }
)

MCADNNET_CONFIG = ModelConfig(
    model_type=ModelType.CNN_2D,
    input_shape=(1, 1, 64, 64),  # Add batch dimension: (batch, channels, height, width)
    window_size=1,
    stride=1,
    preprocessing_params={
        "slice_selection": "middle",
        "resize_method": "interpolation"
    },
    inference_params={
        "output_type": "logits"
    }
)

def get_config_by_name(config_name: str) -> ModelConfig:
    """Get predefined config by name"""
    configs = {
        "capsnet": CAPSNET_CONFIG,
        "mcadnnet": MCADNNET_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]