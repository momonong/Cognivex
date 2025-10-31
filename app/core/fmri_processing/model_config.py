import torch
import torch.nn as nn # Added for type hinting nn.Module
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, Optional, List # Added List
from dataclasses import dataclass, field # Added field
from enum import Enum
import numpy as np
import nibabel as nib # Needed for PaperModel preprocessing
import cv2 # Needed for PaperModel preprocessing

class ModelType(Enum):
    """Supported model types for fMRI processing"""
    CNN_3D = "3d"
    CNN_2D = "2d" # Will represent PaperModel for now
    CAPSULE_3D = "capsule_3d"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"
    # Added PaperModel specifically if needed later
    # PAPER_MODEL_2D = "papermodel_2d" 

@dataclass
class ModelConfig:
    """Configuration class for model-specific parameters"""
    model_type: ModelType
    input_shape: Tuple[int, ...] # Should include batch dim for consistency, e.g. (1, C, ...)
    device: str = "auto"
    preprocessing_params: Dict[str, Any] = field(default_factory=dict) # Use field for mutable default
    inference_params: Dict[str, Any] = field(default_factory=dict)
    
    # --- NEW: Paths and Params needed for post-processing ---
    mni_template_path: Optional[str] = None # Path to MNI T1 template
    atlas_path: Optional[str] = None        # Path to Atlas NIfTI
    atlas_label_path: Optional[str] = None  # Path to Atlas Label file (.txt)
    visualization_threshold: float = 0.1    # Default threshold for plotting
    # --- End New ---
    
    # --- Removed window_size and stride as they are preprocessing details ---
    # window_size: int 
    # stride: int

    def __post_init__(self):
        if self.device == "auto":
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        # Ensure dicts are created if None was passed (dataclass handles this with field)

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
        """Preprocess input data (e.g., NIfTI path) for this model type"""
        pass
    
    @abstractmethod
    def get_layer_selection_strategy(self) -> str:
        """Return strategy name for layer selection"""
        pass
    
    @abstractmethod
    def postprocess_prediction(self, 
                               model_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                               return_logits: bool = False
                               ) -> Union[Any, Tuple[Any, torch.Tensor]]: # Allow returning logits
        """Convert model output to prediction and optionally return logits"""
        pass

# --- Existing CapsNet3DAdapter (Minor update for return_logits) ---
class CapsNet3DAdapter(BaseModelAdapter):
    """Adapter for 3D Capsule Network models"""
    
    def create_model(self) -> torch.nn.Module:
        # Assuming path is relative to project root or configured in PYTHONPATH
        from scripts.capsnet.model import CapsNetRNN 
        model = CapsNetRNN()
        # No need to move to device here, pipeline handles it
        return model 
    
    def preprocess_data(self, data_path: str) -> torch.Tensor:
        """Preprocess fMRI data for 3D CapsNet (example implementation)"""
        # (Assuming preprocessing involves loading NIfTI, maybe windowing)
        # This needs to match the actual preprocessing used for training CapsNetRNN
        print("Warning: Using placeholder preprocessing for CapsNet3DAdapter.")
        # Example: Load and return a dummy tensor matching expected shape
        # Input shape likely needs adjustment based on CapsNetRNN definition
        # Let's assume input_shape config is (1, 1, T, D, H, W)
        dummy_shape = self.config.input_shape 
        # Needs actual loading and preprocessing (like windowing if used)
        # For now, return random data matching shape but without batch dim yet
        # The pipeline will add batch dim and move to device
        return torch.randn(dummy_shape[1:]) # Return (1, T, D, H, W)

    def get_layer_selection_strategy(self) -> str:
        return "improved_capsule" # Use improved capsule layer selection strategy
    
    def postprocess_prediction(self, 
                               model_output: torch.Tensor,
                               return_logits: bool = False
                               ) -> Union[str, Tuple[str, torch.Tensor]]:
        """Convert sigmoid output to AD/CN classification"""
        # Assuming model_output is the raw sigmoid output from CapsNetRNN
        # For GradCAM, we might need logits *before* sigmoid. 
        # This adapter needs revision if logits are required.
        # Let's assume model_output IS the probability for class 1 (AD)
        probs = model_output # Shape [B, 1] or [B]
        preds = (probs > 0.5).int() # Use int instead of float
        
        # Aggregate if batch size > 1 (e.g., mean prediction)
        # For single inference B=1
        final_pred = preds.item() if preds.numel() == 1 else int(torch.round(torch.mean(preds.float())).item())
        
        prediction_str = "AD" if final_pred == 1 else "NC" # Changed CN to NC for consistency
        
        if return_logits:
            # Cannot return logits easily from sigmoid output, return probs instead
            print("Warning: CapsNet3DAdapter cannot return logits from sigmoid output, returning probabilities.")
            return prediction_str, probs
        else:
            return prediction_str

# --- NEW: PaperModel Adapter ---
class PaperModelAdapter(BaseModelAdapter):
    """Adapter for the PaperModel (ShuffleNet-based 2D Slice CNN)"""
    
    def create_model(self) -> torch.nn.Module:
        # Import from the correct path
        from model.shufflenet.model import PaperModel 
        # Model __init__ defaults: num_classes=2, groups=3, dropout_p=DROPOUT_RATE
        model = PaperModel() 
        # No need to move to device here
        return model
    
    def preprocess_data(self, data_path: str) -> torch.Tensor:
        """Preprocess fMRI NIfTI data for ShuffleNet (handles both 3D and 4D data)"""
        import nibabel as nib
        import numpy as np
        import cv2
        
        try:
            # 1. Load NIfTI data
            img = nib.load(data_path)
            data = img.get_fdata()
            
            # 2. Handle 4D fMRI data by taking temporal mean
            if len(data.shape) == 4:
                print(f"4D fMRI data detected with shape {data.shape}, taking temporal mean")
                data = np.mean(data, axis=3)  # Average over time dimension
                print(f"After temporal averaging: {data.shape}")
            elif len(data.shape) == 3:
                print(f"3D structural data detected with shape {data.shape}")
            else:
                raise ValueError(f"Unsupported data dimensions: {data.shape}")
            
            # 3. Extract sagittal slices (same as original preprocess_nii_to_slices)
            sagittal_dim = 0
            num_total_slices = data.shape[sagittal_dim]
            
            if num_total_slices < 10:  # NUM_SLICES_PER_SUBJECT
                raise ValueError(f"Not enough sagittal slices: {num_total_slices} < 10")
            
            # Find center 10 slices
            center_slice_index = num_total_slices // 2
            start_index = center_slice_index - 5  # 10 // 2
            end_index = start_index + 10
            
            selected_slices_data = data[start_index:end_index, :, :]
            
            # 4. Process each slice
            processed_slices = []
            for i in range(10):
                slice_2d = selected_slices_data[i, :, :]
                
                # Rotate image for correct orientation
                slice_2d = np.rot90(slice_2d)
                
                # Normalize to 0-255
                if np.max(slice_2d) > 0:
                    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))
                slice_2d_uint8 = (slice_2d * 255).astype(np.uint8)
                
                # Resize to 128x128
                resized_slice = cv2.resize(slice_2d_uint8, (128, 128), interpolation=cv2.INTER_CUBIC)
                processed_slices.append(resized_slice)
            
            # 5. Stack and convert to tensor
            stacked_slices = np.stack(processed_slices)  # (10, 128, 128)
            stacked_slices = stacked_slices[:, np.newaxis, :, :]  # (10, 1, 128, 128)
            
            # Convert to tensor and normalize 0-1
            slices_tensor = torch.tensor(stacked_slices, dtype=torch.float32) / 255.0
            
            # Add batch dimension: (1, 10, 1, 128, 128)
            input_tensor = slices_tensor.unsqueeze(0)
            
            return input_tensor
            
        except Exception as e:
            raise ValueError(f"Preprocessing failed for {data_path}: {e}")

    def get_layer_selection_strategy(self) -> str:
        # Use the latest, most specific strategy for this model
        return "improved_shufflenet" 
    
    def postprocess_prediction(self, 
                               model_output: Tuple[torch.Tensor, torch.Tensor], # Expects (logits, embeddings)
                               return_logits: bool = False
                               ) -> Union[str, Tuple[str, torch.Tensor]]:
        """Convert logits to AD/NC classification"""
        logits, _ = model_output # Unpack the tuple, ignore embeddings here
        
        # Logits shape: [B, num_classes] e.g., [1, 2]
        probabilities = torch.softmax(logits, dim=1)
        # Get the index (0 or 1) of the highest probability
        predicted_class_index = torch.argmax(probabilities, dim=1) 
        
        # For single inference B=1
        final_pred = predicted_class_index.item()
        
        prediction_str = "AD" if final_pred == 1 else "NC" 
        
        if return_logits:
            return prediction_str, logits # Return the raw logits
        else:
            return prediction_str

# --- Updated ModelFactory ---
class ModelFactory:
    """Factory class for creating model adapters"""
    
    _adapters = {
        ModelType.CAPSULE_3D: CapsNet3DAdapter,
        ModelType.CNN_2D: PaperModelAdapter, # Assign PaperModelAdapter to CNN_2D
        # Add other adapters like MCADNNetAdapter if needed, maybe using a different ModelType
    }
    
    @classmethod
    def create_adapter(cls, config: ModelConfig) -> BaseModelAdapter:
        """Create appropriate model adapter based on config"""
        if config.model_type not in cls._adapters:
            # Try matching enum value (string) if direct type match fails
            adapter_class = None
            for key, adapter in cls._adapters.items():
                 if key.value == config.model_type.value: # Compare string values
                     adapter_class = adapter
                     break
            if adapter_class is None:
                 raise ValueError(f"Unsupported model type: {config.model_type}")
        else:
            adapter_class = cls._adapters[config.model_type]
            
        return adapter_class(config)
    
    @classmethod
    def register_adapter(cls, model_type: ModelType, adapter_class: type):
        """Register a new model adapter"""
        cls._adapters[model_type] = adapter_class

# --- Predefined Configurations ---

# (CAPSNET_CONFIG remains mostly the same, adjust input_shape if needed)
CAPSNET_CONFIG = ModelConfig(
    model_type=ModelType.CAPSULE_3D,
    # Example shape, verify with actual CapsNetRNN implementation
    input_shape=(1, 1, 5, 91, 91, 109), # (B, C, T, D, H, W) - Assuming T=5
    # window_size and stride removed, handle in adapter.preprocess_data
    preprocessing_params={}, 
    inference_params={"threshold": 0.5} 
)

# --- NEW: PaperModel Config ---
PAPERMODEL_CONFIG = ModelConfig(
    model_type=ModelType.CNN_2D, # Use CNN_2D for now
    input_shape=(1, 10, 1, 128, 128), # (B, N_slices, C, H, W)
    preprocessing_params={
        "num_slices": 10, # Matches NUM_SLICES_PER_SUBJECT
        "slice_size": 128, # Matches SLICE_IMG_SIZE
        "plane": "sagittal" # Document the plane used
    },
    inference_params={
        "output_type": "logits_and_embeddings" # Document model output
    },
    # --- ADD Required Paths ---
    # These MUST be updated to your actual file paths
    mni_template_path="data/affine/mni152_template.nii.gz", # Update this!
    atlas_path="data/aal3/AAL3v1_1mm.nii.gz",           # Update if different
    atlas_label_path="data/aal3/AAL3v1_1mm.nii.txt",    # Update if different
    visualization_threshold=0.3 # Example threshold, adjust as needed
)


# --- Updated get_config_by_name ---
def get_config_by_name(config_name: str) -> ModelConfig:
    """Get predefined config by name"""
    configs = {
        "capsnet": CAPSNET_CONFIG,
        "papermodel": PAPERMODEL_CONFIG, # Add papermodel config
        "shufflenet": PAPERMODEL_CONFIG, # Add shufflenet as alias for papermodel
        "mcadnnet": PAPERMODEL_CONFIG,   # Add mcadnnet as alias for papermodel (2D CNN)
    }
    
    config_name_lower = config_name.lower() # Make lookup case-insensitive
    if config_name_lower not in configs:
        raise ValueError(f"Unknown config name: '{config_name}'. Available: {list(configs.keys())}")
    
    return configs[config_name_lower]