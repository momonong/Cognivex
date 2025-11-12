"""
Model loader for ML-based structural MRI analysis
"""

import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .exceptions import ModelLoadError
from .config import MLModelConfig


class MLModelLoader:
    """
    Manages loading and caching of ML model components
    
    This class handles loading of:
    - Random Forest model
    - Feature scaler
    - ROI list
    - Feature names
    """
    
    def __init__(self, config: Optional[MLModelConfig] = None):
        """
        Initialize model loader
        
        Args:
            config: Model configuration. If None, uses default from model/ml/final
        """
        self.config = config or MLModelConfig.from_directory()
        
        # Cache for loaded components
        self._model: Optional[RandomForestClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._roi_list: Optional[List[str]] = None
        self._feature_names: Optional[List[str]] = None
    
    def load_model(self) -> RandomForestClassifier:
        """
        Load Random Forest model
        
        Returns:
            Loaded RandomForestClassifier
            
        Raises:
            ModelLoadError: If model file is missing or invalid
        """
        if self._model is not None:
            return self._model
        
        try:
            model_path = Path(self.config.model_path)
            
            if not model_path.exists():
                raise ModelLoadError(
                    f"Model file not found: {model_path}. "
                    f"Please ensure the model is trained and saved at this location."
                )
            
            with open(model_path, 'rb') as f:
                self._model = pickle.load(f)
            
            # Validate it's a RandomForestClassifier
            if not isinstance(self._model, RandomForestClassifier):
                raise ModelLoadError(
                    f"Expected RandomForestClassifier, got {type(self._model)}"
                )
            
            print(f"✓ Loaded Random Forest model from {model_path}")
            print(f"  - n_estimators: {self._model.n_estimators}")
            print(f"  - n_features: {self._model.n_features_in_}")
            
            return self._model
            
        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def load_scaler(self) -> StandardScaler:
        """
        Load feature scaler
        
        Returns:
            Loaded StandardScaler
            
        Raises:
            ModelLoadError: If scaler file is missing or invalid
        """
        if self._scaler is not None:
            return self._scaler
        
        try:
            scaler_path = Path(self.config.scaler_path)
            
            if not scaler_path.exists():
                raise ModelLoadError(
                    f"Scaler file not found: {scaler_path}"
                )
            
            with open(scaler_path, 'rb') as f:
                self._scaler = pickle.load(f)
            
            # Validate it's a StandardScaler
            if not isinstance(self._scaler, StandardScaler):
                raise ModelLoadError(
                    f"Expected StandardScaler, got {type(self._scaler)}"
                )
            
            print(f"✓ Loaded StandardScaler from {scaler_path}")
            
            return self._scaler
            
        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise
            raise ModelLoadError(f"Failed to load scaler: {e}")
    
    def load_roi_list(self) -> List[str]:
        """
        Load ROI list from CSV
        
        Returns:
            List of ROI names
            
        Raises:
            ModelLoadError: If ROI list file is missing or invalid
        """
        if self._roi_list is not None:
            return self._roi_list
        
        try:
            roi_list_path = Path(self.config.roi_list_path)
            
            if not roi_list_path.exists():
                raise ModelLoadError(
                    f"ROI list file not found: {roi_list_path}"
                )
            
            # Read CSV file
            df = pd.read_csv(roi_list_path)
            
            # Expect a column named 'ROI' or use first column
            if 'ROI' in df.columns:
                self._roi_list = df['ROI'].tolist()
            elif 'roi' in df.columns:
                self._roi_list = df['roi'].tolist()
            else:
                self._roi_list = df.iloc[:, 0].tolist()
            
            print(f"✓ Loaded {len(self._roi_list)} ROIs from {roi_list_path}")
            
            return self._roi_list
            
        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise
            raise ModelLoadError(f"Failed to load ROI list: {e}")
    
    def load_feature_names(self) -> List[str]:
        """
        Load feature names from text file
        
        Returns:
            List of feature names
            
        Raises:
            ModelLoadError: If feature names file is missing
        """
        if self._feature_names is not None:
            return self._feature_names
        
        try:
            feature_names_path = Path(self.config.feature_names_path)
            
            if not feature_names_path.exists():
                raise ModelLoadError(
                    f"Feature names file not found: {feature_names_path}"
                )
            
            with open(feature_names_path, 'r') as f:
                self._feature_names = [line.strip() for line in f if line.strip()]
            
            print(f"✓ Loaded {len(self._feature_names)} feature names from {feature_names_path}")
            
            return self._feature_names
            
        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise
            raise ModelLoadError(f"Failed to load feature names: {e}")
    
    def get_all_components(self) -> Dict[str, Any]:
        """
        Load all model components at once
        
        Returns:
            Dictionary containing all components:
            - 'model': RandomForestClassifier
            - 'scaler': StandardScaler
            - 'roi_list': List of ROI names
            - 'feature_names': List of feature names
            
        Raises:
            ModelLoadError: If any component fails to load
        """
        print("\n=== Loading ML Model Components ===")
        
        components = {
            'model': self.load_model(),
            'scaler': self.load_scaler(),
            'roi_list': self.load_roi_list(),
            'feature_names': self.load_feature_names()
        }
        
        # Validate consistency
        if len(components['roi_list']) != len(components['feature_names']):
            raise ModelLoadError(
                f"ROI list length ({len(components['roi_list'])}) "
                f"does not match feature names length ({len(components['feature_names'])})"
            )
        
        if components['model'].n_features_in_ != len(components['feature_names']):
            raise ModelLoadError(
                f"Model expects {components['model'].n_features_in_} features, "
                f"but {len(components['feature_names'])} feature names provided"
            )
        
        print(f"✓ All components loaded successfully")
        print(f"  - Model: Random Forest with {components['model'].n_estimators} trees")
        print(f"  - Features: {len(components['feature_names'])} ROIs")
        print("=" * 35 + "\n")
        
        return components
    
    def clear_cache(self):
        """Clear all cached components"""
        self._model = None
        self._scaler = None
        self._roi_list = None
        self._feature_names = None
        print("✓ Model cache cleared")
