"""
Unit tests for MLModelLoader
"""

import pytest
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from app.core.ml_processing.model_loader import MLModelLoader
from app.core.ml_processing.config import MLModelConfig
from app.core.ml_processing.exceptions import ModelLoadError


class TestMLModelLoader:
    """Test suite for MLModelLoader"""
    
    def test_load_model_success(self):
        """Test successful model loading"""
        # Use real model if it exists
        config = MLModelConfig.from_directory()
        
        if not Path(config.model_path).exists():
            pytest.skip("Model file not found, skipping test")
        
        loader = MLModelLoader(config)
        model = loader.load_model()
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators > 0
        assert model.n_features_in_ == 32
    
    def test_load_model_caching(self):
        """Test that model is cached after first load"""
        config = MLModelConfig.from_directory()
        
        if not Path(config.model_path).exists():
            pytest.skip("Model file not found, skipping test")
        
        loader = MLModelLoader(config)
        
        # Load twice
        model1 = loader.load_model()
        model2 = loader.load_model()
        
        # Should be the same object (cached)
        assert model1 is model2
    
    def test_load_model_missing_file(self):
        """Test error handling when model file is missing"""
        config = MLModelConfig(
            model_type="random_forest",
            model_path="nonexistent/model.pkl",
            scaler_path="nonexistent/scaler.pkl",
            roi_list_path="nonexistent/roi_list.csv",
            feature_names_path="nonexistent/features.txt"
        )
        
        loader = MLModelLoader(config)
        
        with pytest.raises(ModelLoadError, match="Model file not found"):
            loader.load_model()
    
    def test_load_scaler_success(self):
        """Test successful scaler loading"""
        config = MLModelConfig.from_directory()
        
        if not Path(config.scaler_path).exists():
            pytest.skip("Scaler file not found, skipping test")
        
        loader = MLModelLoader(config)
        scaler = loader.load_scaler()
        
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, 'mean_')
        assert hasattr(scaler, 'scale_')
    
    def test_load_roi_list_success(self):
        """Test successful ROI list loading"""
        config = MLModelConfig.from_directory()
        
        if not Path(config.roi_list_path).exists():
            pytest.skip("ROI list file not found, skipping test")
        
        loader = MLModelLoader(config)
        roi_list = loader.load_roi_list()
        
        assert isinstance(roi_list, list)
        assert len(roi_list) == 32
        assert all(isinstance(roi, str) for roi in roi_list)
    
    def test_load_feature_names_success(self):
        """Test successful feature names loading"""
        config = MLModelConfig.from_directory()
        
        if not Path(config.feature_names_path).exists():
            pytest.skip("Feature names file not found, skipping test")
        
        loader = MLModelLoader(config)
        feature_names = loader.load_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) == 32
        assert all(isinstance(name, str) for name in feature_names)
    
    def test_get_all_components_success(self):
        """Test loading all components at once"""
        config = MLModelConfig.from_directory()
        
        # Check if all files exist
        required_files = [
            config.model_path,
            config.scaler_path,
            config.roi_list_path,
            config.feature_names_path
        ]
        
        if not all(Path(f).exists() for f in required_files):
            pytest.skip("Not all model files found, skipping test")
        
        loader = MLModelLoader(config)
        components = loader.get_all_components()
        
        assert 'model' in components
        assert 'scaler' in components
        assert 'roi_list' in components
        assert 'feature_names' in components
        
        assert isinstance(components['model'], RandomForestClassifier)
        assert isinstance(components['scaler'], StandardScaler)
        assert len(components['roi_list']) == 32
        assert len(components['feature_names']) == 32
    
    def test_clear_cache(self):
        """Test cache clearing"""
        config = MLModelConfig.from_directory()
        
        if not Path(config.model_path).exists():
            pytest.skip("Model file not found, skipping test")
        
        loader = MLModelLoader(config)
        
        # Load model
        model1 = loader.load_model()
        assert loader._model is not None
        
        # Clear cache
        loader.clear_cache()
        assert loader._model is None
        
        # Load again should create new instance
        model2 = loader.load_model()
        assert model2 is not model1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
