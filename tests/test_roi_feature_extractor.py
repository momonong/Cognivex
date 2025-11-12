"""
Unit tests for ROIFeatureExtractor
"""

import pytest
import numpy as np
from pathlib import Path

from app.core.ml_processing.feature_extractor import ROIFeatureExtractor
from app.core.ml_processing.exceptions import FeatureExtractionError, AtlasLoadError


class TestROIFeatureExtractor:
    """Test suite for ROIFeatureExtractor"""
    
    def test_load_atlas_success(self):
        """Test successful atlas loading"""
        extractor = ROIFeatureExtractor()
        atlas_img, atlas_labels = extractor.load_atlas()
        
        assert atlas_img is not None
        assert atlas_labels is not None
        assert len(atlas_labels) > 0
        assert isinstance(atlas_labels, list)
    
    def test_load_atlas_caching(self):
        """Test that atlas is cached after first load"""
        extractor = ROIFeatureExtractor()
        
        atlas_img1, labels1 = extractor.load_atlas()
        atlas_img2, labels2 = extractor.load_atlas()
        
        # Should be the same objects (cached)
        assert atlas_img1 is atlas_img2
        assert labels1 is labels2
    
    def test_get_roi_mapping(self):
        """Test ROI mapping creation"""
        extractor = ROIFeatureExtractor()
        roi_mapping = extractor.get_roi_mapping()
        
        assert isinstance(roi_mapping, dict)
        assert len(roi_mapping) > 0
        
        # Check that all values are integers (indices)
        assert all(isinstance(v, int) for v in roi_mapping.values())
        
        # Check that indices start from 1 (AAL convention)
        assert min(roi_mapping.values()) >= 1
    
    def test_validate_roi_list_valid(self):
        """Test validation with valid ROI list"""
        extractor = ROIFeatureExtractor()
        extractor.load_atlas()
        
        # Use first few ROIs from atlas
        roi_mapping = extractor.get_roi_mapping()
        valid_rois = list(roi_mapping.keys())[:5]
        
        # Should not raise exception
        extractor._validate_roi_list(valid_rois)
    
    def test_validate_roi_list_invalid(self):
        """Test validation with invalid ROI names"""
        extractor = ROIFeatureExtractor()
        extractor.load_atlas()
        
        invalid_rois = ["NonexistentROI_1", "FakeRegion_2"]
        
        with pytest.raises(FeatureExtractionError, match="Invalid ROI names"):
            extractor._validate_roi_list(invalid_rois)
    
    @pytest.mark.skipif(
        not Path("data/raw").exists(),
        reason="Test data not available"
    )
    def test_extract_features_correct_shape(self):
        """Test that extracted features have correct shape"""
        # This test requires actual MRI data
        # Skip if test data is not available
        
        extractor = ROIFeatureExtractor()
        roi_mapping = extractor.get_roi_mapping()
        
        # Use first 5 ROIs for testing
        test_rois = list(roi_mapping.keys())[:5]
        
        # Find a test MRI file
        test_files = list(Path("data/raw").rglob("*.nii.gz"))
        if not test_files:
            pytest.skip("No test MRI files found")
        
        test_file = str(test_files[0])
        
        features = extractor.extract_features(test_file, test_rois)
        
        assert features.shape == (len(test_rois),)
        assert isinstance(features, np.ndarray)
        assert features.dtype in [np.float32, np.float64]
    
    def test_extract_features_missing_file(self):
        """Test error handling when MRI file is missing"""
        extractor = ROIFeatureExtractor()
        roi_mapping = extractor.get_roi_mapping()
        test_rois = list(roi_mapping.keys())[:5]
        
        with pytest.raises(FeatureExtractionError, match="MRI file not found"):
            extractor.extract_features("nonexistent_file.nii.gz", test_rois)
    
    @pytest.mark.skipif(
        not Path("data/raw").exists(),
        reason="Test data not available"
    )
    def test_get_feature_dict(self):
        """Test feature extraction as dictionary"""
        extractor = ROIFeatureExtractor()
        roi_mapping = extractor.get_roi_mapping()
        test_rois = list(roi_mapping.keys())[:5]
        
        # Find a test MRI file
        test_files = list(Path("data/raw").rglob("*.nii.gz"))
        if not test_files:
            pytest.skip("No test MRI files found")
        
        test_file = str(test_files[0])
        
        feature_dict = extractor.get_feature_dict(test_file, test_rois)
        
        assert isinstance(feature_dict, dict)
        assert len(feature_dict) == len(test_rois)
        assert all(roi in feature_dict for roi in test_rois)
        assert all(isinstance(v, (float, np.floating)) for v in feature_dict.values())
    
    def test_clear_cache(self):
        """Test cache clearing"""
        extractor = ROIFeatureExtractor()
        
        # Load atlas
        extractor.load_atlas()
        assert extractor._atlas_img is not None
        
        # Clear cache
        extractor.clear_cache()
        assert extractor._atlas_img is None
        assert extractor._atlas_labels is None
        assert extractor._masker is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
