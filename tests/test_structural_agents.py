"""
Unit tests for structural MRI agents
"""

import pytest
from pathlib import Path
from app.graph.state import AgentState
from app.agents.structural_mri_inference import run_structural_mri_inference
from app.agents.structural_feature_analyzer import analyze_feature_importance
from app.agents.structural_visualizer import generate_structural_visualizations


class TestStructuralMRIInference:
    """Test suite for structural_mri_inference agent"""
    
    @pytest.mark.skipif(
        not Path("model/ml/final/final_model.pkl").exists(),
        reason="Model files not available"
    )
    def test_inference_with_valid_input(self):
        """Test inference with valid MRI file"""
        # Find a test MRI file
        test_files = list(Path("data/raw").rglob("*.nii.gz"))
        if not test_files:
            pytest.skip("No test MRI files found")
        
        initial_state: AgentState = {
            "subject_id": "test_subject",
            "fmri_scan_path": str(test_files[0]),
            "trace_log": [],
            "error_log": []
        }
        
        result = run_structural_mri_inference(initial_state)
        
        # Check that result contains expected fields
        assert "classification_result" in result
        assert result["classification_result"] in ["NC", "AD", "ERROR"]
        
        if result["classification_result"] != "ERROR":
            assert "prediction_confidence" in result
            assert 0 <= result["prediction_confidence"] <= 1
            assert "roi_features" in result
            assert "feature_importances" in result
            assert len(result["roi_features"]) == 32
            assert len(result["feature_importances"]) == 32
    
    def test_inference_with_missing_file(self):
        """Test error handling when MRI file is missing"""
        initial_state: AgentState = {
            "subject_id": "test_subject",
            "fmri_scan_path": "nonexistent_file.nii.gz",
            "trace_log": [],
            "error_log": []
        }
        
        result = run_structural_mri_inference(initial_state)
        
        # Should return error
        assert "error_log" in result
        assert len(result["error_log"]) > 0
        assert result["classification_result"] in ["ERROR", "ERROR: Feature extraction failed"]
    
    def test_inference_with_missing_path(self):
        """Test error handling when scan path is missing"""
        initial_state: AgentState = {
            "subject_id": "test_subject",
            "trace_log": [],
            "error_log": []
        }
        
        result = run_structural_mri_inference(initial_state)
        
        # Should return error
        assert "error_log" in result
        assert len(result["error_log"]) > 0
        assert "Missing MRI scan path" in result["error_log"][0]


class TestStructuralFeatureAnalyzer:
    """Test suite for structural_feature_analyzer agent"""
    
    def test_analyze_with_valid_importances(self):
        """Test feature analysis with valid importances"""
        # Create mock feature importances
        feature_importances = {
            f"ROI_{i}_L": 0.1 - i * 0.01 for i in range(16)
        }
        feature_importances.update({
            f"ROI_{i}_R": 0.09 - i * 0.01 for i in range(16)
        })
        
        initial_state: AgentState = {
            "feature_importances": feature_importances,
            "roi_features": feature_importances,  # Mock values
            "trace_log": []
        }
        
        result = analyze_feature_importance(initial_state)
        
        # Check results
        assert "activated_regions" in result
        assert len(result["activated_regions"]) == 32
        
        # Check that regions are sorted by importance
        regions = result["activated_regions"]
        for i in range(len(regions) - 1):
            assert regions[i]["activation_score"] >= regions[i + 1]["activation_score"]
        
        # Check that importance_rank is set correctly
        for i, region in enumerate(regions, 1):
            assert region["importance_rank"] == i
        
        # Check hemisphere detection
        for region in regions:
            if region["region_name"].endswith("_L"):
                assert region["hemisphere"] == "Left"
            elif region["region_name"].endswith("_R"):
                assert region["hemisphere"] == "Right"
    
    def test_analyze_with_empty_importances(self):
        """Test handling of empty importances"""
        initial_state: AgentState = {
            "feature_importances": {},
            "trace_log": []
        }
        
        result = analyze_feature_importance(initial_state)
        
        # Should return empty list
        assert "activated_regions" in result
        assert len(result["activated_regions"]) == 0
    
    def test_analyze_sorting_logic(self):
        """Test that sorting works correctly"""
        feature_importances = {
            "ROI_A": 0.05,
            "ROI_B": 0.15,
            "ROI_C": 0.10,
            "ROI_D": 0.20
        }
        
        initial_state: AgentState = {
            "feature_importances": feature_importances,
            "trace_log": []
        }
        
        result = analyze_feature_importance(initial_state)
        regions = result["activated_regions"]
        
        # Check order: D > B > C > A
        assert regions[0]["region_name"] == "ROI_D"
        assert regions[1]["region_name"] == "ROI_B"
        assert regions[2]["region_name"] == "ROI_C"
        assert regions[3]["region_name"] == "ROI_A"


class TestStructuralVisualizer:
    """Test suite for structural_visualizer agent"""
    
    def test_visualizer_with_valid_importances(self):
        """Test visualization generation with valid importances"""
        feature_importances = {
            f"Hippocampus_{side}": 0.08 - i * 0.01
            for i, side in enumerate(["L", "R"])
        }
        feature_importances.update({
            f"Temporal_Mid_{side}": 0.06 - i * 0.01
            for i, side in enumerate(["L", "R"])
        })
        
        initial_state: AgentState = {
            "subject_id": "test_viz",
            "feature_importances": feature_importances,
            "trace_log": []
        }
        
        result = generate_structural_visualizations(initial_state)
        
        # Check that visualization paths are returned
        assert "visualization_paths" in result
        assert "feature_importance_plot_path" in result
        assert "roi_visualization_path" in result
        
        if len(result["visualization_paths"]) > 0:
            # Check that files were created
            for path in result["visualization_paths"]:
                assert Path(path).exists()
                assert path.endswith(".png")
    
    def test_visualizer_with_empty_importances(self):
        """Test handling of empty importances"""
        initial_state: AgentState = {
            "subject_id": "test_empty",
            "feature_importances": {},
            "trace_log": []
        }
        
        result = generate_structural_visualizations(initial_state)
        
        # Should return empty list
        assert "visualization_paths" in result
        assert len(result["visualization_paths"]) == 0
    
    def test_visualizer_creates_output_directory(self):
        """Test that output directory is created"""
        feature_importances = {"ROI_A": 0.1, "ROI_B": 0.2}
        
        initial_state: AgentState = {
            "subject_id": "test_dir_creation",
            "feature_importances": feature_importances,
            "trace_log": []
        }
        
        result = generate_structural_visualizations(initial_state)
        
        # Check that output directory exists
        output_dir = Path(f"output/ml_analysis/test_dir_creation")
        assert output_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
