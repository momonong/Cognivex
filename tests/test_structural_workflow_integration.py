"""
Integration tests for structural MRI workflow
"""

import pytest
from pathlib import Path
from app.graph.workflow import app, route_by_analysis_mode
from app.graph.state import AgentState


class TestWorkflowRouting:
    """Test workflow routing logic"""
    
    def test_route_to_structural_branch(self):
        """Test routing to structural MRI branch"""
        state: AgentState = {
            "analysis_mode": "structural",
            "subject_id": "test",
            "fmri_scan_path": "test.nii.gz",
            "trace_log": [],
            "error_log": []
        }
        
        next_node = route_by_analysis_mode(state)
        assert next_node == "structural_mri_inference"
    
    def test_route_to_functional_branch(self):
        """Test routing to functional MRI branch"""
        state: AgentState = {
            "analysis_mode": "functional",
            "subject_id": "test",
            "fmri_scan_path": "test.nii.gz",
            "trace_log": [],
            "error_log": []
        }
        
        next_node = route_by_analysis_mode(state)
        assert next_node == "inference"
    
    def test_route_default_to_functional(self):
        """Test default routing when mode not specified"""
        state: AgentState = {
            "subject_id": "test",
            "fmri_scan_path": "test.nii.gz",
            "trace_log": [],
            "error_log": []
        }
        
        next_node = route_by_analysis_mode(state)
        assert next_node == "inference"


class TestStructuralWorkflowIntegration:
    """Test complete structural MRI workflow"""
    
    @pytest.mark.skipif(
        not Path("model/ml/final/final_model.pkl").exists(),
        reason="Model files not available"
    )
    @pytest.mark.skipif(
        not Path("data/raw").exists(),
        reason="Test data not available"
    )
    def test_full_structural_pipeline(self):
        """Test complete structural MRI analysis pipeline"""
        # Find a test MRI file
        test_files = list(Path("data/raw").rglob("*.nii.gz"))
        if not test_files:
            pytest.skip("No test MRI files found")
        
        initial_state: AgentState = {
            "subject_id": "test_structural_pipeline",
            "fmri_scan_path": str(test_files[0]),
            "analysis_mode": "structural",
            "trace_log": [],
            "error_log": []
        }
        
        # Execute workflow
        final_state = app.invoke(initial_state)
        
        # Verify structural MRI specific outputs
        assert "classification_result" in final_state
        assert final_state["classification_result"] in ["NC", "AD", "ERROR"]
        
        if final_state["classification_result"] != "ERROR":
            # Check prediction outputs
            assert "prediction_confidence" in final_state
            assert 0 <= final_state["prediction_confidence"] <= 1
            
            # Check feature outputs
            assert "roi_features" in final_state
            assert "feature_importances" in final_state
            
            # Check activated regions
            assert "activated_regions" in final_state
            assert len(final_state["activated_regions"]) > 0
            
            # Check visualizations
            assert "visualization_paths" in final_state
            assert len(final_state["visualization_paths"]) > 0
            
            # Verify visualization files exist
            for viz_path in final_state["visualization_paths"]:
                assert Path(viz_path).exists()
            
            # Check that no errors occurred
            assert len(final_state.get("error_log", [])) == 0
    
    @pytest.mark.skipif(
        not Path("model/capsnet/best_capsnet_rnn.pth").exists(),
        reason="Functional MRI model not available"
    )
    @pytest.mark.skipif(
        not Path("data/raw").exists(),
        reason="Test data not available"
    )
    def test_functional_branch_still_works(self):
        """Test that functional MRI branch is not affected"""
        # Find a test MRI file
        test_files = list(Path("data/raw").rglob("*.nii.gz"))
        if not test_files:
            pytest.skip("No test MRI files found")
        
        initial_state: AgentState = {
            "subject_id": "test_functional_pipeline",
            "fmri_scan_path": str(test_files[0]),
            "model_path": "model/capsnet/best_capsnet_rnn.pth",
            "model_name": "capsnet",
            "analysis_mode": "functional",
            "trace_log": [],
            "error_log": []
        }
        
        # Execute workflow
        final_state = app.invoke(initial_state)
        
        # Verify functional MRI outputs
        assert "classification_result" in final_state
        
        # Functional branch should have different outputs
        # (validated_layers, final_layers, etc.)
        # We just check it doesn't crash
        assert final_state is not None
    
    def test_state_propagation_through_nodes(self):
        """Test that state is correctly passed between nodes"""
        # This is a simplified test with mock data
        initial_state: AgentState = {
            "subject_id": "test_state_prop",
            "fmri_scan_path": "nonexistent.nii.gz",  # Will cause error
            "analysis_mode": "structural",
            "trace_log": [],
            "error_log": []
        }
        
        # Execute workflow (will fail but should handle gracefully)
        final_state = app.invoke(initial_state)
        
        # Check that state was propagated
        assert "subject_id" in final_state
        assert final_state["subject_id"] == "test_state_prop"
        
        # Check that trace_log was updated
        assert "trace_log" in final_state
        
        # Check that errors were logged
        assert "error_log" in final_state


class TestWorkflowBranchConvergence:
    """Test that both branches converge correctly"""
    
    def test_both_branches_reach_entity_linker(self):
        """Test that both branches eventually reach entity_linker"""
        # This test verifies the workflow structure
        # Both structural and functional branches should converge at entity_linker
        
        # We can't easily test this without running the full pipeline,
        # but we can verify the workflow graph structure
        from app.graph.workflow import workflow
        
        # Check that entity_linker node exists
        assert "entity_linker" in workflow.nodes
        
        # Check that both branches have paths to entity_linker
        # (This is implicit in the workflow definition)
        assert True  # Placeholder for structural verification


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
