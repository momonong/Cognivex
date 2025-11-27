import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to the sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.fmri_processing.generic_pipeline_steps import GenericInferencePipeline
from app.core.fmri_processing.model_config import ModelConfig, ModelType, get_config_by_name

# Mock data for inspect_torch_model
MOCK_ALL_LAYERS = [
    {"model_path": "conv1", "layer_type": "Conv3d", "output_shape": "(1, 32, 64, 64, 64)", "params": 1000},
    {"model_path": "conv2", "layer_type": "Conv3d", "output_shape": "(1, 64, 32, 32, 32)", "params": 2000},
    {"model_path": "conv3", "layer_type": "Conv3d", "output_shape": "(1, 128, 16, 16, 16)", "params": 4000},
    {"model_path": "capsnet.conv1", "layer_type": "Conv3d", "output_shape": "(1, 256, 8, 8, 8)", "params": 8000},
    {"model_path": "capsnet.conv2", "layer_type": "Conv3d", "output_shape": "(1, 256, 8, 8, 8)", "params": 8000},
    {"model_path": "capsnet.conv3", "layer_type": "Conv3d", "output_shape": "(1, 256, 8, 8, 8)", "params": 8000},
    {"model_path": "capsnet.caps1", "layer_type": "PrimaryCaps", "output_shape": "(1, 8, 16, 16, 16)", "params": 16000},
    {"model_path": "backbone.stage1", "layer_type": "Sequential", "output_shape": "(1, 24, 64, 64)", "params": 500},
    {"model_path": "backbone.stage2", "layer_type": "Sequential", "output_shape": "(1, 48, 32, 32)", "params": 1000},
    {"model_path": "backbone.stage3", "layer_type": "Sequential", "output_shape": "(1, 96, 16, 16)", "params": 2000},
    {"model_path": "backbone.stage4", "layer_type": "Sequential", "output_shape": "(1, 192, 8, 8)", "params": 4000},
]

# Mock responses for select_visualization_layers based on strategy
MOCK_LLM_RESPONSES = {
    "improved_capsule": json.dumps([
        {"model_path": "capsnet.conv2", "reason": "Mid-level features for capsule network."},
        {"model_path": "capsnet.conv3", "reason": "High-level features before primary capsules."},
        {"model_path": "capsnet.caps1", "reason": "Primary capsule representations."} 
    ]),
    "improved_shufflenet": json.dumps([
        {"model_path": "backbone.stage3", "reason": "Key stage for ShuffleNet feature extraction."},
        {"model_path": "backbone.stage4", "reason": "Final feature extraction stage."} 
    ]),
    "default": json.dumps([
        {"model_path": "conv1", "reason": "Default strategy selected this layer."} 
    ]),
    # Add other strategies if needed for more comprehensive tests
}

class TestLayerSelectionFlexibility(unittest.TestCase):

    @patch('os.path.exists', return_value=True) # Mock path existence for config validation
    @patch('app.core.fmri_processing.pipelines.choose_layer.select_visualization_layers', side_effect=lambda layers, strategy: MOCK_LLM_RESPONSES.get(strategy, MOCK_LLM_RESPONSES["default"]))
    @patch('app.core.fmri_processing.pipelines.inspect_model.inspect_torch_model', return_value=MOCK_ALL_LAYERS)
    @patch('app.core.fmri_processing.model_config.CapsNet3DAdapter.create_model', return_value=MagicMock())
    @patch('app.core.fmri_processing.model_config.PaperModelAdapter.create_model', return_value=MagicMock())
    def test_capsnet_improved_capsule_strategy(self, mock_paper_model_create, mock_capsnet_create, mock_inspect_torch_model, mock_select_layers, mock_path_exists):
        print("\n--- Test CapsNet with improved_capsule strategy ---")
        pipeline = GenericInferencePipeline(
            model_config="capsnet",
            layer_selection_strategy="improved_capsule"
        )
        selected_layers, _ = pipeline.inspect_and_select_layers()
        
        self.assertIsNotNone(selected_layers)
        self.assertGreater(len(selected_layers), 0)
        self.assertTrue(any(l['model_path'] == "capsnet.caps1" for l in selected_layers))
        mock_select_layers.assert_called_with(MOCK_ALL_LAYERS, strategy="improved_capsule")
        print(f"Selected layers: {[l['model_path'] for l in selected_layers]}")

    @patch('os.path.exists', return_value=True)
    @patch('app.core.fmri_processing.pipelines.choose_layer.select_visualization_layers', side_effect=lambda layers, strategy: MOCK_LLM_RESPONSES.get(strategy, MOCK_LLM_RESPONSES["default"]))
    @patch('app.core.fmri_processing.pipelines.inspect_model.inspect_torch_model', return_value=MOCK_ALL_LAYERS)
    @patch('app.core.fmri_processing.model_config.CapsNet3DAdapter.create_model', return_value=MagicMock())
    @patch('app.core.fmri_processing.model_config.PaperModelAdapter.create_model', return_value=MagicMock())
    def test_papermodel_improved_shufflenet_strategy(self, mock_paper_model_create, mock_capsnet_create, mock_inspect_torch_model, mock_select_layers, mock_path_exists):
        print("\n--- Test PaperModel with improved_shufflenet strategy ---")
        pipeline = GenericInferencePipeline(
            model_config="papermodel",
            layer_selection_strategy="improved_shufflenet"
        )
        selected_layers, _ = pipeline.inspect_and_select_layers()
        
        self.assertIsNotNone(selected_layers)
        self.assertGreater(len(selected_layers), 0)
        self.assertTrue(any(l['model_path'] == "backbone.stage3" for l in selected_layers))
        mock_select_layers.assert_called_with(MOCK_ALL_LAYERS, strategy="improved_shufflenet")
        print(f"Selected layers: {[l['model_path'] for l in selected_layers]}")

    @patch('os.path.exists', return_value=True)
    @patch('app.core.fmri_processing.pipelines.inspector.inspect_torch_model', return_value=MOCK_ALL_LAYERS)
    @patch('app.core.fmri_processing.model_config.CapsNet3DAdapter.create_model', return_value=MagicMock())
    @patch('app.core.fmri_processing.model_config.PaperModelAdapter.create_model', return_value=MagicMock())
    def test_papermodel_force_select_stage2_strategy(self, mock_paper_model_create, mock_capsnet_create, mock_inspect_torch_model, mock_select_layers, mock_path_exists):
        print("\n--- Test PaperModel with force_select_stage2 strategy ---")
        pipeline = GenericInferencePipeline(
            model_config="papermodel",
            layer_selection_strategy="force_select_stage2"
        )
        selected_layers, _ = pipeline.inspect_and_select_layers()
        
        self.assertIsNotNone(selected_layers)
        self.assertEqual(len(selected_layers), 1)
        self.assertEqual(selected_layers[0]['model_path'], "backbone.stage2")
        mock_select_layers.assert_not_called() # LLM should be bypassed
        print(f"Selected layers: {[l['model_path'] for l in selected_layers]}")

    @patch('os.path.exists', return_value=True)
    @patch('app.core.fmri_processing.pipelines.choose_layer.select_visualization_layers', side_effect=lambda layers, strategy: MOCK_LLM_RESPONSES.get(strategy, MOCK_LLM_RESPONSES["default"]))
    @patch('app.core.fmri_processing.pipelines.inspect_model.inspect_torch_model', return_value=MOCK_ALL_LAYERS)
    @patch('app.core.fmri_processing.model_config.CapsNet3DAdapter.create_model', return_value=MagicMock())
    @patch('app.core.fmri_processing.model_config.PaperModelAdapter.create_model', return_value=MagicMock())
    def test_default_strategy_fallback_capsnet(self, mock_paper_model_create, mock_capsnet_create, mock_inspect_torch_model, mock_select_layers, mock_path_exists):
        print("\n--- Test Default Strategy Fallback (CapsNet) ---")
        # No layer_selection_strategy provided, should fall back to adapter's default
        pipeline = GenericInferencePipeline(
            model_config="capsnet"
        )
        selected_layers, _ = pipeline.inspect_and_select_layers()
        
        self.assertIsNotNone(selected_layers)
        self.assertGreater(len(selected_layers), 0)
        # Should call with the adapter's default strategy, which is now "improved_capsule"
        mock_select_layers.assert_called_with(MOCK_ALL_LAYERS, strategy="improved_capsule")
        print(f"Selected layers: {[l['model_path'] for l in selected_layers]}")

    @patch('os.path.exists', return_value=True)
    @patch('app.core.fmri_processing.pipelines.choose_layer.select_visualization_layers', side_effect=lambda layers, strategy: MOCK_LLM_RESPONSES.get(strategy, MOCK_LLM_RESPONSES["default"]))
    @patch('app.core.fmri_processing.pipelines.inspect_model.inspect_torch_model', return_value=MOCK_ALL_LAYERS)
    @patch('app.core.fmri_processing.model_config.CapsNet3DAdapter.create_model', return_value=MagicMock())
    @patch('app.core.fmri_processing.model_config.PaperModelAdapter.create_model', return_value=MagicMock())
    def test_default_strategy_fallback_papermodel(self, mock_paper_model_create, mock_capsnet_create, mock_inspect_torch_model, mock_select_layers, mock_path_exists):
        print("\n--- Test Default Strategy Fallback (PaperModel) ---")
        # No layer_selection_strategy provided, should fall back to adapter's default
        pipeline = GenericInferencePipeline(
            model_config="papermodel"
        )
        selected_layers, _ = pipeline.inspect_and_select_layers()
        
        self.assertIsNotNone(selected_layers)
        self.assertGreater(len(selected_layers), 0)
        # Should call with the adapter's default strategy, which is now "improved_shufflenet"
        mock_select_layers.assert_called_with(MOCK_ALL_LAYERS, strategy="improved_shufflenet")
        print(f"Selected layers: {[l['model_path'] for l in selected_layers]}")


if __name__ == '__main__':
    unittest.main()
