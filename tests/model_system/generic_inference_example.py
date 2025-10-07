#!/usr/bin/env python3
"""
Generic Model Inference Example

This example demonstrates how to use the new generic inference pipeline
with different model types. It shows how easy it is to switch between
different models without changing the main pipeline code.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.fmri_processing.generic_pipeline_steps import GenericInferencePipeline
from app.core.fmri_processing.model_config import (
    ModelConfig, ModelType, ModelFactory,
    get_config_by_name, CAPSNET_CONFIG, MCADNNET_CONFIG
)

def example_1_using_predefined_config():
    """Example 1: Using predefined configurations"""
    print("=" * 60)
    print("Example 1: Using Predefined Configurations")
    print("=" * 60)
    
    # Use CapsNet with predefined config
    print("\n1.1 Using CapsNet with predefined config:")
    pipeline_capsnet = GenericInferencePipeline(
        model_config="capsnet",  # Use string name
        model_path="model/capsnet/best_capsnet_rnn.pth"  # Optional
    )
    
    print(f"Model type: {pipeline_capsnet.config.model_type}")
    print(f"Input shape: {pipeline_capsnet.config.input_shape}")
    print(f"Window size: {pipeline_capsnet.config.window_size}")
    print(f"Device: {pipeline_capsnet.config.device}")
    
    # Use MCADNNet with predefined config
    print("\n1.2 Using MCADNNet with predefined config:")
    pipeline_mcadn = GenericInferencePipeline(
        model_config="mcadnnet",
        model_path="model/mcadnnet/best_model.pth"
    )
    
    print(f"Model type: {pipeline_mcadn.config.model_type}")
    print(f"Input shape: {pipeline_mcadn.config.input_shape}")
    print(f"Window size: {pipeline_mcadn.config.window_size}")

def example_2_custom_config():
    """Example 2: Creating custom configurations"""
    print("\n" + "=" * 60)
    print("Example 2: Creating Custom Configurations") 
    print("=" * 60)
    
    # Create a custom CapsNet config with different parameters
    custom_capsnet_config = ModelConfig(
        model_type=ModelType.CAPSULE_3D,
        input_shape=(1, 1, 64, 64, 80),  # Different input shape
        window_size=3,  # Smaller window
        stride=2,       # Different stride
        preprocessing_params={
            "normalize_method": "z_score",  # Different normalization
            "time_axis": 3
        },
        inference_params={
            "threshold": 0.6,  # Higher threshold
            "aggregation": "max"  # Max instead of mean
        }
    )
    
    print("Custom CapsNet Configuration:")
    print(f"  Input shape: {custom_capsnet_config.input_shape}")
    print(f"  Window/Stride: {custom_capsnet_config.window_size}/{custom_capsnet_config.stride}")
    print(f"  Preprocessing: {custom_capsnet_config.preprocessing_params}")
    
    # Create pipeline with custom config
    custom_pipeline = GenericInferencePipeline(
        model_config=custom_capsnet_config,
        model_path="model/capsnet/custom_model.pth"
    )
    
    print(f"Pipeline created with custom config for {custom_pipeline.config.model_type.value}")

def example_3_model_inspection():
    """Example 3: Model inspection with different strategies"""
    print("\n" + "=" * 60)
    print("Example 3: Model Inspection with Different Strategies")
    print("=" * 60)
    
    # CapsNet inspection with capsule-focused strategy
    print("3.1 CapsNet Model Inspection:")
    capsnet_pipeline = GenericInferencePipeline("capsnet")
    
    try:
        selected_layers, layer_names = capsnet_pipeline.inspect_model_structure()
        print(f"Selected {len(selected_layers)} layers with capsule-focused strategy:")
        for layer in selected_layers:
            print(f"  - {layer['model_path']} ({layer['layer_type']}): {layer['reason']}")
    except Exception as e:
        print(f"Model inspection failed: {e}")
    
    # 2D CNN inspection with conv-focused strategy  
    print("\n3.2 MCADNNet Model Inspection:")
    mcadn_pipeline = GenericInferencePipeline("mcadnnet")
    
    try:
        selected_layers, layer_names = mcadn_pipeline.inspect_model_structure()
        print(f"Selected {len(selected_layers)} layers with conv-focused strategy:")
        for layer in selected_layers:
            print(f"  - {layer['model_path']} ({layer['layer_type']}): {layer['reason']}")
    except Exception as e:
        print(f"Model inspection failed: {e}")

def example_4_adding_new_model():
    """Example 4: How to add a new model type"""
    print("\n" + "=" * 60)
    print("Example 4: Adding a New Model Type")
    print("=" * 60)
    
    from app.core.fmri_processing.model_config import BaseModelAdapter, ModelType
    import torch
    import numpy as np
    
    # Define a new model type
    # (You would add this to the enum in model_config.py)
    print("4.1 Define new ModelType (conceptual):")
    print("    TRANSFORMER = \"transformer\"")
    
    # Create a new adapter class
    class TransformerAdapter(BaseModelAdapter):
        """Example adapter for a transformer model"""
        
        def create_model(self) -> torch.nn.Module:
            # This would create your transformer model
            print("    Creating transformer model...")
            return torch.nn.Identity()  # Placeholder
        
        def preprocess_data(self, data_path: str) -> torch.Tensor:
            print(f"    Preprocessing data for transformer: {data_path}")
            # Custom preprocessing for transformer
            return torch.randn(1, 100, 512)  # Example shape
        
        def get_layer_selection_strategy(self) -> str:
            return "attention_focused"
        
        def postprocess_prediction(self, model_output: torch.Tensor) -> str:
            print("    Postprocessing transformer output...")
            return "Example prediction"
    
    print("4.2 Created TransformerAdapter with methods:")
    print("    - create_model(): Creates transformer instance")
    print("    - preprocess_data(): Custom preprocessing")
    print("    - get_layer_selection_strategy(): Returns 'attention_focused'")
    print("    - postprocess_prediction(): Custom output processing")
    
    print("\n4.3 To use the new adapter:")
    print("    # Register the adapter")
    print("    ModelFactory.register_adapter(ModelType.TRANSFORMER, TransformerAdapter)")
    print("    ")
    print("    # Create config and use")
    print("    transformer_config = ModelConfig(")
    print("        model_type=ModelType.TRANSFORMER,")
    print("        input_shape=(1, 100, 512),")
    print("        window_size=1,")
    print("        stride=1")
    print("    )")

def example_5_backward_compatibility():
    """Example 5: Backward compatibility with existing code"""
    print("\n" + "=" * 60)
    print("Example 5: Backward Compatibility")
    print("=" * 60)
    
    from app.core.fmri_processing.generic_pipeline_steps import run_inference_and_classification
    
    # Simulate the state that would come from your existing workflow
    mock_state = {
        'subject_id': 'test_subject_001',
        'model_path': 'model/capsnet/best_capsnet_rnn.pth',
        'fmri_scan_path': 'data/raw/test_scan.nii.gz',
        'trace_log': []
    }
    
    print("5.1 Using the backward-compatible function:")
    print(f"Input state: {mock_state}")
    
    # This function can directly replace the original in your existing code
    print("\n5.2 Call run_inference_and_classification():")
    print("    result = run_inference_and_classification(state, model_config='capsnet')")
    print("    # This works exactly like your old function!")
    
    print("\n5.3 Switch to different model just by changing config:")
    print("    result = run_inference_and_classification(state, model_config='mcadnnet')")
    print("    # Same function, different model!")

def main():
    """Main function demonstrating all examples"""
    print("Generic Model Inference System Examples")
    print("======================================")
    
    try:
        example_1_using_predefined_config()
        example_2_custom_config()
        example_3_model_inspection()
        example_4_adding_new_model()
        example_5_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("Summary: Benefits of the Generic System")
        print("=" * 60)
        print("✓ Easy to switch between different model types")
        print("✓ No need to modify pipeline code for new models")
        print("✓ Model-specific preprocessing and postprocessing")
        print("✓ Strategy-based layer selection")
        print("✓ Backward compatibility with existing code")
        print("✓ Extensible through adapter pattern")
        print("✓ Configuration-driven approach")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()