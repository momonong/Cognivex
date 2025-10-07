#!/usr/bin/env python3
"""
Test Layer Selection and Visualization

This script tests the layer selection for different models and shows
the detailed layer information to help debug visualization issues.
"""

import sys
import os
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.fmri_processing.generic_pipeline_steps import GenericInferencePipeline
from app.core.fmri_processing.model_config import get_config_by_name, ModelFactory
from app.core.fmri_processing.pipelines.inspect_model import inspect_torch_model

def test_model_inspection(model_name: str):
    """Test model inspection for a specific model"""
    print(f"\n{'='*60}")
    print(f"Testing Model: {model_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Create pipeline
        pipeline = GenericInferencePipeline(model_name)
        config = pipeline.config
        
        print(f"Model Type: {config.model_type.value}")
        print(f"Input Shape: {config.input_shape}")
        print(f"Window/Stride: {config.window_size}/{config.stride}")
        print(f"Device: {config.device}")
        
        # Create model
        model = pipeline.adapter.create_model()
        print(f"Model created successfully: {type(model).__name__}")
        
        # Get all layer information
        print(f"\n--- All Model Layers ---")
        inspect_input_shape = config.input_shape[1:]  # Remove batch dimension
        print(f"Inspection input shape: {inspect_input_shape}")
        
        all_layers = inspect_torch_model(model, inspect_input_shape, config.device)
        
        print(f"Total layers found: {len(all_layers)}")
        for i, layer in enumerate(all_layers):
            print(f"  {i+1:2d}. {layer['model_path']:25} | {layer['layer_type']:15} | {layer['output_shape']:20} | {layer['params']:>8} params")
        
        # Test layer selection
        print(f"\n--- Layer Selection ---")
        selected_layers, layer_names = pipeline.inspect_model_structure()
        
        print(f"Strategy: {pipeline.adapter.get_layer_selection_strategy()}")
        print(f"Selected {len(selected_layers)} layers:")
        
        for layer in selected_layers:
            print(f"  ✓ {layer['model_path']}")
            print(f"    Type: {layer['layer_type']}")
            print(f"    Reason: {layer['reason']}")
            print()
        
        # Validate layers
        print(f"--- Layer Validation ---")
        validated_layers = pipeline.validate_layers(selected_layers, all_layers)
        print(f"Final validated layers: {len(validated_layers)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layer_selection_strategies():
    """Test different layer selection strategies"""
    print(f"\n{'='*60}")
    print("Testing Layer Selection Strategies")
    print(f"{'='*60}")
    
    # Test strategies with CapsNet
    from app.core.fmri_processing.pipelines.choose_layer import select_visualization_layers, STRATEGY_INSTRUCTIONS
    from app.core.fmri_processing.generic_pipeline_steps import GenericInferencePipeline
    
    pipeline = GenericInferencePipeline("capsnet")
    model = pipeline.adapter.create_model()
    config = pipeline.config
    
    all_layers = inspect_torch_model(model, config.input_shape[1:], config.device)
    
    for strategy_name, instruction in STRATEGY_INSTRUCTIONS.items():
        print(f"\n--- Strategy: {strategy_name} ---")
        try:
            response = select_visualization_layers(all_layers, strategy=strategy_name)
            selected_layers = json.loads(response)
            
            print(f"Selected {len(selected_layers)} layers:")
            for layer in selected_layers:
                print(f"  • {layer['model_path']} ({layer['layer_type']})")
                print(f"    Reason: {layer['reason'][:80]}...")
                print()
                
        except Exception as e:
            print(f"  Error with strategy {strategy_name}: {e}")

def show_model_architectures():
    """Show detailed model architectures"""
    print(f"\n{'='*60}")
    print("Model Architectures Comparison")
    print(f"{'='*60}")
    
    models_to_test = ["capsnet", "mcadnnet"]
    
    for model_name in models_to_test:
        print(f"\n--- {model_name.upper()} Architecture ---")
        try:
            pipeline = GenericInferencePipeline(model_name)
            model = pipeline.adapter.create_model()
            
            print("Model structure:")
            for name, module in model.named_modules():
                if name == "":  # Skip root
                    continue
                print(f"  {name:30} -> {type(module).__name__}")
                
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")

def main():
    """Main test function"""
    print("Layer Selection and Visualization Test")
    print("=====================================")
    
    # Test individual models
    models_to_test = ["capsnet", "mcadnnet"]
    results = {}
    
    for model_name in models_to_test:
        results[model_name] = test_model_inspection(model_name)
    
    # Test layer selection strategies
    test_layer_selection_strategies()
    
    # Show model architectures
    show_model_architectures()
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_name:15} | {status}")
    
    print("\n🔍 Next Steps:")
    print("1. Check if selected layers make sense for visualization")
    print("2. Test with actual fMRI data")
    print("3. Generate and review activation maps")
    print("4. Adjust layer selection strategies if needed")

if __name__ == "__main__":
    main()