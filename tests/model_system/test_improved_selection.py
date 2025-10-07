#!/usr/bin/env python3
"""
Test Improved Layer Selection Strategies

This script tests the improved layer selection strategies and compares
them with the original strategies.
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.fmri_processing.generic_pipeline_steps import GenericInferencePipeline
from app.core.fmri_processing.pipelines.choose_layer import select_visualization_layers, STRATEGY_INSTRUCTIONS
from app.core.fmri_processing.pipelines.inspect_model import inspect_torch_model

def compare_layer_selection_strategies():
    """Compare different layer selection strategies"""
    print("=" * 70)
    print("Layer Selection Strategy Comparison")
    print("=" * 70)
    
    # Get CapsNet layers for testing
    pipeline = GenericInferencePipeline("capsnet")
    model = pipeline.adapter.create_model()
    config = pipeline.config
    all_layers = inspect_torch_model(model, config.input_shape[1:], config.device)
    
    strategies_to_test = [
        "capsule_focused",      # Original
        "improved_capsule",     # New improved
        "conv_focused", 
        "improved_conv",        # New improved
        "default"
    ]
    
    results = {}
    
    for strategy in strategies_to_test:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            if strategy in STRATEGY_INSTRUCTIONS:
                response = select_visualization_layers(all_layers, strategy=strategy)
                selected_layers = json.loads(response)
                results[strategy] = selected_layers
                
                print(f"Selected {len(selected_layers)} layers:")
                for layer in selected_layers:
                    print(f"  ✓ {layer['model_path']:20} ({layer['layer_type']:15})")
                    print(f"    Reason: {layer['reason'][:80]}...")
                    print()
            else:
                print(f"  Strategy {strategy} not found in STRATEGY_INSTRUCTIONS")
                
        except Exception as e:
            print(f"  ❌ Error with strategy {strategy}: {e}")
            results[strategy] = None
    
    return results

def analyze_selection_improvements():
    """Analyze the improvements in layer selection"""
    print("\n" + "=" * 70)
    print("Selection Improvement Analysis")
    print("=" * 70)
    
    results = compare_layer_selection_strategies()
    
    # Compare original vs improved strategies
    comparisons = [
        ("capsule_focused", "improved_capsule", "CapsNet Strategies"),
        ("conv_focused", "improved_conv", "CNN Strategies")
    ]
    
    for original, improved, category in comparisons:
        print(f"\n🔍 {category}:")
        print("-" * 40)
        
        if original in results and improved in results and results[original] and results[improved]:
            original_layers = [layer['model_path'] for layer in results[original]]
            improved_layers = [layer['model_path'] for layer in results[improved]]
            
            print(f"Original ({original}):")
            for layer in original_layers:
                print(f"  • {layer}")
                
            print(f"\nImproved ({improved}):")  
            for layer in improved_layers:
                print(f"  • {layer}")
            
            # Analysis
            removed = set(original_layers) - set(improved_layers)
            added = set(improved_layers) - set(original_layers)
            kept = set(original_layers) & set(improved_layers)
            
            if removed:
                print(f"\n  ❌ Removed layers: {', '.join(removed)}")
            if added:
                print(f"  ✅ Added layers: {', '.join(added)}")
            if kept:
                print(f"  ↔️  Kept layers: {', '.join(kept)}")
                
        else:
            print(f"  Unable to compare {original} vs {improved} (missing results)")

def test_with_actual_pipeline():
    """Test the improved strategies with the actual pipeline"""
    print("\n" + "=" * 70)
    print("Testing Improved Strategies in Pipeline")
    print("=" * 70)
    
    models_to_test = ["capsnet"]  # Skip mcadnnet for now due to device issues
    
    for model_name in models_to_test:
        print(f"\n--- Testing {model_name.upper()} with Improved Strategy ---")
        try:
            pipeline = GenericInferencePipeline(model_name)
            
            # Test layer inspection with improved strategy
            selected_layers, layer_names = pipeline.inspect_model_structure()
            
            print(f"Strategy used: {pipeline.adapter.get_layer_selection_strategy()}")
            print(f"Selected layers:")
            for layer in selected_layers:
                print(f"  ✓ {layer['model_path']:20} - {layer['reason'][:60]}...")
            
            # Test validation
            if pipeline.model is None:
                pipeline.model = pipeline.adapter.create_model()
            all_layers = inspect_torch_model(
                pipeline.model, 
                pipeline.config.input_shape[1:], 
                pipeline.config.device
            )
            validated_layers = pipeline.validate_layers(selected_layers, all_layers)
            
            print(f"\nValidation result: {len(validated_layers)}/{len(selected_layers)} layers passed")
            
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {e}")

def provide_visualization_recommendations():
    """Provide specific recommendations for visualization"""
    print("\n" + "=" * 70)
    print("Visualization Recommendations")
    print("=" * 70)
    
    recommendations = {
        "CapsNet Improved Selection": {
            "layers": ["capsnet.conv2", "capsnet.conv3", "capsnet.caps1"],
            "rationale": [
                "conv2: Mid-level features with good spatial detail",
                "conv3: High-level features before capsule processing", 
                "caps1: Primary capsule representations with part-whole relationships"
            ],
            "expected_visualization": [
                "conv2: Structural brain patterns, tissue boundaries",
                "conv3: Complex anatomical features, regional assemblies",
                "caps1: Brain networks, functional connectivity patterns"
            ]
        },
        "MCADNNet Improved Selection": {
            "layers": ["conv1", "conv2"],
            "rationale": [
                "conv1: Mid-level features, good balance of detail/abstraction",
                "conv2: High-level features, most relevant for classification"
            ],
            "expected_visualization": [
                "conv1: Brain regions, anatomical structures",
                "conv2: Disease-relevant patterns, classification features"
            ]
        }
    }
    
    for model, rec in recommendations.items():
        print(f"\n🎯 {model}:")
        print("-" * (len(model) + 5))
        
        print("📋 Recommended layers:")
        for layer in rec["layers"]:
            print(f"  • {layer}")
        
        print("\n🧠 Rationale:")
        for rationale in rec["rationale"]:
            print(f"  • {rationale}")
            
        print("\n👁️  Expected visualization:")
        for viz in rec["expected_visualization"]:
            print(f"  • {viz}")

def main():
    """Main testing function"""
    print("Testing Improved Layer Selection Strategies")
    print("==========================================")
    
    try:
        # Run all tests
        analyze_selection_improvements()
        test_with_actual_pipeline()
        provide_visualization_recommendations()
        
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        
        print("\n✅ Improvements implemented:")
        print("• Added improved_capsule strategy - avoids early layers, focuses on conv2/conv3 + caps1")
        print("• Added improved_conv strategy - focuses on mid-to-late conv layers")
        print("• Updated model adapters to use improved strategies")
        
        print("\n🎯 Expected benefits:")
        print("• Better visualization quality - less noise, more meaningful patterns")
        print("• Reduced redundancy - avoid selecting similar layers")
        print("• Focus on classification-relevant features")
        
        print("\n📋 Next steps:")
        print("1. Run actual fMRI inference with improved selection")
        print("2. Generate and compare activation maps")
        print("3. Evaluate visualization quality visually")
        print("4. Fine-tune strategies based on results")
        
    except Exception as e:
        print(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()