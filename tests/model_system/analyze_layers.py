#!/usr/bin/env python3
"""
Analyze Layer Selection for Visualization

This script analyzes the selected layers and provides recommendations
for better visualization results.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.fmri_processing.generic_pipeline_steps import GenericInferencePipeline

def analyze_capsnet_layers():
    """Analyze CapsNet layer selection"""
    print("=" * 60)
    print("CapsNet Layer Analysis & Recommendations")
    print("=" * 60)
    
    pipeline = GenericInferencePipeline("capsnet")
    
    # Get the model architecture
    model = pipeline.adapter.create_model()
    
    print("\n📊 CapsNet Architecture Summary:")
    print("-" * 40)
    for name, module in model.named_modules():
        if name == "":
            continue
        print(f"{name:25} -> {type(module).__name__}")
    
    print(f"\n🎯 Current Layer Selection (capsule_focused strategy):")
    print("-" * 50)
    
    # The layers we know are being selected
    current_selection = [
        ("capsnet.conv1", "Conv3d", "Early feature extraction"),
        ("capsnet.caps1", "CapsuleLayer3D", "First capsule layer"),  
        ("capsnet.caps2", "CapsuleLayer3D", "Second capsule layer")
    ]
    
    for layer_name, layer_type, description in current_selection:
        print(f"✓ {layer_name:20} ({layer_type:15}) - {description}")
    
    print(f"\n💡 Visualization Recommendations:")
    print("-" * 40)
    
    recommendations = [
        {
            "layer": "capsnet.conv1", 
            "good_for": "Low-level features (edges, textures)",
            "expected": "Simple spatial patterns, basic fMRI structures"
        },
        {
            "layer": "capsnet.conv3",
            "good_for": "High-level conv features before capsules", 
            "expected": "More complex spatial patterns, integrated features"
        },
        {
            "layer": "capsnet.caps1", 
            "good_for": "Part-whole relationships, spatial hierarchies",
            "expected": "Brain region assemblies, anatomical structures"
        },
        {
            "layer": "capsnet.caps2",
            "good_for": "Higher-level capsule representations",
            "expected": "Complex brain network patterns"
        }
    ]
    
    for rec in recommendations:
        print(f"\n🔍 {rec['layer']}:")
        print(f"   Good for: {rec['good_for']}")
        print(f"   Expected: {rec['expected']}")
    
    print(f"\n⚠️  Potential Issues with Current Selection:")
    print("-" * 45)
    issues = [
        "capsnet.conv1 might be too early - may show low-level noise",
        "Missing capsnet.conv3 - this could be more informative than conv1",
        "Both caps1 and caps2 might be redundant - consider choosing one",
        "Consider including capsnet.conv2 for mid-level features"
    ]
    
    for issue in issues:
        print(f"• {issue}")
    
    print(f"\n✨ Suggested Better Selection:")
    print("-" * 35)
    better_selection = [
        ("capsnet.conv2", "Mid-level convolutional features"),
        ("capsnet.conv3", "High-level convolutional features"), 
        ("capsnet.caps1", "Primary capsule representations")
    ]
    
    for layer, description in better_selection:
        print(f"✓ {layer:20} - {description}")

def analyze_mcadnnet_layers():
    """Analyze MCADNNet layer selection (when working)"""
    print("\n" + "=" * 60)
    print("MCADNNet Layer Analysis & Recommendations")
    print("=" * 60)
    
    print("\n📊 MCADNNet Architecture Summary:")
    print("-" * 40)
    
    # Based on the model code we saw
    mcadn_layers = [
        ("conv0", "Conv2d", "10 filters, 5x5 kernel", "Early feature extraction"),
        ("pool0", "MaxPool2d", "2x2 pooling", "Spatial downsampling"),
        ("conv1", "Conv2d", "20 filters, 5x5 kernel", "Mid-level features"),
        ("pool1", "MaxPool2d", "2x2 pooling", "Spatial downsampling"),
        ("conv2", "Conv2d", "50 filters, 5x5 kernel", "High-level features"),
        ("pool2", "MaxPool2d", "2x2 pooling", "Spatial downsampling"),
        ("fc1", "Linear", "256 features", "Classification prep"),
        ("fc2", "Linear", "2 classes", "Final classification")
    ]
    
    for layer_name, layer_type, details, description in mcadn_layers:
        print(f"{layer_name:10} ({layer_type:10}) - {details:20} - {description}")
    
    print(f"\n🎯 Recommended Layer Selection for Visualization:")
    print("-" * 52)
    
    recommended = [
        ("conv1", "Mid-level convolutional features", "Good balance of detail and abstraction"),
        ("conv2", "High-level convolutional features", "Most informative for classification")
    ]
    
    for layer, description, reasoning in recommended:
        print(f"✓ {layer:10} - {description}")
        print(f"  Reasoning: {reasoning}")
        print()
    
    print(f"⚠️  Avoid: conv0 (too low-level), fc1/fc2 (no spatial info)")

def provide_general_recommendations():
    """Provide general recommendations for layer selection"""
    print("\n" + "=" * 60)
    print("General Layer Selection Guidelines")
    print("=" * 60)
    
    guidelines = [
        {
            "principle": "Spatial Information Preservation",
            "details": "Choose layers that maintain spatial dimensions",
            "examples": "Conv layers > Fully connected layers"
        },
        {
            "principle": "Feature Complexity Balance", 
            "details": "Not too early (noise) or too late (over-abstracted)",
            "examples": "Middle to late conv layers are often best"
        },
        {
            "principle": "Model-Specific Considerations",
            "details": "Consider the unique aspects of your model architecture",
            "examples": "Capsule layers for CapsNets, Attention for Transformers"
        },
        {
            "principle": "Complementary Representations",
            "details": "Select layers that show different aspects of the data",
            "examples": "One early + one late layer"
        }
    ]
    
    for guideline in guidelines:
        print(f"\n🎯 {guideline['principle']}")
        print(f"   Details: {guideline['details']}")
        print(f"   Examples: {guideline['examples']}")
    
    print(f"\n🔧 How to Improve Layer Selection:")
    print("-" * 35)
    
    improvements = [
        "1. Run actual inference with current selection",
        "2. Generate activation maps and examine them visually", 
        "3. Check if maps show meaningful brain patterns",
        "4. If too noisy: choose later layers",
        "5. If too abstract: choose earlier layers",
        "6. Compare multiple layer selections side by side"
    ]
    
    for improvement in improvements:
        print(improvement)

def main():
    """Main analysis function"""
    print("Layer Selection Analysis for fMRI Visualization")
    print("==============================================")
    
    try:
        analyze_capsnet_layers()
        analyze_mcadnnet_layers()
        provide_general_recommendations()
        
        print(f"\n" + "=" * 60)
        print("Summary & Next Steps")
        print("=" * 60)
        
        print("\n✅ What's working:")
        print("• CapsNet model loads and layer selection works")
        print("• Generic system successfully abstracts different models")
        print("• Layer validation catches invalid selections")
        
        print("\n🔧 What needs attention:")  
        print("• MCADNNet device compatibility issues")
        print("• Layer selection might need refinement based on actual visualization")
        print("• Need to test with real fMRI data")
        
        print("\n🚀 Recommended next steps:")
        print("1. Fix MCADNNet device issues") 
        print("2. Run inference with actual fMRI data")
        print("3. Generate activation maps and review them")
        print("4. Adjust layer selection based on visual quality")
        print("5. Consider implementing multiple layer selection strategies")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()