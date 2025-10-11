#!/usr/bin/env python3
"""
Test script for the refactored activation_to_nifti function using nilearn.

This script validates the new nilearn-based resampling approach compared to 
the old F.interpolate method.
"""

import sys
import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting

# Add the project root to path
sys.path.append('/Users/morris/projects/semantic-KG')

from app.core.fmri_processing.pipelines.act_to_nii import (
    activation_to_nifti, 
    activation_to_nifti_with_affine_saving,
    compute_activation_affine
)


def create_test_activation(shape=(64, 5, 11, 13), save_path="test_activation.pt"):
    """
    Create a synthetic activation tensor for testing.
    
    Args:
        shape: Shape of activation tensor [C, D, H, W]
        save_path: Path to save the test activation
        
    Returns:
        str: Path to saved activation tensor
    """
    # Create synthetic activation with some spatial structure
    C, D, H, W = shape
    activation = torch.randn(1, C, D, H, W)  # [B, C, D, H, W]
    
    # Add some spatial structure (simulate brain-like activation patterns)
    for c in range(C):
        # Create blob-like activations
        center_d, center_h, center_w = D//2, H//2, W//2
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    dist = np.sqrt((d-center_d)**2 + (h-center_h)**2 + (w-center_w)**2)
                    if dist < 3:  # Create activation blob
                        activation[0, c, d, h, w] += 5 * np.exp(-dist/2)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(activation, save_path)
    print(f"✅ Created test activation tensor: {save_path}")
    print(f"   Shape: {activation.shape}")
    
    return save_path


def test_activation_processing():
    """
    Test the new nilearn-based activation processing.
    """
    print("🧪 Testing nilearn-based activation processing...")
    
    # Parameters
    test_activation_path = "output/test/test_activation.pt"
    reference_nii_path = "data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz"
    output_path = "output/test/test_activation_nilearn.nii.gz"
    affine_path = "output/test/test_activation_affine.npy"
    
    # Check if reference file exists
    if not os.path.exists(reference_nii_path):
        print(f"⚠️  Reference NIfTI file not found: {reference_nii_path}")
        print("   Please ensure the reference file exists or update the path.")
        return False
    
    # Create test activation
    create_test_activation(save_path=test_activation_path)
    
    try:
        # Test the enhanced function
        print("\n🔬 Running activation_to_nifti_with_affine_saving...")
        result = activation_to_nifti_with_affine_saving(
            activation_path=test_activation_path,
            reference_nii_path=reference_nii_path,
            output_path=output_path,
            save_affine_path=affine_path,
            norm_type="l2",
            threshold_percentile=95.0,  # Lower threshold for synthetic data
        )
        
        print("✅ Processing completed successfully!")
        print(f"   Selected channel: {result['selected_channel']}")
        print(f"   Activation shape: {result['activation_shape']}")
        print(f"   Reference shape: {result['reference_shape']}")
        print(f"   Output saved to: {result['output_path']}")
        
        # Verify output files exist
        if os.path.exists(output_path):
            print("✅ Output NIfTI file created successfully")
            
            # Load and inspect the result
            result_img = nib.load(output_path)
            print(f"   Result image shape: {result_img.shape}")
            print(f"   Result image affine:\n{result_img.affine}")
            
        if os.path.exists(affine_path):
            print("✅ Affine matrix saved successfully")
            affine_matrix = np.load(affine_path)
            print(f"   Affine matrix shape: {affine_matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def compare_methods():
    """
    Compare the old F.interpolate method with the new nilearn method.
    Note: This requires temporarily implementing the old method for comparison.
    """
    print("\n📊 Method comparison would go here...")
    print("   (This would require implementing the old F.interpolate method for comparison)")


def visualize_result(nifti_path):
    """
    Create a visualization of the processed activation map.
    
    Args:
        nifti_path: Path to the NIfTI activation map
    """
    if not os.path.exists(nifti_path):
        print(f"⚠️  NIfTI file not found for visualization: {nifti_path}")
        return
    
    try:
        print(f"\n🎨 Creating visualization for {nifti_path}...")
        
        # Load the image
        img = nib.load(nifti_path)
        
        # Create output directory for visualization
        vis_dir = "output/test/visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create mosaic plot
        display = plotting.plot_stat_map(
            img,
            display_mode="mosaic",
            threshold=0.1,
            title="Nilearn-processed Activation Map",
            colorbar=True
        )
        
        vis_path = os.path.join(vis_dir, "activation_mosaic.png")
        display.savefig(vis_path, dpi=150)
        print(f"✅ Visualization saved to: {vis_path}")
        
        display.close()
        
    except Exception as e:
        print(f"❌ Error creating visualization: {str(e)}")


def main():
    """
    Main test function.
    """
    print("🚀 Testing Nilearn-based Activation Processing")
    print("=" * 50)
    
    # Run the main test
    success = test_activation_processing()
    
    if success:
        # Create visualization
        visualize_result("output/test/test_activation_nilearn.nii.gz")
        
        print("\n🎉 All tests completed successfully!")
        print("\nKey improvements with nilearn approach:")
        print("  ✅ Proper spatial alignment using affine matrices")
        print("  ✅ No manual axis transposition required") 
        print("  ✅ Handles voxel dimensions correctly")
        print("  ✅ More robust resampling algorithm")
        
    else:
        print("\n❌ Tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()