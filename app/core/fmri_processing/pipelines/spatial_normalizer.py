# File: app/core/xai/spatial_normalizer.py
import ants
import os
import time
from typing import Optional

def normalize_native_heatmap_to_mni(
    t1_native_path: str,
    heatmap_native_path: str,
    mni_template_path: str,
    output_path: str,
    transform_type: str = 'SyN',
    interpolator: str = 'linear'
) -> bool:
    """
    Normalizes a native-space NIfTI heatmap to MNI space using ANTs.

    Args:
        t1_native_path (str): Path to the original T1 NIfTI in native space.
        heatmap_native_path (str): Path to the heatmap NIfTI in native space.
        mni_template_path (str): Path to the MNI T1 template NIfTI.
        output_path (str): Path to save the normalized MNI-space heatmap.
        transform_type (str): ANTs registration type (e.g., 'SyN', 'Affine').
        interpolator (str): ANTs interpolation type (e.g., 'linear', 'nearestNeighbor').

    Returns:
        bool: True if normalization was successful, False otherwise.
    """
    print(f"\n--- Running Spatial Normalization (ANTs) ---")
    print(f"  Native T1: {os.path.basename(t1_native_path)}")
    print(f"  Native Heatmap: {os.path.basename(heatmap_native_path)}")
    print(f"  MNI Template: {os.path.basename(mni_template_path)}")
    
    # --- Check Input Files ---
    if not all(os.path.exists(p) for p in [t1_native_path, heatmap_native_path, mni_template_path]):
        print("Error: One or more input files not found.")
        return False

    try:
        # --- Load Images ---
        print("  Loading images for ANTs...")
        fixed_mni = ants.image_read(mni_template_path)
        moving_t1 = ants.image_read(t1_native_path)
        heatmap_native_ants = ants.image_read(heatmap_native_path)

        # --- Calculate Transformation ---
        print(f"  Calculating registration (Transform: {transform_type})...")
        start_time = time.time()
        transform = ants.registration(fixed=fixed_mni, 
                                      moving=moving_t1, 
                                      type_of_transform=transform_type) 
        reg_time = time.time() - start_time
        print(f"  Registration calculated in {reg_time:.2f} seconds.")
        
        # --- Apply Transformation to Heatmap ---
        print(f"  Applying transform to heatmap (Interpolator: {interpolator})...")
        start_time = time.time()
        heatmap_normalized = ants.apply_transforms(fixed=fixed_mni, 
                                                   moving=heatmap_native_ants,
                                                   transformlist=transform['fwdtransforms'],
                                                   interpolator=interpolator)
        apply_time = time.time() - start_time
        print(f"  Transform applied in {apply_time:.2f} seconds.")
        
        # --- Save Normalized Heatmap ---
        output_dir = os.path.dirname(output_path)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        ants.image_write(heatmap_normalized, output_path)
        print(f"  Normalized heatmap saved to: {output_path}")
        return True

    except Exception as e:
        print(f"  Error during ANTs normalization: {e}")
        return False


if __name__ == "__main__":
    print("--- Starting ANTs Normalization Test ---")
    import numpy as np
    import nibabel as nib
    from typing import Optional, Tuple
    # --- Configuration ---
    TEST_OUTPUT_DIR = "output/test_ants_normalization"
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # Define mock file paths
    MOCK_T1_PATH = os.path.join(TEST_OUTPUT_DIR, "mock_t1_native.nii.gz")
    MOCK_HEATMAP_PATH = os.path.join(TEST_OUTPUT_DIR, "mock_heatmap_native.nii.gz")
    MNI_TEMPLATE_PATH = "data/affine/mni152_template.nii.gz" # Use a known template path

    # Final output path for verification
    FINAL_OUTPUT_PATH = os.path.join(TEST_OUTPUT_DIR, "heatmap_MNI_final.nii.gz")

    # 1. Create Mock Input Files
    # Note: Using small shapes for speed, but ANTs prefers realistic shapes.
    # We will use slightly larger, more typical MNI-like shapes here for better registration
    NATIVE_SHAPE = (180, 200, 180) # Example of a common T1 size
    
    def create_mock_nifti(path: str, shape: Tuple[int, int, int], value: int = 100) -> bool:
        """Creates a simple NIfTI file with a known affine for testing."""
        if os.path.exists(path):
            return True
        
        # Create a simple, known affine matrix (e.g., identity matrix with 1mm spacing)
        # This affine is crucial for ANTs to work.
        affine = np.array([
            [-1, 0, 0, 90],
            [0, 1, 0, -126],
            [0, 0, 1, -72],
            [0, 0, 0, 1]
        ])
        
        # Create random data and center a non-zero block for better registration likelihood
        data = np.zeros(shape, dtype=np.float32)
        center = [s // 4 for s in shape]
        data[center[0]:center[0]+shape[0]//2, center[1]:center[1]+shape[1]//2, center[2]:center[2]+shape[2]//2] = value

        try:
            img = nib.Nifti1Image(data, affine)
            nib.save(img, path)
            print(f"Created mock NIfTI: {os.path.basename(path)} with shape {shape}")
            return True
        except Exception as e:
            print(f"Error creating mock NIfTI {path}: {e}")
            return False

    # Check if MNI Template path exists (critical path check)
    if not os.path.exists(MNI_TEMPLATE_PATH):
        print(f"CRITICAL ERROR: MNI template not found at {MNI_TEMPLATE_PATH}")
        print("Please ensure this path is correct and the file exists on your system.")
        exit()
        
    # Create T1 and Heatmap mocks
    success_mock_t1 = create_mock_nifti(MOCK_T1_PATH, NATIVE_SHAPE, value=150)
    success_mock_heatmap = create_mock_nifti(MOCK_HEATMAP_PATH, NATIVE_SHAPE, value=0.8) # Heatmap values are small (0-1)

    if not (success_mock_t1 and success_mock_heatmap):
        print("Test aborted due to mock file creation failure.")
        exit()

    # 2. Run the Normalization Function
    print("\nRunning normalize_native_heatmap_to_mni...")
    
    start_time_total = time.time()
    
    normalization_success = normalize_native_heatmap_to_mni(
        t1_native_path=MOCK_T1_PATH,
        heatmap_native_path=MOCK_HEATMAP_PATH,
        mni_template_path=MNI_TEMPLATE_PATH, # Should be the T1 Template
        output_path=FINAL_OUTPUT_PATH,
        transform_type='Affine', # Use Affine for faster testing than SyN
        interpolator='linear'
    )
    
    end_time_total = time.time()
    
    # 3. Verification
    print("\n--- Test Summary ---")
    if normalization_success and os.path.exists(FINAL_OUTPUT_PATH):
        print("✅ SUCCESS: Normalization completed and output file was saved.")
        print(f"   Output: {FINAL_OUTPUT_PATH}")
        print(f"   Total Time: {end_time_total - start_time_total:.2f} seconds.")
    else:
        print("❌ FAILURE: Normalization process failed or output file not found.")
        print(f"   Check ANTs configuration and output folder.")