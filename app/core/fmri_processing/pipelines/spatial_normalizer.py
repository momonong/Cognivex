# File: app/core/xai/spatial_normalizer.py
import ants
import os
import time
from typing import Optional

def normalize_native_heatmap_to_mni_accurate_masked( # Renamed function
    t1_native_path: str,
    heatmap_native_path: str,
    mni_template_path: str, # Should be skull-stripped MNI T1
    output_prefix: str, 
    transform_type: str = 'SyN', 
    interpolator: str = 'linear'
) -> Optional[str]: 
    """
    [ACCURATE VERSION v3 - WITH MASKING]
    Normalizes native heatmap to MNI space using multi-stage ANTs 
    with initial brain extraction for improved accuracy.
    (Concise logging version)
    """
    print(f"\n--- Running Accurate Spatial Normalization (ANTs + Brain Masking) ---")
    start_total_time = time.time()
    
    output_dir = os.path.dirname(output_prefix)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    
    final_output_path = f"{output_prefix}_heatmap_MNI_masked_accurate.nii.gz"

    # --- Check Input Files ---
    if not all(os.path.exists(p) for p in [t1_native_path, heatmap_native_path, mni_template_path]):
        print("  Error: One or more input files not found.")
        return None

    try:
        # --- Load Images ---
        # print("  Loading images...") # Removed for brevity
        fixed_mni_brain = ants.image_read(mni_template_path) # Skull-stripped template
        moving_t1_full = ants.image_read(t1_native_path)     # Original T1 with skull
        heatmap_native_ants = ants.image_read(heatmap_native_path)

        # --- Step 1: Brain Extraction (Masking) ---
        print("  Step 1: Performing brain extraction on native T1...")
        # start_time = time.time() # Removed for brevity
        try:
            moving_t1_mask = ants.get_mask(moving_t1_full, low_thresh=moving_t1_full.mean() * 0.3, high_thresh=moving_t1_full.max(), cleanup=2)
            moving_t1_brain = ants.mask_image(moving_t1_full, moving_t1_mask)
            # masking_time = time.time() - start_time # Removed for brevity
            # print(f"  Brain extraction finished in {masking_time:.2f} seconds.") # Removed for brevity
        except Exception as mask_error:
             print(f"  Warning: ants.get_mask failed ({mask_error}). Proceeding with unmasked T1.")
             moving_t1_brain = moving_t1_full # Fallback

        # --- Step 2: Multi-Stage Registration (Masked T1 -> Masked MNI) ---
        print(f"  Step 2: Calculating {transform_type} registration (this may take time)...")
        # start_time = time.time() # Removed for brevity
        
        transform = ants.registration(
             fixed=fixed_mni_brain,
             moving=moving_t1_brain,
             type_of_transform=transform_type, 
             aff_metric='Mattes',            
             aff_sampling=32,                
             aff_random_sampling_rate=0.2,   
             aff_iterations=(1000, 500, 250, 100), 
             syn_metric='CC',                
             syn_sampling=2,                 
             reg_iterations=(100, 100, 70, 50, 20), 
             # --- THIS IS THE KEY CHANGE ---
             verbose=False # Set to False to suppress detailed ANTs output
             # --- END OF KEY CHANGE ---
        )
        
        fwd_transforms = transform['fwdtransforms'] 
        # print(f"  Forward transforms calculated: {fwd_transforms}") # Removed for brevity
        # reg_time = time.time() - start_time # Removed for brevity
        # print(f"  Registration calculated in {reg_time:.2f} seconds.") # Removed for brevity
        
        # --- Step 3: Apply Combined Transform to Heatmap ---
        print(f"  Step 3: Applying transform to heatmap (Interpolator: {interpolator})...")
        # start_time = time.time() # Removed for brevity
        
        heatmap_normalized = ants.apply_transforms(
             fixed=fixed_mni_brain,
             moving=heatmap_native_ants,
             transformlist=fwd_transforms,
             interpolator=interpolator         
        )
        # apply_time = time.time() - start_time # Removed for brevity
        # print(f"  Transform applied in {apply_time:.2f} seconds.") # Removed for brevity
        
        # --- Step 4: Save Normalized Heatmap ---
        ants.image_write(heatmap_normalized, final_output_path)
        print(f"  Normalized heatmap saved: {final_output_path}")
        
        # --- Step 5: (Optional) Save Warped T1 for QC ---
        warped_t1_path = f"{output_prefix}_t1_warped_to_MNI_masked.nii.gz"
        print(f"  QC T1 warped image saved: {warped_t1_path}")
        
        ants.apply_transforms(
             fixed=fixed_mni_brain, 
             moving=moving_t1_full,
             transformlist=fwd_transforms, 
             interpolator='linear', 
             output_filename=warped_t1_path,
             # Suppress verbose output from apply_transforms as well
             verbose=False
        )

        total_time = time.time() - start_total_time
        print(f"--- Accurate Normalization Finished in {total_time:.2f} seconds ---")
        return final_output_path # Return the path

    except Exception as e:
        print(f"  Error during ANTs accurate normalization: {e}")
        import traceback
        traceback.print_exc() 
        return None


# --- Main Test Block (Calls the new masked function) ---
if __name__ == "__main__":
    print("--- Starting ANTs Accurate Normalization Test (with Masking) ---")
    
    # --- Configuration ---
    T1_NATIVE_PATH = "/Volumes/3T-disk/fMRI/Model/sMRI_data/AD/T1_3D_MPRAGE_SAG_0003_008/T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii" 
    # Use the native heatmap generated previously
    HEATMAP_NATIVE_PATH = "output/pipeline_test_run_test_subject_008/test_subject_008_backbone_stage4_1_gconv2_native_heatmap.nii.gz" 
    MNI_TEMPLATE_PATH = "data/affine/mni152_template.nii.gz" # Skull-stripped
    
    OUTPUT_PREFIX = "output/single_subject_normalized_ants_masked/subject_008_ants" # New output folder/prefix

    # --- Check Input Files ---
    if not all(os.path.exists(p) for p in [T1_NATIVE_PATH, HEATMAP_NATIVE_PATH, MNI_TEMPLATE_PATH]):
        print("Error: One or more input files not found. Check paths.")
        exit()

    # --- Run the Accurate Normalization Function with Masking ---
    print("\nRunning normalize_native_heatmap_to_mni_accurate_masked (Concise Log)...")
    start_time_total = time.time()
    
    final_output_file = normalize_native_heatmap_to_mni_accurate_masked(
        t1_native_path=T1_NATIVE_PATH,
        heatmap_native_path=HEATMAP_NATIVE_PATH,
        mni_template_path=MNI_TEMPLATE_PATH, 
        output_prefix=OUTPUT_PREFIX, 
        transform_type='SyN', 
        interpolator='linear'
    )
    
    end_time_total = time.time()
    
    # --- Verification ---
    print("\n--- Test Summary ---")
    if final_output_file and os.path.exists(final_output_file):
        print("✅ SUCCESS: Accurate Normalization with Masking completed.")
        print(f"   Normalized Heatmap Output: {final_output_file}")
        print(f"   (Check for transform files and warped T1 QC near '{OUTPUT_PREFIX}')")
        print(f"   Total Time: {end_time_total - start_time_total:.2f} seconds.")
    else:
        print("❌ FAILURE: Accurate Normalization with Masking process failed.")
        print(f"   Check ANTs configuration and error messages above.")