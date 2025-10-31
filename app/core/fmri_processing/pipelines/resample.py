import nibabel as nib
from nilearn.image import resample_to_img # Only import resample_to_img
import os
from typing import Optional # For return type hint

def resample_activation_to_atlas( # This is the ORIGINAL function
    act_path: str,
    atlas_path: str,
    output_dir: str,
    interpolation: str = "linear" # Default to linear for heatmaps
) -> Optional[str]:
    """
    [UPDATED VERSION]
    Resamples an activation map (NIfTI) to match the exact grid (shape and affine) 
    of an atlas NIfTI, with proper coordinate system alignment.

    Args:
        act_path (str): Path to activation NIfTI (e.g., heatmap_3D_native.nii.gz)
        atlas_path (str): Path to atlas NIfTI (e.g., AAL3v1_1mm.nii.gz)
        output_dir (str): Directory to save resampled output
        interpolation (str): 'linear' (recommended for heatmaps) or 'nearest' (for labels)

    Returns:
        str: Output file path or None on failure
    """
    print("Resampling activation map to match atlas grid...")
    
    try:
        act_img = nib.load(act_path)
        atlas_img = nib.load(atlas_path)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return None

    # Check coordinate system alignment
    act_orient = nib.aff2axcodes(act_img.affine)
    atlas_orient = nib.aff2axcodes(atlas_img.affine)
    
    print(f"Activation orientation: {act_orient}")
    print(f"Atlas orientation: {atlas_orient}")
    
    # If orientations don't match, fix the coordinate system manually
    if act_orient != atlas_orient:
        print("⚠️  Coordinate systems don't match - applying coordinate transformation...")
        
        # Handle the specific case: L->R flip in first axis
        if act_orient[0] == 'L' and atlas_orient[0] == 'R':
            print("🔄 Flipping X-axis from L to R orientation...")
            import numpy as np
            
            # Get the data and flip along the first axis
            act_data = act_img.get_fdata()
            act_data_flipped = np.flip(act_data, axis=0)
            
            # Create new affine matrix for R orientation
            new_affine = act_img.affine.copy()
            # Flip the X direction in the affine matrix
            new_affine[0, 0] = -new_affine[0, 0]  # Flip X scaling
            new_affine[0, 3] = -new_affine[0, 3]  # Flip X origin
            
            # Create new image with flipped data and corrected affine
            act_img_reoriented = nib.Nifti1Image(act_data_flipped, new_affine, act_img.header)
            print("✅ Applied L->R coordinate transformation")
        else:
            print(f"⚠️  Unsupported orientation change: {act_orient} -> {atlas_orient}")
            print("Proceeding with original orientations...")
            act_img_reoriented = act_img
        
        atlas_img_canonical = atlas_img
    else:
        print("✅ Coordinate systems match")
        act_img_reoriented = act_img
        atlas_img_canonical = atlas_img

    # --- Resample to atlas ---
    try:
        resampled_img = resample_to_img(
            source_img=act_img_reoriented,  # Use reoriented activation
            target_img=atlas_img_canonical, # Use canonical atlas
            interpolation=interpolation,
            force_resample=True,            # Ensure resampling happens
            copy_header=True                # Copy atlas header info
        )
        print("✅ Resampling completed successfully")
    except Exception as e:
        print(f"Error during resampling to atlas: {e}")
        return None

    # --- Save result ---
    try:
        # Create a more descriptive filename
        base_name = os.path.basename(act_path).replace(".nii", "").replace(".gz", "")
        atlas_name = os.path.basename(atlas_path).split('.')[0] # Get atlas name part
        filename = f"{base_name}_resampled_to_{atlas_name}.nii.gz"
        
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        resampled_img.to_filename(output_path)
        print(f"Resampled NIfTI saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving resampled file: {e}")
        return None

# --- Main execution block (should call the function above) ---
if __name__ == "__main__":
    
    # Input: The MNI-normalized heatmap from the ANTs step
    MNI_HEATMAP_PATH = "output/single_subject_normalized_ants_masked/subject_008_ants_heatmap_MNI_masked_accurate.nii.gz" 

    # Target: Your AAL atlas in MNI space
    ATLAS_PATH = "data/aal3/AAL3v1_1mm.nii.gz" 
    
    # Output: Final heatmap precisely aligned with the atlas grid
    OUTPUT_DIR = "output/single_subject_final_resampled_accurate"
    
    # --- Check files ---
    if not os.path.exists(MNI_HEATMAP_PATH):
        print(f"Error: Input MNI heatmap not found at {MNI_HEATMAP_PATH}")
    elif not os.path.exists(ATLAS_PATH):
        print(f"Error: Atlas NIfTI not found at {ATLAS_PATH}")
    else:
        print("--- Running Final Resampling to Atlas Grid (Correct Version) ---")
        # Call the CORRECT function
        resampled_path = resample_activation_to_atlas( 
            act_path=MNI_HEATMAP_PATH,
            atlas_path=ATLAS_PATH,
            output_dir=OUTPUT_DIR,
            interpolation="linear" # Use linear for heatmaps
        )
        if resampled_path:
             print("--- Resampling Complete ---")
             # The actual filename is returned by the function
             print(f"Final atlas-aligned heatmap saved as: {os.path.basename(resampled_path)}") 
        else:
             print("--- Resampling Failed ---")