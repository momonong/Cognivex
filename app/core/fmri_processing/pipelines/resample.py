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
    [ORIGINAL VERSION]
    Resamples an activation map (NIfTI, assumed to be in MNI space)
    to match the exact grid (shape and affine) of an atlas NIfTI (also in MNI space).

    Args:
        act_path (str): Path to activation NIfTI (e.g., heatmap_3D_MNI_ants.nii.gz)
        atlas_path (str): Path to atlas NIfTI (e.g., AAL3v1_1mm.nii.gz)
        output_dir (str): Directory to save resampled output
        interpolation (str): 'linear' (recommended for heatmaps) or 'nearest' (for labels)

    Returns:
        str: Output file path or None on failure
    """
    print("Resampling MNI activation map to match atlas grid...")
    
    try:
        act_img = nib.load(act_path)
        atlas_img = nib.load(atlas_path)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return None

    # --- Resample directly to atlas ---
    try:
        resampled_img = resample_to_img(
            source_img=act_img,      # Input is the MNI heatmap
            target_img=atlas_img,    # Target is the atlas grid
            interpolation=interpolation,
            force_resample=True,     # Ensure resampling happens
            copy_header=True         # Copy atlas header info
        )
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