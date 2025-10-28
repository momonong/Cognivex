import nibabel as nib
from nilearn.image import resample_to_img, resample_img # Import resample_img
from nilearn import datasets # To fetch a template
import os
from typing import Optional

def resample_activation_to_atlas_affine_approx( # Renamed function
    act_path: str,
    atlas_path: str,
    output_dir: str,
    interpolation: str = "linear" # Changed default interpolation
) -> Optional[str]:
    """
    [MODIFIED SCRIPT - APPROXIMATION]
    Attempts to resample activation map (NIfTI in native space) to match the 
    atlas space (MNI) using an intermediate affine resampling to an MNI template.
    
    WARNING: This is an approximation and less accurate than proper nonlinear normalization.
    
    Returns:
        str: Output file path or None on failure
    """
    print("Attempting affine resampling of activation to atlas space...")
    
    try:
        act_img = nib.load(act_path)
        atlas_img = nib.load(atlas_path)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return None
        
    # --- Step 1: Affine Resample to MNI Template Space (Approximation) ---
    print("Fetching MNI template...")
    # Fetch a standard MNI template (e.g., MNI152 2mm)
    # Using a template with similar resolution to the atlas might be better
    try: 
        template = datasets.load_mni152_template(resolution=1) # Get 1mm MNI template
        # Alternatively, use atlas_img itself as the target for affine registration
        # target_affine = atlas_img.affine
        # target_shape = atlas_img.shape
        target_affine = template.affine
        target_shape = template.shape
        
        print(f"Resampling (affine) activation to template space (Shape: {target_shape})...")
        # Use resample_img for affine transformation only
        act_img_mni_approx = resample_img(
            act_img,
            target_affine=target_affine,
            target_shape=target_shape,
            interpolation='linear' # Use linear for continuous data
        )
    except Exception as e:
        print(f"Error during initial affine resampling: {e}")
        return None

    # --- Step 2: Resample to the exact Atlas Space ---
    print(f"Resampling MNI-approximated activation to exact atlas space (Shape: {atlas_img.shape})...")
    # Now use resample_to_img to match the atlas precisely
    # Use the specified interpolation for the final step
    try:
        resampled_final_img = resample_to_img(
            source_img=act_img_mni_approx,
            target_img=atlas_img,
            interpolation=interpolation, 
            force_resample=True,
            copy_header=True # Copy header info from atlas
        )
    except Exception as e:
        print(f"Error during final resampling to atlas: {e}")
        return None

    # --- Save result ---
    try:
        filename = os.path.basename(act_path).replace(".nii", "").replace(".gz", "") + "_affine_resampled.nii.gz"
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        resampled_final_img.to_filename(output_path)
        print(f"Resampled NIfTI saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving resampled file: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Use the output from the *previous* step
    ACTIVATION_NIFTI = "output/papermodel_test/heatmap_3D_final.nii.gz" 
    ATLAS_NIFTI = "data/aal3/AAL3v1_1mm.nii.gz" # Your MNI atlas
    OUTPUT_DIR = "output/papermodel_test/resampled_affine" # New output dir

    if not os.path.exists(ACTIVATION_NIFTI):
        print(f"Error: Input activation NIfTI not found at {ACTIVATION_NIFTI}")
    elif not os.path.exists(ATLAS_NIFTI):
        print(f"Error: Atlas NIfTI not found at {ATLAS_NIFTI}")
    else:
        resample_activation_to_atlas_affine_approx(
            act_path=ACTIVATION_NIFTI,
            atlas_path=ATLAS_NIFTI,
            output_dir=OUTPUT_DIR,
            interpolation="linear" # Use linear for heatmaps
        )