import nibabel as nib
from nilearn.image import resample_to_img
import os

def resample_activation_to_atlas(
    act_path: str,
    atlas_path: str,
    output_dir: str,
    interpolation: str = "nearest"
) -> str:
    """
    Resample activation map (NIfTI) to match the atlas space.

    Args:
        act_path (str): Path to activation NIfTI (.nii or .nii.gz)
        atlas_path (str): Path to atlas NIfTI
        output_dir (str): Directory to save resampled output
        interpolation (str): Interpolation method ('nearest' or 'continuous')

    Returns:
        str: Output file path to resampled activation map
    """
    print("Resampling activation to atlas space...")
    
    act_img = nib.load(act_path)
    atlas_img = nib.load(atlas_path)

    print("Original activation image")
    print(f"  shape:  {act_img.shape}")
    print(f"  affine:\n{act_img.affine}\n")

    print("Target atlas image")
    print(f"  shape:  {atlas_img.shape}")
    print(f"  affine:\n{atlas_img.affine}\n")

    # --- Resample ---
    resampled_img = resample_to_img(
        source_img=act_img,
        target_img=atlas_img,
        interpolation=interpolation,
        force_resample=True,
        copy_header=True
    )

    print("Resampled activation image")
    print(f"  shape:  {resampled_img.shape}")
    print(f"  affine:\n{resampled_img.affine}\n")

    # --- Save result ---
    filename = os.path.basename(act_path).replace(".nii", "").replace(".gz", "") + "_resampled.nii.gz"
    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    resampled_img.to_filename(output_path)

    print(f"Resampled NIfTI saved to: {output_path}")
    return output_path

# Example usage
if __name__ == "__main__":
    resample_activation_to_atlas(
        act_path="output/capsnet/module_test.nii.gz",
        atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
        output_dir="output/capsnet/resampled/module_test",
    )
