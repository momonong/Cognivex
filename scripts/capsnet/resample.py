import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np

# Load images
act_path = "output/capsnet/sub-14_conv3_strongest_masked.nii.gz"
atlas_path = "data/aal3/AAL3v1_1mm.nii.gz"

act_img = nib.load(act_path)
atlas_img = nib.load(atlas_path)

print("🔍 原始 activation image")
print(f"shape: {act_img.shape}")
print(f"affine:\n{act_img.affine}\n")

print("📚 目標 atlas image")
print(f"shape: {atlas_img.shape}")
print(f"affine:\n{atlas_img.affine}\n")

# Resample
resampled_act_img = resample_to_img(
    source_img=act_img,
    target_img=atlas_img,
    interpolation='nearest'
)

print("✅ Resample 完成後")
print(f"shape: {resampled_act_img.shape}")
print(f"affine:\n{resampled_act_img.affine}\n")

# Save result
resampled_path = "output/capsnet/sub-14_conv3_resampled_to_atlas.nii.gz"
resampled_act_img.to_filename(resampled_path)
print("📦 Resampled activation map saved to:", resampled_path)
