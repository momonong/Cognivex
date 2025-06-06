import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np
import os

# === 檔案路徑 ===
activation_path = "output/capsnet/sub-14_conv3_strongest_masked.nii.gz"
bna_path = "data/bna/BN_Atlas_246_1mm.nii.gz"
resampled_path = "output/capsnet/sub-14_conv3_resampled_to_bna.nii.gz"

# === 讀取影像 ===
act_img = nib.load(activation_path)
bna_img = nib.load(bna_path)

# === 若尺寸不同就 resample ===
if act_img.shape != bna_img.shape:
    print("🔁 尺寸不同，開始 resample...")
    act_img = resample_to_img(act_img, bna_img, interpolation="nearest")
    act_img.to_filename(resampled_path)
    print("✅ Resampled activation saved to:", resampled_path)
else:
    print("✅ 尺寸已對齊，無需 resample")
