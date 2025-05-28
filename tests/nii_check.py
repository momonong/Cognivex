import os
import nibabel as nib
import numpy as np
from nilearn import datasets

# === 1. 設定你的資料夾路徑 ===
root_dir = "data/raw"  # <- 你資料夾的根目錄

# === 2. 載入 AAL atlas 當作 MNI 參考 ===
aal = datasets.fetch_atlas_aal()
atlas_img = nib.load(aal['maps'])
atlas_shape = atlas_img.shape
atlas_affine = atlas_img.affine
atlas_spacing = np.round(np.abs(np.diag(atlas_affine))[:3], 2)

print("🎯 AAL Atlas:")
print("Shape:", atlas_shape)
print("Voxel Spacing:", atlas_spacing)
print("Affine:\n", atlas_affine)
print("=" * 60)

# === 3. 遍歷資料夾並檢查每個 nii 檔 ===
def is_mni_compatible(affine, shape, atlas_affine, atlas_shape):
    spacing = np.round(np.abs(np.diag(affine))[:3], 2)
    atlas_spacing = np.round(np.abs(np.diag(atlas_affine))[:3], 2)

    same_shape = shape == atlas_shape
    similar_spacing = np.allclose(spacing, atlas_spacing, atol=0.5)

    return same_shape or similar_spacing

for root, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            file_path = os.path.join(root, file)
            try:
                img = nib.load(file_path)
                shape = img.shape
                affine = img.affine
                spacing = np.round(np.abs(np.diag(affine))[:3], 2)

                mni_ok = is_mni_compatible(affine, shape, atlas_affine, atlas_shape)
                status = "✅ MNI-like" if mni_ok else "❌ Not MNI-aligned"

                print(f"{file_path}")
                print(f"Shape: {shape} | Voxel Spacing: {spacing} → {status}")
                print("-" * 60)
            except Exception as e:
                print(f"⚠️ Failed to read {file_path}: {e}")
