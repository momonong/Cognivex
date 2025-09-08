import nibabel as nib
import numpy as np
import glob
import os

# === [1] Get all resampled activation file paths ===
paths = sorted(glob.glob("output/group/resampled/*.nii.gz"))
print(f"📂 Found {len(paths)} activation maps in total.")

# === [2] Initialize storage for normalized data ===
z_maps = []
affine = None
header = None

# === [3] Perform z-score normalization for each subject and record statistics ===
for i, path in enumerate(paths):
    img = nib.load(path)
    data = img.get_fdata()

    if affine is None:
        affine = img.affine
        header = img.header

    # Only process voxels with activation > 0
    mask = data > 0
    nonzero_data = data[mask]

    if nonzero_data.size == 0:
        print(f"⚠️ File {path} has no voxels with activation > 0, skipping.")
        continue

    mean = np.mean(nonzero_data)
    std = np.std(nonzero_data)

    # Print raw statistics
    print(f"\n📄 Subject {i+1:02d}: {os.path.basename(path)}")
    print(f"   ➤ Nonzero voxels : {nonzero_data.size}")
    print(f"   ➤ Mean           : {mean:.5f}")
    print(f"   ➤ Std            : {std:.5f}")

    # Create z-score map
    z_data = np.zeros_like(data)
    z_data[mask] = (data[mask] - mean) / std

    # Additionally print z-score statistics
    z_vals = z_data[mask]
    print(f"   ➤ Z-Score range  : min={z_vals.min():.4f}, max={z_vals.max():.4f}")
    print(f"   ➤ Z-Score stats  : mean={z_vals.mean():.4f}, std={z_vals.std():.4f}")

    z_maps.append(z_data)

# === [4] Compute group-level average ===
z_stack = np.stack(z_maps, axis=0)  # shape: (n_subjects, x, y, z)
mean_z = np.mean(z_stack, axis=0)

# Print statistics of group mean z-score
group_mask = mean_z != 0
group_vals = mean_z[group_mask]
print(f"\n✅ Group-level z-score activation map statistics:")
print(f"   ➤ Shape          : {mean_z.shape}")
print(f"   ➤ Voxels > 0     : {group_vals.size}")
print(f"   ➤ Mean           : {group_vals.mean():.4f}")
print(f"   ➤ Std            : {group_vals.std():.4f}")
print(f"   ➤ Min/Max        : min={group_vals.min():.4f}, max={group_vals.max():.4f}")

# === [5] Save as NIfTI file ===
output_path = "output/group/group_mean_activation_zscore.nii.gz"
mean_img = nib.Nifti1Image(mean_z, affine, header)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
nib.save(mean_img, output_path)
print(f"\n💾 Saved group-level z-score activation map to: {output_path}")
