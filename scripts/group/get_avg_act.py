import nibabel as nib
import numpy as np
import glob
import os

# === [1] 取得所有已 resampled 的 activation 檔案路徑 ===
paths = sorted(glob.glob("output/group/resampled/*.nii.gz"))
print(f"📂 共找到 {len(paths)} 個 activation maps.")

# === [2] 初始化儲存標準化後的資料 ===
z_maps = []
affine = None
header = None

# === [3] 對每個 subject 做 z-score 標準化並記錄統計資訊 ===
for i, path in enumerate(paths):
    img = nib.load(path)
    data = img.get_fdata()

    if affine is None:
        affine = img.affine
        header = img.header

    # 只處理 activation > 0 的 voxel
    mask = data > 0
    nonzero_data = data[mask]

    if nonzero_data.size == 0:
        print(f"⚠️ 檔案 {path} 沒有 activation > 0 的 voxel，跳過。")
        continue

    mean = np.mean(nonzero_data)
    std = np.std(nonzero_data)

    # 印出原始統計資訊
    print(f"\n📄 Subject {i+1:02d}: {os.path.basename(path)}")
    print(f"   ➤ Nonzero voxels : {nonzero_data.size}")
    print(f"   ➤ Mean           : {mean:.5f}")
    print(f"   ➤ Std            : {std:.5f}")

    # 建立 z-score map
    z_data = np.zeros_like(data)
    z_data[mask] = (data[mask] - mean) / std

    # 額外印出 z-score 的統計
    z_vals = z_data[mask]
    print(f"   ➤ Z-Score range  : min={z_vals.min():.4f}, max={z_vals.max():.4f}")
    print(f"   ➤ Z-Score stats  : mean={z_vals.mean():.4f}, std={z_vals.std():.4f}")

    z_maps.append(z_data)

# === [4] 計算 group-level 平均 ===
z_stack = np.stack(z_maps, axis=0)  # shape: (n_subjects, x, y, z)
mean_z = np.mean(z_stack, axis=0)

# 印出 group mean z-score 的統計資訊
group_mask = mean_z != 0
group_vals = mean_z[group_mask]
print(f"\n✅ Group-level z-score activation map 統計：")
print(f"   ➤ Shape          : {mean_z.shape}")
print(f"   ➤ Voxels > 0     : {group_vals.size}")
print(f"   ➤ Mean           : {group_vals.mean():.4f}")
print(f"   ➤ Std            : {group_vals.std():.4f}")
print(f"   ➤ Min/Max        : min={group_vals.min():.4f}, max={group_vals.max():.4f}")

# === [5] 儲存為 NIfTI 檔案 ===
output_path = "output/group/group_mean_activation_zscore.nii.gz"
mean_img = nib.Nifti1Image(mean_z, affine, header)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
nib.save(mean_img, output_path)
print(f"\n💾 已儲存 group-level z-score activation map 到：{output_path}")
