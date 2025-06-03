import nibabel as nib

path = "output/nifti_activation/sub-21_activation_map.nii.gz"
img = nib.load(path)
data = img.get_fdata()

# 取出特定位置的值
t, z, y, x = 156, 24, 6, 3
val = data[x, y, z, t]  # 注意 nibabel 的順序為 (X, Y, Z, T)

print(f"Voxel value at (t={t}, z={z}, y={y}, x={x}):", val)
