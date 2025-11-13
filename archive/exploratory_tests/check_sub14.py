import os
import numpy as np
import nibabel as nib

# ---------- 設定 ----------
SUBJECT_ID = "sub-14"
ACT_NII_PATH = f"output/nifti_activation/{SUBJECT_ID}_activation_map.nii.gz"
Z = 24
T = 156
X = 6
Y = 3

# ---------- 讀取 NIfTI ----------
if not os.path.exists(ACT_NII_PATH):
    print(f"❌ Activation map not found: {ACT_NII_PATH}")
else:
    nii_img = nib.load(ACT_NII_PATH)
    data = nii_img.get_fdata()
    print(f"✅ Loaded NIfTI shape: {data.shape}")
    print(f"📦 Max value in volume: {data.max():.6f}")

    # 查詢指定 voxel
    if Z < data.shape[2] and T < data.shape[3]:
        voxel_val = data[X, Y, Z, T]
        print(f"🎯 Voxel value at (t={T}, z={Z}, y={Y}, x={X}): {voxel_val:.6f}")
    else:
        print(f"⚠️ Invalid z/t index: z={Z}, t={T}, shape={data.shape}")

    # 尋找最大 activation voxel
    max_val = data.max()
    max_pos = np.unravel_index(np.argmax(data), data.shape)
    x_max, y_max, z_max, t_max = max_pos
    print(f"🔍 Max activation location: (t={t_max}, z={z_max}, y={y_max}, x={x_max})")
    print(f"⭐ Max voxel value: {max_val:.6f}")
