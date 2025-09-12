import nibabel as nib
from nilearn import plotting, image
import numpy as np

# === [1] 載入 activation map 與 DMN mask ===
group_act_img = nib.load("output/group/group_mean_activation.nii.gz")
yeo_img = nib.load("data/yeo/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz")  # 需你確認實際路徑
yeo_data = yeo_img.get_fdata()

# === [2] 提取 DMN 對應的區域，建立 binary mask ===
dmn_mask_data = (yeo_data == 7).astype(np.int32)

# 修正 4D 問題
if dmn_mask_data.ndim == 4 and dmn_mask_data.shape[-1] == 1:
    dmn_mask_data = np.squeeze(dmn_mask_data, axis=-1)
elif dmn_mask_data.ndim != 3:
    raise ValueError(f"❌ DMN mask shape must be 3D. Got: {dmn_mask_data.shape}")

dmn_mask_img = nib.Nifti1Image(dmn_mask_data, affine=yeo_img.affine)


# === [3] 顯示 activation map 並疊加 DMN mask ===
display = plotting.plot_stat_map(
    group_act_img,
    display_mode="mosaic",
    threshold=0.1,
    title="Group Activation with DMN Mask",
    annotate=True
)

# === [4] 疊加輪廓（contour）===
display.add_contours(dmn_mask_img, colors='magenta', linewidths=1.5)

# === [5] 儲存結果圖 ===
output_path = "figures/group/group_activation_with_dmn.png"
display.savefig(output_path)
print(f"✅ 圖已儲存至 {output_path}")