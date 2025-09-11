import nibabel as nib
from nilearn import plotting, image
import numpy as np

# === [1] 載入 activation map 與 DMN mask ===
group_act_img = nib.load("output/group/group_mean_activation.nii.gz")
yeo_img = nib.load("data/yeo/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz")
yeo_data = yeo_img.get_fdata()

# === [2] 提取 DMN 對應的區域，建立 binary mask ===
dmn_mask_data = (yeo_data == 1).astype(np.int32)

# 修正維度
if dmn_mask_data.ndim == 4 and dmn_mask_data.shape[-1] == 1:
    dmn_mask_data = np.squeeze(dmn_mask_data, axis=-1)
elif dmn_mask_data.ndim != 3:
    raise ValueError(f"❌ DMN mask shape must be 3D. Got: {dmn_mask_data.shape}")

dmn_mask_img = nib.Nifti1Image(dmn_mask_data, affine=yeo_img.affine)

# === [3] 顯示 activation map 並疊加 DMN mask ===
# "plot_stat_map" 產生 activation display，搭配 "plot_roi" 疊加色塊
display = plotting.plot_stat_map(
    group_act_img,
    display_mode="mosaic",
    threshold=0.1,
    title="Group Activation with DMN Mask",
    annotate=True
)

# 這行移除！display.plot_roi(...) 是錯誤語法

# 直接呼叫 plot_roi 疊加 mask（透明色塊效果明顯）
# 請注意此語法會產生一個新的 figure（不會和 display 疊在同一個對象）
plotting.plot_roi(
    dmn_mask_img,
    bg_img=group_act_img,       # 以 activation map 當背景
    display_mode="mosaic",      # 與 stat_map 保持一致
    annotate=True,
    alpha=0.4,                  # 透明度：0~1
    cmap='autumn',              # 可換其他 colormap
    title="Group Activation with DMN Mask (with Color Mask)"
)

# 儲存用 plt.savefig，或 plot_roi(..., output_file="你要的.png")
# 例：plotting.plot_roi(..., output_file=output_path)

# 或用 plt.savefig('output.png') 捕捉當前圖形
import matplotlib.pyplot as plt
plt.savefig("figures/group/group_activation_with_dmn_color_block.png", dpi=150, bbox_inches="tight")
