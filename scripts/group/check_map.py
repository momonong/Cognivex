import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

# === [1] 載入 group mean activation map ===
group_act_img = nib.load("output/group/group_mean_activation.nii.gz")

# === [2] 顯示 activation map ===
display = plotting.plot_stat_map(
    group_act_img,
    display_mode="mosaic",   # 可選: "ortho", "z", "x", "y", "mosaic"
    threshold=0.1,           # 可調：用 0 表示不做 thresholding
    title="Group-Level Activation Map"
)
display.savefig("figures/group/group_activation_map_mosaic.png")
print("✅ Group activation map saved to figures/group/group_activation_map_mosaic.png")
