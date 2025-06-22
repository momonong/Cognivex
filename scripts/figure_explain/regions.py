import nibabel as nib
import numpy as np
from nilearn import plotting, image, datasets
import os

# --- 檔案與資料夾設定 ---
atlas_path = "data/yeo/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz"
output_dir = "figures/yeo7_networks"
os.makedirs(output_dir, exist_ok=True)

# --- Label 對應表 ---
yeo_labels = {
    1: "Visual",
    2: "Somatomotor",
    3: "DorsalAttention",
    4: "VentralAttention",
    5: "Limbic",
    6: "Control",
    7: "DefaultMode"
}

# --- 載入 atlas 並 resample 至 Nilearn 預設背景 ---
atlas_img = nib.load(atlas_path)
template_img = datasets.load_mni152_template()
resampled_atlas = image.resample_to_img(atlas_img, template_img, interpolation='nearest')
resampled_data = resampled_atlas.get_fdata()

# --- 每一個 label 產出獨立圖像 ---
for label_value, label_name in yeo_labels.items():
    mask_data = (resampled_data == label_value).astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, affine=resampled_atlas.affine)

    out_path = os.path.join(output_dir, f"{label_name}.png")
    
    display = plotting.plot_roi(
        roi_img=mask_img,
        bg_img=template_img,
        title=f"Yeo7 - {label_name} Network",
        display_mode="mosaic",
        cmap="coolwarm",
        alpha=0.7,
        cut_coords=12  # 控制切片數量，也可以改成 list
    )
    display.savefig(out_path, dpi=150)
    display.close()
    print(f"✅ Saved: {out_path}")
