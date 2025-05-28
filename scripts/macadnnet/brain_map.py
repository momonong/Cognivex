import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import datasets, image, plotting
import nibabel as nib
from scipy.ndimage import zoom
from collections import defaultdict

# ---------- 設定 ----------
subject_id = "sub-14"
conv_dir = "output/macadnnet/conv_volume"
conv_path = os.path.join(conv_dir, f"{subject_id}_conv_volume.npy")
output_fig = f"figures/{subject_id}_top_regions_overlay.png"
top_n = 10

# ---------- 載入 volume ----------
conv_volume = np.load(conv_path)  # shape: [num_slices, 9, 9]

# ---------- 上採樣至 atlas 空間 ----------
target_shape = (91, 109, 91)
zoom_factors = (
    target_shape[0] / conv_volume.shape[0],
    target_shape[1] / conv_volume.shape[1],
    target_shape[2] / conv_volume.shape[2],
)
upsampled = zoom(conv_volume, zoom=zoom_factors, order=1)

# ---------- 載入 atlas ----------
atlas = datasets.fetch_atlas_aal()
atlas_img = image.load_img(atlas.maps)
atlas_data = atlas_img.get_fdata().astype(int)
atlas_affine = atlas_img.affine
label_dict = {int(k): v for k, v in zip(atlas.indices, atlas.labels)}

# ---------- 對應 activation 到 atlas 腦區 ----------
activation_by_region = {}
for label in np.unique(atlas_data):
    if label == 0: continue
    mask = atlas_data == label
    region_act = np.mean(upsampled[mask])
    activation_by_region[label] = region_act

# ---------- 取出 top-N activation 腦區 ----------
top_labels = sorted(activation_by_region.items(), key=lambda x: -x[1])[:top_n]
print(f"📌 Top {top_n} Activation Regions:")
for i, (label, act) in enumerate(top_labels, 1):
    name = label_dict.get(label, f"Label_{label}")
    print(f"{i:2d}. {name:<25} → activation = {act:.6f}")

# ---------- 建立 highlight mask ----------
highlight = np.zeros_like(atlas_data)
for label, _ in top_labels:
    highlight[atlas_data == label] = 1

highlight_img = nib.Nifti1Image(highlight.astype(np.uint8), affine=atlas_affine)

# ---------- 畫圖 ----------
display = plotting.plot_roi(
    highlight_img,
    bg_img=atlas_img,
    title=f"{subject_id} Top-{top_n} Activated AAL Regions",
    cut_coords=(0, 0, 0),
    display_mode='ortho',
    cmap='autumn',
)

display.savefig(output_fig)
display.close()
print(f"\n✅ 已儲存圖像：{output_fig}")
