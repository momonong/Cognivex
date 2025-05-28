import os
import numpy as np
import nibabel as nib
from nilearn import datasets, image, plotting
import matplotlib.pyplot as plt

# ---------- 設定 ----------
subject_id = "sub-14"
activation_path = f"output/macadnnet/conv_volume/{subject_id}_conv_volume.npy"
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)
# output_nii_path = os.path.join(output_dir, f"{subject_id}_highlight_volume.nii.gz")
output_png_path = os.path.join(output_dir, f"{subject_id}_highlight_map.png")
top_k = 10

# ---------- 載入 activation & mean over slices ----------
activation_volume = np.load(activation_path)  # shape: [n_slices, 9, 9]
activation_mean = activation_volume.mean(axis=0)  # shape: [9, 9]
activation_flat = activation_mean.flatten()       # shape: [81]

# ---------- 載入 AAL atlas ----------
aal = datasets.fetch_atlas_aal()
atlas_img = image.load_img(aal['maps'])
atlas_data = atlas_img.get_fdata()
atlas_affine = atlas_img.affine
labels = np.unique(atlas_data)[1:]  # AAL labels: 1~116
label_names = aal.labels  # e.g., ['Precentral_L', 'Frontal_Sup_L', ...]

# ---------- 模擬 label activation 對應（前 81 個 AAL 區塊） ----------
label_activation_map = {}
for i, label in enumerate(labels[:81]):
    label_activation_map[label] = activation_flat[i]

# ---------- 取 Top-K label ----------
sorted_labels = sorted(label_activation_map.items(), key=lambda x: x[1], reverse=True)
top_labels = [l for l, _ in sorted_labels[:top_k]]

print(f"\n📌 Top {top_k} Activated AAL Regions for {subject_id}:")
for rank, (label_id, act_value) in enumerate(sorted_labels[:top_k], 1):
    label_index = int(label_id) - 1
    if 0 <= label_index < len(label_names):
        name = label_names[label_index]
    else:
        name = f"Unknown (label {label_id})"
    print(f"{rank:2d}. {name:25s} → activation = {act_value:.6f}")


# ---------- 建立 highlight volume ----------
highlight_data = np.zeros_like(atlas_data)

for label in top_labels:
    mask = atlas_data == label
    highlight_data[mask] = label_activation_map[label]

# ---------- 儲存為 NIfTI ----------
highlight_img = nib.Nifti1Image(highlight_data.astype(np.float32), affine=atlas_affine)
# nib.save(highlight_img, output_nii_path)
# print(f"\n✅ 已儲存 NIfTI volume 到：{output_nii_path}")

# ---------- 視覺化 ----------
display = plotting.plot_stat_map(
    highlight_img,
    title=f"{subject_id} Top-{top_k} Activated AAL Regions",
    cut_coords=(0, 0, 0),
    display_mode='ortho',
    colorbar=True,
    threshold=np.percentile(highlight_data[highlight_data > 0], 10) if np.any(highlight_data > 0) else 0.0,
)
display.savefig(output_png_path)
display.close()
print(f"🧠 已儲存視覺化圖檔到：{output_png_path}")
