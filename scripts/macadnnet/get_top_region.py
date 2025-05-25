import numpy as np
from nilearn import datasets
import nibabel as nib
from skimage.transform import resize
from collections import defaultdict

# === 1. 讀取 activation npy ===
act = np.load("output/activation_fc1.npy")  # shape: (1, 50, 9, 9)
act = np.squeeze(act)  # → shape: (50, 9, 9)
act_map = act.mean(axis=0)  # → shape: (9, 9)

# === 2. 讀取 AAL atlas ===
atlas = datasets.fetch_atlas_aal(version="SPM12")
aal_img = nib.load(atlas["maps"])
aal_data = aal_img.get_fdata()
aal_labels = atlas["labels"]
aal_indices = atlas["indices"]
label_dict = {int(k): v for k, v in zip(aal_indices, aal_labels)}

# === 3. 取某個 z slice（自己決定）並 resize
z_slice = 50
aal_slice = aal_data[:, :, z_slice]  # shape: (91, 109)
resized_aal = resize(
    aal_slice, act_map.shape, order=0, preserve_range=True, anti_aliasing=False
)
resized_aal = np.rint(resized_aal).astype(int)

# === 4. 每個 label 的活化平均
region_activation = defaultdict(list)
for i in range(act_map.shape[0]):
    for j in range(act_map.shape[1]):
        label_id = resized_aal[i, j]
        if label_id > 0:
            region_activation[label_id].append(act_map[i, j])

region_mean = {k: np.mean(v) for k, v in region_activation.items()}
top5 = sorted(region_mean.items(), key=lambda x: x[1], reverse=True)[:5]

# === 5. 印出 top region 名稱
top_named_regions = [(label_dict.get(k, f"ID:{k}"), v) for k, v in top5]

print("🧠 Top activated brain regions:")
for name, val in top_named_regions:
    print(f"{name}: {val:.4f}")

