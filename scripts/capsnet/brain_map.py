from collections import Counter
import numpy as np
import nibabel as nib

# === [1] 載入 AAL3v1 atlas ===
atlas_path = "data/aal3/AAL3v1_1mm.nii.gz"
label_path = "data/aal3/AAL3v1_1mm.nii.txt"

atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata().astype(int)

# 讀取 label ID → 腦區名稱 對照表
id_to_label = {}
with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            label_id = int(parts[0])
            label_name = " ".join(parts[1:])
            id_to_label[label_id] = label_name

# === [2] 載入 activation map ===
act_img = nib.load("output/capsnet/sub-14_conv3_resampled_to_atlas.nii.gz")
act_data = act_img.get_fdata()

# === [3] 找出 activation > 0 的 voxel 對應的 atlas label ID ===
activated_voxels = np.where(act_data > 0)
atlas_labels = atlas_data[activated_voxels].astype(int)

# === [4] 統計各腦區 label 出現的次數 ===
counts = Counter(atlas_labels)

print("🔍 Activation 對應腦區統計：\n")
for label_id, count in counts.most_common():
    if label_id == 0:
        continue  # 忽略背景
    label_name = id_to_label.get(label_id, "❌ Unknown")
    print(f"Label {label_id:>4}: {label_name:<30} | {count:>5} voxels")

# === [5] 額外印出 Atlas 中所有可用 label ID 數量（驗證用）===
print(f"\n📊 Atlas 中總共包含 {len(id_to_label)} 個腦區 label。")
