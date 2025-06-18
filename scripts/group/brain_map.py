from collections import defaultdict
import numpy as np
import nibabel as nib

# === [1] 載入 AAL3v1 atlas ===
atlas_path = "data/aal3/AAL3v1_1mm.nii.gz"
label_path = "data/aal3/AAL3v1_1mm.nii.txt"

atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata().astype(int)
print("📚 Atlas shape:", atlas_data.shape)

# === 讀取 label ID → 名稱對照表 ===
id_to_label = {}
with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            label_id = int(parts[0])
            label_name = " ".join(parts[1:])
            id_to_label[label_id] = label_name
print(f"📖 讀入 {len(id_to_label)} 筆腦區 label 對照表。")

# === [2] 載入 group-level activation map ===
act_img = nib.load("output/group/group_mean_activation.nii.gz")
act_data = act_img.get_fdata()
print("🧠 Activation map shape:", act_data.shape)

# === [3] 找出 activation > 0 的 voxel 的位置 ===
activated_voxels = np.where(act_data > 0)
act_values = act_data[activated_voxels]
atlas_labels = atlas_data[activated_voxels]

print(f"🔍 共找到 {len(act_values)} 個 activation > 0 的 voxel。")

# === DEBUG: 顯示前 10 個對應關係 ===
print("\n🧾 前 10 個 activation voxel 對應到的 atlas label：")
for i in range(min(10, len(act_values))):
    coord = (activated_voxels[0][i], activated_voxels[1][i], activated_voxels[2][i])
    label_id = int(atlas_labels[i])
    label_name = id_to_label.get(label_id, "❌ Unknown")
    print(f"Voxel {i}: {coord} ➜ Label {label_id:>4}: {label_name}")

# === [4] 累積每個腦區的 activation 總和與 voxel 數量 ===
label_activation_sum = defaultdict(float)
label_voxel_count = defaultdict(int)

for label_id, act_value in zip(atlas_labels, act_values):
    label_id = int(label_id)
    if label_id == 0:
        continue  # 跳過背景
    label_activation_sum[label_id] += act_value
    label_voxel_count[label_id] += 1

# === [5] 列出每個腦區的平均 activation（由高至低排序） ===
print("\nActivation 平均值統計（依 activation 排序）：\n")
sorted_labels = sorted(
    label_activation_sum.keys(),
    key=lambda k: (label_activation_sum[k] / label_voxel_count[k]) if label_voxel_count[k] > 0 else 0,
    reverse=True
)

for label_id in sorted_labels:
    name = id_to_label.get(label_id, "❌ Unknown")
    total = label_activation_sum[label_id]
    count = label_voxel_count[label_id]
    avg = total / count if count > 0 else 0
    print(f"Label {label_id:>4}: {name:<23} | Voxel={count:>5} | Total={total:.2f} | Mean={avg:.4f}")

# === [6] 額外印出 Atlas 腦區總數 ===
print(f"\n📊 Atlas 中總共包含 {len(id_to_label)} 個腦區 label。")
