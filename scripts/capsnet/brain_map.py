from collections import Counter
import numpy as np
import nibabel as nib

# === [1] 載入 AAL3v1 atlas ===
atlas_path = "data/aal3/AAL3v1_1mm.nii.gz"
label_path = "data/aal3/AAL3v1_1mm.nii.txt"

atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata().astype(int)
print("📚 Atlas shape:", atlas_data.shape)  # ex: (181, 217, 181)

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

# === [2] 載入已經 resample 過的 activation map ===
act_img = nib.load("output/capsnet/sub-14_conv3_resampled_to_atlas.nii.gz")
act_data = act_img.get_fdata()
print("🧠 Activation map shape:", act_data.shape)  # ex: (181, 217, 181)

# === [3] 找出 activation > 0 的 voxel 的位置與值 ===
activated_voxels = np.where(act_data > 0)
print(f"🔍 共找到 {len(activated_voxels[0])} 個 activation > 0 的 voxel。")

# === 對應回 atlas 中的 label ID ===
atlas_labels = atlas_data[activated_voxels].astype(int)

# === DEBUG: 顯示前 10 個 voxel 與其對應 label ===
print("\n🧾 前 10 個 activation voxel 對應到的 atlas label：")
for i in range(min(10, len(atlas_labels))):
    coord = (activated_voxels[0][i], activated_voxels[1][i], activated_voxels[2][i])
    label_id = atlas_labels[i]
    label_name = id_to_label.get(label_id, "❌ Unknown")
    print(f"Voxel {i}: {coord} ➜ Label {label_id:>4}: {label_name}")

# === [4] 統計各腦區 label 出現的次數 ===
counts = Counter(atlas_labels)

print("\n🔍 Activation 對應腦區統計：\n")
for label_id, count in counts.most_common():
    if label_id == 0:
        continue  # 忽略背景
    label_name = id_to_label.get(label_id, "❌ Unknown")
    print(f"Label {label_id:>4}: {label_name:<30} | {count:>5} voxels")

# === [5] 額外印出 Atlas 中所有可用 label ID 數量（驗證用）===
print(f"\n📊 Atlas 中總共包含 {len(id_to_label)} 個腦區 label。")
