import nibabel as nib
import numpy as np
import csv
from nilearn.image import resample_to_img

# === 檔案路徑 ===
AAL_PATH = "data/aal3/AAL3v1_1mm.nii.gz"
AAL_LABEL_PATH = "data/aal3/AAL3v1_1mm.nii.txt"
YEO_PATH = "data/yeo/Yeo_17Networks.nii.gz"

# === Yeo17 network ID ➝ 名稱對照 ===
yeo17_labels = {
    1: "Visual_A",
    2: "Visual_B",
    3: "Somatomotor_A",
    4: "Somatomotor_B",
    5: "Dorsal Attention_A",
    6: "Dorsal Attention_B",
    7: "Salience/Ventral Attention_A",
    8: "Salience/Ventral Attention_B",
    9: "Limbic_A",
    10: "Limbic_B",
    11: "Control_A",
    12: "Control_B",
    13: "Control_C",
    14: "Default_A",
    15: "Default_B",
    16: "Default_C",
    17: "Temporoparietal"
}

# === 載入影像 ===
aal_img = nib.load(AAL_PATH)
yeo_img = nib.load(YEO_PATH)

# === 載入 AAL label 對照表 ===
id_to_name = {}
with open(AAL_LABEL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            id_to_name[int(parts[0])] = parts[1]

# === 尺寸對齊 ===
if aal_img.shape != yeo_img.shape:
    print("⚠️ 尺寸不一致，執行 resample 對齊...")
    yeo_img = resample_to_img(yeo_img, aal_img, interpolation='nearest')

aal_data = aal_img.get_fdata().astype(int)
yeo_data = yeo_img.get_fdata().astype(int)

# === 建立對應 ===
mapping = []

for aal_id in np.unique(aal_data):
    if aal_id == 0:
        continue

    mask = aal_data == aal_id
    overlap_counts = np.bincount(yeo_data[mask].flatten(), minlength=18)
    overlap_counts[0] = 0  # 忽略背景

    if overlap_counts.sum() == 0:
        best_network = -1
        network_name = "Unmapped"
    else:
        best_network = np.argmax(overlap_counts)
        network_name = yeo17_labels.get(best_network, "Unknown")

    region_name = id_to_name.get(aal_id, f"Region_{aal_id}")
    mapping.append([aal_id, region_name, best_network, network_name])

# === 輸出 CSV ===
with open("output/aal3_to_yeo17.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["aal_id", "region_name", "yeo17_network_id", "yeo17_network_name"])
    writer.writerows(mapping)

print("✅ 完成！對應結果儲存於：output/aal3_to_yeo17.csv")
