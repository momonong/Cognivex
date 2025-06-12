import nibabel as nib
import pandas as pd
from collections import Counter
from nilearn.image import resample_to_img
import os

# === [1] 載入 AAL3 atlas（目標空間）===
aal_path = "data/aal3/AAL3v1_1mm.nii.gz"
aal_img = nib.load(aal_path)
aal_data = aal_img.get_fdata().astype(int)
print(f"📚 AAL atlas shape: {aal_data.shape}")

# === [2] 載入 Yeo7 atlas（原始為 256³，需要 resample）===
yeo_path = "data/yeo/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz"
yeo_img_orig = nib.load(yeo_path)

print("🔄 Resampling Yeo atlas to match AAL atlas space...")
yeo_img_resampled = resample_to_img(yeo_img_orig, aal_img, interpolation="nearest")
yeo_data = yeo_img_resampled.get_fdata().astype(int).squeeze()
print(f"✅ Resampled Yeo atlas shape: {yeo_data.shape}")

# === [3] 載入 AAL3 label 對照表 ===
id_to_label = {}
label_txt_path = "data/aal3/AAL3v1_1mm.nii.txt"
with open(label_txt_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            label_id = int(parts[0])
            label_name = " ".join(parts[1:])
            id_to_label[label_id] = label_name
print(f"📖 AAL labels loaded: {len(id_to_label)}")

# === [4] 建立 Yeo7 ID ➝ Network 名稱對照（人工設定）===
yeo_id_to_name = {
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal Attention",
    4: "Ventral Attention",
    5: "Limbic",
    6: "Frontoparietal",
    7: "Default"
}
print(f"📖 Yeo7 network labels loaded: {len(yeo_id_to_name)}")

## === [5] Region ➝ Network 對應統計 ===
region_to_network = {}
distribution_list = []  # 用於儲存詳細分布資訊

for region_id, region_name in id_to_label.items():
    region_mask = aal_data == region_id
    region_networks = yeo_data[region_mask]

    counts = Counter(region_networks)
    counts.pop(0, None)  # 忽略背景

    if counts:
        dominant_id, dominant_count = counts.most_common(1)[0]
        dominant_name = yeo_id_to_name.get(dominant_id, "Unknown")
        total = sum(counts.values())
        region_to_network[region_name] = (dominant_name, dominant_id, dominant_count, total)

        # 顯示分布前幾名（最多 3 個）
        print(f"\n🔍 {region_name} (ID: {region_id}) ➝ Top Yeo Networks:")
        for yeo_id, count in counts.most_common(3):
            yeo_name = yeo_id_to_name.get(yeo_id, "Unknown")
            percentage = round(count / total * 100, 2)
            print(f"   - {yeo_name:<18} (ID={yeo_id}) → {count} voxels ({percentage}%)")

        # 儲存全部分布紀錄
        for yeo_id, count in counts.items():
            yeo_name = yeo_id_to_name.get(yeo_id, "Unknown")
            percentage = round(count / total, 4)
            distribution_list.append({
                "region": region_name,
                "region_id": region_id,
                "network": yeo_name,
                "yeo_id": yeo_id,
                "voxels": count,
                "total_voxels": total,
                "percentage": percentage
            })
    else:
        region_to_network[region_name] = ("None", 0, 0, 0)

# === [6] 輸出 top-1 匯總 ===
print("\n🧠 AAL Region ➝ Yeo Network Mapping (top-1):\n")
print(f"{'Region':<30} {'Network':<20} {'Yeo ID':<7} {'Voxels':<7} {'Total Region Voxels'}")
print("-" * 80)
for region, (network, net_id, count, total) in region_to_network.items():
    print(f"{region:<30} {network:<20} {net_id:<7} {count:<7} {total}")

# === [7] 儲存為 CSV（top-1）===
summary_df = pd.DataFrame([
    {
        "region": region,
        "network": network,
        "yeo_id": net_id,
        "voxels": count,
        "total_voxels": total,
        "percentage": round(count / total, 4) if total > 0 else 0.0
    }
    for region, (network, net_id, count, total) in region_to_network.items()
])
summary_csv_path = "output/kg/aal3_to_yeo7_mapping_top1.csv"
summary_df.to_csv(summary_csv_path, index=False)
print(f"\n💾 Top-1 CSV exported to: {summary_csv_path}")

# === [8] 儲存為 CSV（完整分布）===
distribution_df = pd.DataFrame(distribution_list)
dist_csv_path = "output/kg/aal3_to_yeo7_mapping_distribution.csv"
distribution_df.to_csv(dist_csv_path, index=False)
print(f"💾 Full distribution CSV exported to: {dist_csv_path}")
