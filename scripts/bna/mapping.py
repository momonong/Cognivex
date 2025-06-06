import nibabel as nib
import numpy as np
import pandas as pd
import os

# === [1] 讀取檔案路徑（請根據實際情況修改） ===
activation_path = "output/capsnet/sub-14_conv3_resampled_to_bna.nii.gz"
bna_atlas_path = "data/bna/BN_Atlas_246_1mm.nii.gz"
yeo_mapping_csv = "data/subregion_func_network_Yeo_updated.csv"
output_csv_path = "output/bna_activation_yeo17.csv"

# === [2] 讀取資料 ===
activation_nii = nib.load(activation_path)
activation_data = activation_nii.get_fdata()

bna_nii = nib.load(bna_atlas_path)
bna_data = bna_nii.get_fdata().astype(int)

df_mapping = pd.read_csv(yeo_mapping_csv)
df_mapping.columns = [col.strip() for col in df_mapping.columns]  # <== 修正欄位空白
df_mapping.rename(columns={"Label": "BNA_label"}, inplace=True)
print(df_mapping.columns)  # 檢查 df_mapping 的欄位


# === [3] 遍歷每個腦區，計算 activation 強度 ===
results = []
for label in np.unique(bna_data):
    if label == 0:
        continue
    mask = bna_data == label
    values = activation_data[mask]
    mean_act = values.mean()
    sum_act = values.sum()
    voxel_count = np.count_nonzero(mask)

    results.append({
        "BNA_label": label,
        "mean_activation": mean_act,
        "sum_activation": sum_act,
        "voxel_count": voxel_count
    })

df_act = pd.DataFrame(results)
print(df_act.columns)  # 檢查 df_act 的欄位

# === [4] 合併 BNA → Yeo17 對應表 ===
df_result = df_act.merge(df_mapping, on="BNA_label", how="left")

# === [5] 排序與儲存 ===
df_result_sorted = df_result.sort_values(by="sum_activation", ascending=False)

# Optional: 移除沒有對應的 label（Yeo_17network 為 0 的）
df_result_sorted = df_result_sorted[df_result_sorted["Yeo_17network"] != 0]

# 儲存 CSV
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df_result_sorted.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"✅ 對應完成，已輸出結果至 {output_csv_path}")
