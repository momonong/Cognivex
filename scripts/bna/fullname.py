import pandas as pd
import os

# === 檔案路徑 ===
activation_csv_path = "output/bna_activation_yeo17.csv"  # 原始結果
fullname_csv_path = "data/bna/BNA_subregions.csv"  # 有全名的對照檔
output_csv_path = "data/bna/bna_activation_yeo17_full_named.csv"  # 輸出檔案

# === 讀取資料 ===
df_act = pd.read_csv(activation_csv_path)
df_full = pd.read_csv(fullname_csv_path)

# === 整理對照表 ===
# 去除空值並保留必要欄位
df_map = df_full[['Left and Right Hemisphere', 'Anatomical and modified Cyto-architectonic descriptions']].dropna()
df_map.rename(columns={
    'Left and Right Hemisphere': 'region',
    'Anatomical and modified Cyto-architectonic descriptions': 'region_fullname'
}, inplace=True)

# 移除多餘空格，避免合併失敗
df_map['region'] = df_map['region'].str.strip()
df_act['region'] = df_act['region'].str.strip()

# === 合併補全 ===
df_merged = df_act.merge(df_map, on='region', how='left')

# === 儲存結果 ===
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df_merged.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

print(f"✅ 補全完成，結果已儲存至 {output_csv_path}")
