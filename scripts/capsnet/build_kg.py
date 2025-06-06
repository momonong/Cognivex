import pandas as pd
import os

# === [0] 檔案與資料夾路徑 ===
mapping_path = "output/aal3_to_yeo17.csv"  # 替換為你真實路徑
output_dir = "output/neo4j"
os.makedirs(output_dir, exist_ok=True)

# === [1] 載入對應資料 ===
df = pd.read_csv(mapping_path)
df["yeo17_network_name"] = df["yeo17_network_name"].fillna("Unmapped")
df["yeo17_network_id"] = df["yeo17_network_id"].fillna(-1).astype(int)

# === [2] 節點：Region ===
df_region = df[["aal_id", "region_name"]].drop_duplicates().rename(columns={
    "aal_id": "id:ID(Region)",
    "region_name": "name"
})
df_region["label:LABEL"] = "Region"
df_region.to_csv(f"{output_dir}/nodes_region.csv", index=False, encoding="utf-8-sig")

# === [3] 節點：Network ===
df_network = df[["yeo17_network_id", "yeo17_network_name"]].drop_duplicates().rename(columns={
    "yeo17_network_id": "id:ID(Network)",
    "yeo17_network_name": "name"
})
df_network["label:LABEL"] = "Network"
df_network.to_csv(f"{output_dir}/nodes_network.csv", index=False, encoding="utf-8-sig")

# === [4] 邊：Region ➝ Network ===
df_r2n = df.rename(columns={
    "aal_id": ":START_ID(Region)",
    "yeo17_network_id": ":END_ID(Network)"
})
df_r2n[":TYPE"] = "BELONGS_TO"
df_r2n = df_r2n[[":START_ID(Region)", ":END_ID(Network)", ":TYPE"]]
df_r2n.to_csv(f"{output_dir}/edges_region_to_network.csv", index=False, encoding="utf-8-sig")

# === [5] 模擬性醫學知識資料 ===
region_to_effect = {
    "Angular_R": "Language and number processing",
    "Frontal_Sup_Medial_R": "Executive dysfunction",
    "Temporal_Mid_R": "Memory impairment",
    "Hippocampus_L": "Short-term memory loss",
    "Cingulate_Mid_R": "Emotional regulation issues",
}
network_to_ad = {
    "Default_A": "Default Mode Network disruption is associated with memory decline",
    "Salience/Ventral Attention_A": "Salience network alteration may affect attention and early AD detection",
    "Limbic_A": "Limbic degeneration correlates with emotional and memory impairment",
}

# === [6] 節點：Effect（症狀）===
df_effect = pd.DataFrame(region_to_effect.items(), columns=["region_name", "effect"])
df_effect["id:ID(Effect)"] = df_effect["effect"]
df_effect["name"] = df_effect["effect"]
df_effect["label:LABEL"] = "Effect"
df_effect = df_effect[["id:ID(Effect)", "name", "label:LABEL"]].drop_duplicates()
df_effect.to_csv(f"{output_dir}/nodes_effect.csv", index=False, encoding="utf-8-sig")

# === [7] 邊：Region ➝ Effect ===
df_r2e = pd.DataFrame(region_to_effect.items(), columns=["region_name", ":END_ID(Effect)"])
df_r2e = df_r2e.merge(df[["region_name", "aal_id"]].drop_duplicates(), on="region_name", how="left")
df_r2e = df_r2e.rename(columns={"aal_id": ":START_ID(Region)"})
df_r2e[":TYPE"] = "ASSOCIATED_WITH"
df_r2e = df_r2e[[":START_ID(Region)", ":END_ID(Effect)", ":TYPE"]]
df_r2e.to_csv(f"{output_dir}/edges_region_to_effect.csv", index=False, encoding="utf-8-sig")

# === [8] 節點：AD_Effect（AD 影響）===
df_ad = pd.DataFrame(network_to_ad.items(), columns=["network_name", "impact"])
df_ad["id:ID(AD_Effect)"] = df_ad["impact"]
df_ad["name"] = df_ad["impact"]
df_ad["label:LABEL"] = "AD_Effect"
df_ad = df_ad[["id:ID(AD_Effect)", "name", "label:LABEL"]].drop_duplicates()
df_ad.to_csv(f"{output_dir}/nodes_ad_effect.csv", index=False, encoding="utf-8-sig")

# === [9] 邊：Network ➝ AD_Effect ===
df_n2ad = pd.DataFrame(network_to_ad.items(), columns=["network_name", ":END_ID(AD_Effect)"])
df_n2ad = df_n2ad.merge(df[["yeo17_network_name", "yeo17_network_id"]].drop_duplicates(),
                        left_on="network_name", right_on="yeo17_network_name", how="left")
df_n2ad = df_n2ad.rename(columns={"yeo17_network_id": ":START_ID(Network)"})
df_n2ad[":TYPE"] = "AFFECTS_AD"
df_n2ad = df_n2ad[[":START_ID(Network)", ":END_ID(AD_Effect)", ":TYPE"]].drop_duplicates()
df_n2ad.to_csv(f"{output_dir}/edges_network_to_ad_effect.csv", index=False, encoding="utf-8-sig")

print("✅ 已成功產出 7 個 Neo4j 可匯入的 CSV：")
print(" - nodes_region.csv")
print(" - nodes_network.csv")
print(" - nodes_effect.csv")
print(" - nodes_ad_effect.csv")
print(" - edges_region_to_network.csv")
print(" - edges_region_to_effect.csv")
print(" - edges_network_to_ad_effect.csv")

