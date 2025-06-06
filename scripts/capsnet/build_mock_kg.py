import pandas as pd
import os

# === 建立資料夾 ===
output_dir = "output/neo4j/manual_annotation"
os.makedirs(output_dir, exist_ok=True)

# === 你手動整理的資料（先示範幾筆，可自行擴充） ===
data = [
    {
        "region_id": 70,
        "region_name": "Angular_R",
        "network_name": "Default_C",
        "function": "Language and number processing",
        "impact": "Early decline in DMN activity"
    },
    {
        "region_id": 4,
        "region_name": "Frontal_Sup_2_R",
        "network_name": "Control_C",
        "function": "Executive control",
        "impact": "Planning and attention issues"
    },
    {
        "region_id": 66,
        "region_name": "Parietal_Inf_R",
        "network_name": "Control_C",
        "function": "Spatial attention, executive",
        "impact": "AD-related spatial neglect"
    },
    {
        "region_id": 64,
        "region_name": "Parietal_Sup_R",
        "network_name": "Dorsal Attention_A",
        "function": "Top-down attention",
        "impact": "Visual neglect"
    },
    {
        "region_id": 20,
        "region_name": "Frontal_Sup_Medial_R",
        "network_name": "Temporoparietal",
        "function": "Self-referential processing",
        "impact": "DMN degradation"
    }
]

df = pd.DataFrame(data)

# === 節點: Region ===
df_region = df[["region_id", "region_name"]].drop_duplicates().rename(columns={
    "region_id": "id:ID(Region)",
    "region_name": "name"
})
df_region["label:LABEL"] = "Region"
df_region.to_csv(f"{output_dir}/nodes_region.csv", index=False)

# === 節點: Network ===
df_network = df[["network_name"]].drop_duplicates().rename(columns={
    "network_name": "name"
})
df_network["id:ID(Network)"] = df_network["name"]
df_network["label:LABEL"] = "Network"
df_network = df_network[["id:ID(Network)", "name", "label:LABEL"]]
df_network.to_csv(f"{output_dir}/nodes_network.csv", index=False)

# === 節點: Function ===
df_func = df[["function"]].drop_duplicates()
df_func["id:ID(Function)"] = df_func["function"]
df_func["name"] = df_func["function"]
df_func["label:LABEL"] = "Function"
df_func = df_func[["id:ID(Function)", "name", "label:LABEL"]]
df_func.to_csv(f"{output_dir}/nodes_function.csv", index=False)

# === 節點: Impact ===
df_impact = df[["impact"]].drop_duplicates()
df_impact["id:ID(Impact)"] = df_impact["impact"]
df_impact["name"] = df_impact["impact"]
df_impact["label:LABEL"] = "Impact"
df_impact = df_impact[["id:ID(Impact)", "name", "label:LABEL"]]
df_impact.to_csv(f"{output_dir}/nodes_impact.csv", index=False)

# === 邊: Region -[:BELONGS_TO]-> Network ===
df_r2n = df.rename(columns={
    "region_id": ":START_ID(Region)",
    "network_name": ":END_ID(Network)"
})
df_r2n[":TYPE"] = "BELONGS_TO"
df_r2n = df_r2n[[":START_ID(Region)", ":END_ID(Network)", ":TYPE"]]
df_r2n.to_csv(f"{output_dir}/edges_region_to_network.csv", index=False)

# === 邊: Region -[:HAS_FUNCTION]-> Function ===
df_r2f = df.rename(columns={
    "region_id": ":START_ID(Region)",
    "function": ":END_ID(Function)"
})
df_r2f[":TYPE"] = "HAS_FUNCTION"
df_r2f = df_r2f[[":START_ID(Region)", ":END_ID(Function)", ":TYPE"]]
df_r2f.to_csv(f"{output_dir}/edges_region_to_function.csv", index=False)

# === 邊: Region -[:AFFECTS_PATIENT]-> Impact ===
df_r2i = df.rename(columns={
    "region_id": ":START_ID(Region)",
    "impact": ":END_ID(Impact)"
})
df_r2i[":TYPE"] = "AFFECTS_PATIENT"
df_r2i = df_r2i[[":START_ID(Region)", ":END_ID(Impact)", ":TYPE"]]
df_r2i.to_csv(f"{output_dir}/edges_region_to_impact.csv", index=False)

print("✅ 所有 CSV 檔案已產生在：", output_dir)
