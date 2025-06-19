import pandas as pd

# === [1] 載入 AAL3 ➝ Yeo7 對應分布 ===
df = pd.read_csv("output/kg/aal3_to_yeo7_mapping_distribution.csv")

# === [2] 建立 AD 關聯腦區清單（從文獻抽取）===
ad_related_regions = {
    "Hippocampus": "Memory encoding and spatial navigation",
    "Amygdala": "Emotion processing and memory modulation",
    "ParaHippocampal": "Contextual memory and scene recognition",
    "Temporal_Sup": "Auditory processing and language comprehension",
    "Temporal_Mid": "Semantic memory and emotion",
    "Cingulate_Mid": "Attention and cognitive control",
    "Cingulate_Post": "Memory retrieval and default mode network",
    "Thalamus": "Relay of sensory and cognitive information",
    "Frontal_Sup": "Executive function and attention control",
    "Occipital_Mid": "Visual processing and memory"
}

# 額外補充部分功能描述（可以再擴充）
network_to_function = {
    "Default": "Internally directed cognition and memory (DMN)",
    "Limbic": "Emotion and memory",
    "Somatomotor": "Sensorimotor processing",
    "Dorsal Attention": "Top-down attention control",
    "Ventral Attention": "Stimulus-driven attention",
    "Frontoparietal": "Executive control and working memory",
    "Visual": "Visual perception and spatial processing",
    "None": "Not assigned to any functional network"
}

# === [3] 產生知識圖譜 dataframe（腦區 ➝ 網路 ➝ 功能 ➝ AD 關聯）===
rows = []

for _, row in df.iterrows():
    region = row["region"]
    network = row["network"]
    yeo_id = row["yeo_id"]
    percentage = row["percentage"]

    # 比對腦區是否與 AD 有關（模糊比對）
    is_ad_related = any(key in region for key in ad_related_regions)
    related_func = next((desc for key, desc in ad_related_regions.items() if key in region), "N/A")

    network_func = network_to_function.get(network, "N/A")

    rows.append({
        "Region": region,
        "Yeo_Network": network,
        "Yeo_ID": yeo_id,
        "Region➝Network_Percentage": percentage,
        "Region_Function": related_func,
        "Network_Function": network_func,
        "AD_Associated": "Yes" if is_ad_related else "No"
    })

kg_df = pd.DataFrame(rows)
kg_df.to_csv("output/kg/knowledge_graph_brain_regions.csv", index=False)
print("✅ CSV 已輸出：output/kg/knowledge_graph_brain_regions.csv")
