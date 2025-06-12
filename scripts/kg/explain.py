import pandas as pd
from collections import defaultdict

# === 預先讀入 Yeo 分佈 mapping ===
yeo_mapping_df = pd.read_csv("output/kg/aal3_to_yeo7_mapping_distribution.csv")

# 建立 lookup: (region, region_id) -> list of (network, yeo_id, percentage)
region_to_yeo = defaultdict(list)
for _, row in yeo_mapping_df.iterrows():
    key = (row["region"], int(str(row["region_id"]).split()[0]))
    region_to_yeo[key].append((row["network"], row["yeo_id"], row["percentage"]))


# === 主函式：處理單一 subject 的腦區 list ===
def explain_subject_regions(subject_id: str, region_list: list[tuple[str, int]], top_k: int = 1):
    print(f"🧠 Subject: {subject_id}\n{'='*60}")
    for region_name, region_id in region_list:
        key = (region_name, region_id)
        if key not in region_to_yeo:
            print(f"⚠️ 找不到對應 Yeo 資訊：{region_name} ({region_id})")
            continue

        mappings = sorted(region_to_yeo[key], key=lambda x: x[2], reverse=True)
        top = mappings[:top_k]
        rest = mappings[top_k:]

        # 主導網路描述
        main_net, main_id, main_pct = top[0]
        line = (
            f"「{region_name}」（ID: {region_id}）主要參與 Yeo7 的「{main_net}」網路 "
            f"(ID: {main_id})，佔比約 {main_pct * 100:.1f}%。"
        )

        # 額外貢獻網路（>5%）
        extras = [
            f"{net}（{perc * 100:.1f}%）"
            for net, _, perc in rest if perc > 0.05
        ]
        if extras:
            line += " 其他亦涵蓋：" + "、".join(extras) + "。"

        print("- " + line)
    print()


# === 範例：sub-14 的兩個 activation 腦區 ===
example_subject_id = "sub-14"
example_regions = [
    ("Angular_L", 69),
    ("Frontal_Sup_Medial_R", 20),
    ("Temporal_Pole_Sup_R", 83),
    ("Frontal_Mid_Orb_L", 9),
]

explain_subject_regions(example_subject_id, example_regions)
