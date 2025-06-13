import pandas as pd
from collections import defaultdict

# 讀入 CSV
csv_path = "output/kg/aal3_to_yeo7_mapping_distribution.csv"
df = pd.read_csv(csv_path)

# 分組處理每個 AAL 區域
region_groups = defaultdict(list)

for _, row in df.iterrows():
    region = row["region"]
    region_id = row["region_id"]
    network = row["network"]
    yeo_id = row["yeo_id"]
    percentage = row["percentage"]

    region_key = (region, region_id)
    region_groups[region_key].append((network, yeo_id, percentage))

# 產出文字說明
# output_lines = []

for (region, region_id), network_list in region_groups.items():
    # 根據百分比排序，選出 top 1
    sorted_nets = sorted(network_list, key=lambda x: x[2], reverse=True)
    main_net = sorted_nets[0]
    others = sorted_nets[1:]

    main_text = (
        f"「{region}」區域（AAL ID: {region_id}）有 {main_net[2]*100:.1f}% 的體素對應於 "
        f"Yeo7 的「{main_net[0]}」網路（ID: {main_net[1]}），為其主要參與網路。"
    )

    others_text = ""
    for net, yeo_id, perc in others:
        if perc >= 0.05:
            others_text += f" 其中 {perc*100:.1f}% 體素對應「{net}」（ID: {yeo_id}）；"

    if others_text:
        final_text = f"{main_text} 此外，{others_text.rstrip('；')}。"
    else:
        final_text = main_text

    # output_lines.append(final_text)
    print(final_text)

# # 儲存為文字檔
# with open("output/kg/region_to_yeo7_explanation.txt", "w") as f:
#     for line in output_lines:
#         f.write(line + "\n")

# print("✅ 語意說明已產出：output/kg/region_to_yeo7_explanation.txt")
