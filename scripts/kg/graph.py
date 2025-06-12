import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# === 讀取 CSV ===
csv_path = "output/kg/aal3_to_yeo7_mapping_distribution.csv"
df = pd.read_csv(csv_path)

# === 建立有向圖 ===
G = nx.DiGraph()

for _, row in df.iterrows():
    region = row["region"]
    region_id = int(str(row["region_id"]).split()[0])
    network = row["network"]
    yeo_id = row["yeo_id"]
    percentage = row["percentage"]

    region_node = f"{region} ({region_id})"
    network_node = f"{network} ({yeo_id})"

    G.add_node(region_node, type="region")
    G.add_node(network_node, type="network")
    G.add_edge(region_node, network_node, weight=percentage)

# === [1] 儲存完整圖 ===
os.makedirs("output/kg", exist_ok=True)
nx.write_graphml(G, "output/kg/aal3_yeo7_full.graphml")
print("✅ Full graph saved to: output/kg/aal3_yeo7_full.graphml")

# === [2] 美化後可視化：整體圖（circular layout） ===
plt.figure(figsize=(18, 12))
pos = nx.circular_layout(G)

node_colors = [
    "#90CAF9" if G.nodes[n]["type"] == "region" else "#A5D6A7"
    for n in G.nodes
]
node_border_color = "#333333"
edge_widths = [G[u][v]["weight"] * 10 for u, v in G.edges]
edge_alphas = [0.7 if G[u][v]["weight"] >= 0.05 else 0.2 for u, v in G.edges]

nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    edgecolors=node_border_color,
    node_size=1600,
    linewidths=1.5,
    alpha=0.9
)
nx.draw_networkx_labels(
    G, pos,
    font_size=8,
    font_color="#1a1a1a"
)
nx.draw_networkx_edges(
    G, pos,
    width=edge_widths,
    alpha=edge_alphas,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=12,
    edge_color="#555555"
)

# 顯示主要邊的百分比權重（僅 > 0.1）
edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges if G[u][v]['weight'] > 0.1}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title("AAL Region ➝ Yeo Network Mapping", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()