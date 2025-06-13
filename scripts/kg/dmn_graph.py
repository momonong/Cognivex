import networkx as nx
import matplotlib.pyplot as plt

# === 載入圖 ===
G_full = nx.read_graphml("output/kg/aal3_yeo7_full.graphml")

# === 找出 DMN node ===
dmn_node = next(
    (n for n, d in G_full.nodes(data=True) if d.get("type") == "network" and "Default" in n), None
)
if not dmn_node:
    raise ValueError("找不到 Default Mode Network 節點")

# === 過濾權重高於 0.05 的邊 ===
edges_to_keep = [
    (u, v) for u, v in G_full.in_edges(dmn_node)
    if float(G_full[u][v]["weight"]) >= 0.05
]
nodes_to_keep = {u for u, _ in edges_to_keep}
nodes_to_keep.add(dmn_node)
G_sub = G_full.subgraph(nodes_to_keep).copy()

# === 視覺化 ===
plt.figure(figsize=(12, 10))
pos = nx.shell_layout(G_sub, nlist=[[dmn_node], list(nodes_to_keep - {dmn_node})])

node_colors = ["#A5D6A7" if n == dmn_node else "#90CAF9" for n in G_sub.nodes]
edge_widths = [float(G_sub[u][v]["weight"]) * 10 for u, v in G_sub.edges]
edge_labels = {
    (u, v): f"{float(G_sub[u][v]['weight']):.2f}"
    for u, v in G_sub.edges
}

nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=1600, edgecolors="#333")
nx.draw_networkx_labels(G_sub, pos, font_size=9)
nx.draw_networkx_edges(G_sub, pos, width=edge_widths, alpha=0.8, arrows=True, edge_color="#555")
nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels, font_size=7)

plt.title("AAL → DMN (Filtered ≥ 5%)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig("figures/kg/dmn_graph.png")