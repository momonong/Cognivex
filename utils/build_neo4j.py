import csv
import os

def export_graph_to_neo4j_csv(G, node_path="graphql/nodes.csv", edge_path="graphql/edges.csv"):
    os.makedirs(os.path.dirname(node_path), exist_ok=True)

    # 匯出節點
    with open(node_path, mode='w', newline='', encoding='utf-8') as f_node:
        writer = csv.writer(f_node)
        writer.writerow(['id', 'label', 'type'])  # Neo4j 建議欄位
        for node, data in G.nodes(data=True):
            writer.writerow([node, data.get('type', 'Unknown'), data.get('type', 'Unknown')])

    # 匯出邊
    with open(edge_path, mode='w', newline='', encoding='utf-8') as f_edge:
        writer = csv.writer(f_edge)
        writer.writerow(['source', 'target', 'relation'])
        for u, v, data in G.edges(data=True):
            writer.writerow([u, v, data.get('relation', 'related_to')])

    print(f"✅ 已輸出 Neo4j 匯入檔：{node_path}, {edge_path}")
