def generate_cypher_path(region, G):
    """
    根據圖譜從某個 region 開始輸出一條 Cypher MATCH 語句（最長 3 跳）
    例如：
    MATCH (r:Region {id: "Hippocampus_L"})-[:PART_OF]->(:Region)-[:SUPPORTS]->(:Function)-[:MANIFESTS_AS]->(:Symptom)-[:ASSOCIATED_WITH]->(:Disease)
    RETURN *
    """
    rel_map = {
        "part_of": "PART_OF",
        "supports": "SUPPORTS",
        "manifests_as": "MANIFESTS_AS",
        "associated_with": "ASSOCIATED_WITH"
    }

    path = []
    visited = set()
    queue = [(region, 0)]

    while queue:
        current, depth = queue.pop(0)
        if current not in G or depth >= 4:
            continue
        for neighbor in G.successors(current):
            rel = G.edges[current, neighbor]["relation"]
            if rel in rel_map and neighbor not in visited:
                path.append((current, rel_map[rel], neighbor))
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    if not path:
        return f"// ⚠️ 無法從 {region} 找到語意路徑"

    start_var = "n0"
    cypher = f'MATCH ({start_var}:Region {{id: "{region}"}})'
    current_var = start_var
    node_count = 1
    for _, rel, tgt in path:
        next_var = f"n{node_count}"
        cypher += f"-[:{rel}]->({next_var})"
        current_var = next_var
        node_count += 1

    cypher += "\nRETURN *"
    return cypher
