def explain_activation(start_node, G, max_depth=3):
    visited = set()
    result = []
    queue = [(start_node, 0)]

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        # 如果節點不是圖中的節點就跳過
        if current not in G:
            continue

        for neighbor in G.successors(current):
            relation = G.edges[current, neighbor]["relation"]
            result.append((current, neighbor, relation))
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))
                visited.add(neighbor)

    return result


def generate_explanation(region, G):
    chain = explain_activation(region, G)
    segments = {
        rel: [] for rel in ["part_of", "supports", "manifests_as", "associated_with"]
    }

    for src, tgt, rel in chain:
        segments[rel].append((src, tgt))

    text = f"模型顯著活化了 {region}，"

    if segments["part_of"]:
        _, net = segments["part_of"][0]
        text += f"該區域屬於 {net} 功能網絡，"

    if segments["supports"]:
        _, func = segments["supports"][0]
        text += f"與 {func} 相關。"

    if segments["manifests_as"]:
        symp = segments["manifests_as"][0][1]
        text += f"該功能受損時常見表現為 {symp}，"

    if segments["associated_with"]:
        dis = segments["associated_with"][0][1]
        text += f"這可能與 {dis} 有關。"

    return text
