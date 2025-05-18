from utils.semantic_utils import explain_activation


def region_to_path(state):
    region = state["region"]
    G = state["G"]

    paths = explain_activation(region, G)
    state["paths"] = paths
    return state


def path_to_template(state):
    region = state["region"]
    paths = state["paths"]

    segments = {
        "part_of": [],
        "supports": [],
        "manifests_as": [],
        "associated_with": [],
    }

    for src, tgt, rel in paths:
        if rel in segments:
            segments[rel].append((src, tgt))

    text = f"模型顯著活化了 {region}，"
    if segments["part_of"]:
        _, net = segments["part_of"][0]
        text += f"該區域屬於 {net} 功能網絡，"
    if segments["supports"]:
        _, func = segments["supports"][0]
        text += f"與 {func} 相關。"
    if segments["manifests_as"]:
        _, symp = segments["manifests_as"][0]
        text += f"該功能受損時常見表現為 {symp}，"
    if segments["associated_with"]:
        _, dis = segments["associated_with"][0]
        text += f"這可能與 {dis} 有關。"

    state["explanation"] = text
    return state
