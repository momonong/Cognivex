import numpy as np
from nilearn import datasets
import nibabel as nib
from skimage.transform import resize
from collections import defaultdict
import matplotlib.pyplot as plt
from utils.semantic_utils import explain_activation, explain_naturally
from utils.semantic_utils import generate_explanation
import networkx as nx
import os

def run_activation_to_top_regions(act_path, z_slice=50, top_k=5):
    act = np.load(act_path)
    act = np.squeeze(act)
    act_map = act.mean(axis=0)

    atlas = datasets.fetch_atlas_aal(version='SPM12')
    aal_img = nib.load(atlas['maps'])
    aal_data = aal_img.get_fdata()
    aal_labels = atlas['labels']
    aal_indices = atlas['indices']
    label_dict = {int(k): v for k, v in zip(aal_indices, aal_labels)}

    aal_slice = aal_data[:, :, z_slice]
    resized_aal = resize(aal_slice, act_map.shape, order=0, preserve_range=True, anti_aliasing=False)
    resized_aal = np.rint(resized_aal).astype(int)

    region_activation = defaultdict(list)
    for i in range(act_map.shape[0]):
        for j in range(act_map.shape[1]):
            label_id = resized_aal[i, j]
            if label_id > 0:
                region_activation[label_id].append(act_map[i, j])

    region_mean = {k: np.mean(v) for k, v in region_activation.items()}
    top = sorted(region_mean.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(label_dict.get(k, f"ID:{k}"), v) for k, v in top]

def build_semantic_graph():
    G = nx.DiGraph()

    # === Layer 1: Anatomical Regions ===
    region_to_network = {
        "Cuneus_L": "Visual-Dorsal Attention Network",
        "Occipital_Sup_R": "Visual-Dorsal Attention Network",
        "Occipital_Sup_L": "Visual-Dorsal Attention Network",
        "Precuneus_L": "Posterior Cingulate-Precuneus",
        "Precuneus_R": "Posterior Cingulate-Precuneus",
        "Angular_L": "Language Network",
        "Angular_R": "Language Network",
        "Hippocampus_L": "Hippocampus-Amygdala",
        "Amygdala_R": "Hippocampus-Amygdala",
        "SupraMarginal_L": "Language Network",
        "Postcentral_L": "Frontoparietal Network",
        "Frontal_Mid_R": "Frontoparietal Network",
        "Frontal_Mid_L": "Frontoparietal Network",
        "Occipital_Mid_L": "Visual-Dorsal Attention Network",
        "Occipital_Mid_R": "Visual-Dorsal Attention Network",
        "Parietal_Inf_L": "Language Network",
        "Parietal_Inf_R": "Language Network",
        "Temporal_Mid_L": "Language Network",
        "Temporal_Mid_R": "Language Network",
        "Cingulum_Post_L": "Posterior Cingulate-Precuneus",
        "Cingulum_Post_R": "Posterior Cingulate-Precuneus",
    }

    for region, net in region_to_network.items():
        G.add_node(region, type="Region")
        G.add_node(net, type="Region")
        G.add_edge(region, net, relation="part_of")

    # === Layer 2+: Network → Function → Symptom → Disease ===
    nodes = [
        ("Memory & Emotion Coupling", {"type": "Function"}),
        ("Self-referential Processing", {"type": "Function"}),
        ("Visuospatial Processing", {"type": "Function"}),
        ("Working Memory & Executive Control", {"type": "Function"}),
        ("Stimulus Filtering & Attention Switching", {"type": "Function"}),
        ("Language Comprehension & Expression", {"type": "Function"}),
        ("Motor Planning", {"type": "Function"}),

        ("Recent Memory Loss & Apathy", {"type": "Symptom"}),
        ("Forgetfulness & Subjective Complaints", {"type": "Symptom"}),
        ("Spatial Disorientation & Recognition Deficit", {"type": "Symptom"}),
        ("Poor Planning & Problem Solving", {"type": "Symptom"}),
        ("Irritability & Attention Instability", {"type": "Symptom"}),
        ("Progressive Aphasia", {"type": "Symptom"}),
        ("Fine Motor Impairment", {"type": "Symptom"}),

        ("MCI", {"type": "Disease"}),
        ("AD", {"type": "Disease"}),
        ("Posterior Cortical Atrophy", {"type": "Disease"}),
        ("Language Variant AD", {"type": "Disease"}),
    ]

    edges = [
        ("Hippocampus-Amygdala", "Memory & Emotion Coupling", {"relation": "supports"}),
        ("Memory & Emotion Coupling", "Recent Memory Loss & Apathy", {"relation": "manifests_as"}),
        ("Recent Memory Loss & Apathy", "AD", {"relation": "associated_with"}),

        ("Posterior Cingulate-Precuneus", "Self-referential Processing", {"relation": "supports"}),
        ("Self-referential Processing", "Forgetfulness & Subjective Complaints", {"relation": "manifests_as"}),
        ("Forgetfulness & Subjective Complaints", "MCI", {"relation": "associated_with"}),

        ("Visual-Dorsal Attention Network", "Visuospatial Processing", {"relation": "supports"}),
        ("Visuospatial Processing", "Spatial Disorientation & Recognition Deficit", {"relation": "manifests_as"}),
        ("Spatial Disorientation & Recognition Deficit", "Posterior Cortical Atrophy", {"relation": "associated_with"}),

        ("Frontoparietal Network", "Working Memory & Executive Control", {"relation": "supports"}),
        ("Working Memory & Executive Control", "Poor Planning & Problem Solving", {"relation": "manifests_as"}),
        ("Poor Planning & Problem Solving", "MCI", {"relation": "associated_with"}),
        ("Poor Planning & Problem Solving", "AD", {"relation": "associated_with"}),

        ("Salience Network", "Stimulus Filtering & Attention Switching", {"relation": "supports"}),
        ("Stimulus Filtering & Attention Switching", "Irritability & Attention Instability", {"relation": "manifests_as"}),
        ("Irritability & Attention Instability", "AD", {"relation": "associated_with"}),

        ("Language Network", "Language Comprehension & Expression", {"relation": "supports"}),
        ("Language Comprehension & Expression", "Progressive Aphasia", {"relation": "manifests_as"}),
        ("Progressive Aphasia", "Language Variant AD", {"relation": "associated_with"}),

        ("Precentral_L", "Frontoparietal Network", {"relation": "part_of"}),
        ("Frontoparietal Network", "Motor Planning", {"relation": "supports"}),
        ("Motor Planning", "Fine Motor Impairment", {"relation": "manifests_as"}),
        ("Fine Motor Impairment", "MCI", {"relation": "associated_with"}),

        ("Frontoparietal Network", "Attention Control", {"relation": "supports"}),
        ("Attention Control", "Distractibility & Decision Delay", {"relation": "manifests_as"}),
        ("Distractibility & Decision Delay", "AD", {"relation": "associated_with"}),

    ]

    G.add_nodes_from(nodes)
    G.add_edges_from([(u, v, d) for u, v, d in edges])
    return G

def visualize_and_save_graph(G, save_path="graphql/semantic_graph.png"):
    type_colors = {
        "Region": "skyblue",
        "Function": "lightgreen",
        "Symptom": "orange",
        "Disease": "tomato"
    }
    node_colors = [type_colors.get(G.nodes[n].get("type", "Region"), "gray") for n in G.nodes]

    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2500,
            font_size=8, edge_color='gray')
    edge_labels = nx.get_edge_attributes(G, "relation")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Alzheimer's Disease Semantic Knowledge Graph", fontsize=14)
    plt.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 知識圖譜儲存於 {save_path}")

if __name__ == '__main__':
    act_path = "output/activation_conv2_AD_AD_027_S_6648_task-rest_bold.nii_z001_t003.npy"
    top_named_regions = run_activation_to_top_regions(act_path, z_slice=50)
    G = build_semantic_graph()

    for region, _ in top_named_regions:
        if region not in G:
            candidates = list(G.successors(region)) if region in G.nodes else []
            if candidates:
                region = candidates[0]
            else:
                print(f"\u26a0\ufe0f 無法對應語意圖調中節點：{region}")
                continue

        print(f"\n\U0001f9e0 模型解釋：")
        print(generate_explanation(region, G))

    # 加入呼叫
    visualize_and_save_graph(G)
