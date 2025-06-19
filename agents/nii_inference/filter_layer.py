import os
import torch

def get_activation_stats(act_path: str):
    act = torch.load(act_path)
    act = act[0] if act.dim() == 5 else act
    stats = {
        "nonzero_ratio": (act > 1e-4).float().mean().item(),
        "mean_activation": act.mean().item(),
        "std_activation": act.std().item(),
        "max_activation": act.max().item(),
    }
    return stats

def reselect_layers_by_activation_stats(selected_layers, activation_dir, save_name_prefix):
    layer_stat_list = []

    for layer in selected_layers:
        model_path = layer["model_path"]
        act_path = os.path.join(activation_dir, f"{save_name_prefix}_{model_path.replace('.', '_')}.pt")
        if not os.path.exists(act_path):
            print(f"[Skip] Activation not found for {model_path}")
            continue

        stats = get_activation_stats(act_path)
        layer_stat_list.append({
            **layer,
            **stats
        })

    filtered_layers = []
    for layer in layer_stat_list:
        if (layer["nonzero_ratio"] > 0.1) and (layer["mean_activation"] > 0.001):
            layer["reason"] = "Passed dynamic threshold filtering based on activation stats"
            print(f"[Filter] Keep {layer['model_path']} | ratio={layer['nonzero_ratio']:.4f}, mean={layer['mean_activation']:.6f}")
            filtered_layers.append(layer)
        else:
            print(f"[Filter] Drop {layer['model_path']} | ratio={layer['nonzero_ratio']:.4f}, mean={layer['mean_activation']:.6f}")

    return filtered_layers
