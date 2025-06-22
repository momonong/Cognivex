import torch
from typing import Type


def save_activation_hook(layer_name: str, store_dict: dict):
    """
    Create a forward hook to store activations of a specific layer.
    """

    def hook(module, input, output):
        store_dict[layer_name] = output.detach().cpu()

    return hook


def attach_hooks(
    model: torch.nn.Module, selector_output: list[str], activation_dict: dict
):
    """
    Attach forward hooks using fully qualified layer paths (e.g., 'capsnet.conv3').
    """
    target_paths = selector_output
    for name, module in model.named_modules():
        if name in target_paths:
            print(f"Hook attached: {name} ({module.__class__.__name__})")
            module.register_forward_hook(save_activation_hook(name, activation_dict))


def resolve_target_layers(selector_output: list[dict]) -> list[str]:
    """
    Extract model_path directly from selector_output as the target layers for hook attachment.
    """
    return list(
        set(sel["model_path"] for sel in selector_output if "model_path" in sel)
    )


def prepare_model_with_hooks(model, selector_output: list[dict]) -> torch.nn.Module:
    activations = {}
    resolved_layers = resolve_target_layers(selector_output)
    attach_hooks(model, resolved_layers, activations)
    model.activations = activations
    return model


if __name__ == "__main__":
    # EXAMPLE USAGE
    from scripts.capsnet.model import CapsNetRNN

    selector_output = [
        {
            "layer_name": "Conv3d-4",
            "layer_type": "Conv3d",
            "model_path": "capsnet.conv3",
            "reason": "Relatively high-level convolutional layer, preserving spatial information",
        },
        {
            "layer_name": "Squash-8",
            "layer_type": "CapsuleLayer3D",
            "model_path": "capsnet.caps2",
            "reason": "Captures higher-level capsule features before final classification",
        },
    ]

    # 1. Initialize model + hook
    model = prepare_model_with_hooks(CapsNetRNN, selector_output)
