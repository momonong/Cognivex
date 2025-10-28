import torch
import torch.nn as nn
from typing import Type, List, Dict, Any

def save_activation_hook(layer_name: str, store_dict: dict):
    """
    Create a forward hook to store activations of a specific layer.
    """

    def hook(module, input, output):
        # Store the detached tensor on the CPU
        store_dict[layer_name] = output.detach().cpu()

    return hook


def attach_hooks(
    model: torch.nn.Module, target_paths: List[str], activation_dict: dict
):
    """
    Attach forward hooks using fully qualified layer paths (e.g., 'backbone.conv1').
    """
    # target_paths is now a list of strings, e.g., ["backbone.stage4.1.bn3"]
    for name, module in model.named_modules():
        if name in target_paths:
            # print(f"Hook attached: {name} ({module.__class__.__name__})")
            module.register_forward_hook(save_activation_hook(name, activation_dict))


def resolve_target_layers(selector_output: List[Dict[str, Any]]) -> List[str]:
    """
    Extract model_path directly from selector_output as the target layers for hook attachment.
    """
    # Creates a unique list of 'model_path' strings
    return list(
        set(sel["model_path"] for sel in selector_output if "model_path" in sel)
    )


def prepare_model_with_hooks(model: nn.Module, selector_output: List[Dict[str, Any]]) -> nn.Module:
    """
    Orchestrates the hook attachment process.
    Attaches an 'activations' dictionary to the model instance.
    """
    activations = {}
    resolved_layers = resolve_target_layers(selector_output)
    attach_hooks(model, resolved_layers, activations)
    
    # Attach the dictionary to the model itself for easy access
    model.activations = activations
    return model


if __name__ == "__main__":
    # --- UPDATED Main test block ---
    # This block now tests the hook functions using the *correct*
    # PaperModel and the new model loader.

    # 1. Import the new central loader
    from model.loader import get_model_and_input_shape

    # 2. Get the model and its input shape
    MODEL, input_shape = get_model_and_input_shape()
    
    if MODEL is None:
        print("Error: Model loading failed. Exiting hook manager test.")
        exit()

    # 3. Define a MOCK selector_output (simulating the LLM's choice)
    # This is what the selector script (step 2) would return
    # for the PaperModel, based on the 'shufflenet_focused' strategy.
    mock_selector_output = [
        {
            "layer_name": "ShuffleUnit-8",
            "layer_type": "BatchNorm2d",
            "model_path": "backbone.stage4.1.bn3", # This is the target
            "reason": "This is the final layer of the backbone..."
        }
    ]
    
    print(f"--- Testing Hook Manager ---")
    print(f"Model: {MODEL.__class__.__name__}")
    print(f"Target Hook Layer: {mock_selector_output[0]['model_path']}")

    # 4. Call the function we are testing
    model_with_hooks = prepare_model_with_hooks(MODEL, mock_selector_output)

    # 5. Create a dummy input tensor to trigger the forward pass
    # We add a batch dimension (B=1) to the input_shape
    # input_shape is (10, 1, 128, 128)
    dummy_input = torch.randn(1, *input_shape) 

    # 6. Run a forward pass to trigger the hooks
    print("\nRunning dummy forward pass...")
    try:
        _ = model_with_hooks(dummy_input)
    except Exception as e:
        print(f"Error during forward pass: {e}")
        exit()

    # 7. Verify that the activations were captured
    if hasattr(model_with_hooks, 'activations') and "backbone.stage4.1.bn3" in model_with_hooks.activations:
        print("\n--- [SUCCESS] ---")
        captured_shape = model_with_hooks.activations["backbone.stage4.1.bn3"].shape
        print(f"Hook successfully captured activation for 'backbone.stage4.1.bn3'")
        
        # Note: The backbone runs on (B*N_slices, C, H, W)
        # So B=1, N_slices=10 -> B_flat = 10
        print(f"Captured Tensor Shape: {captured_shape}")
        
    else:
        print("\n--- [FAILURE] ---")
        print("Activations were not captured.")
        if hasattr(model_with_hooks, 'activations'):
             print(f"Activation dict exists but keys are: {model_with_hooks.activations.keys()}")