# (In app/core/xai/hook_manager.py)
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any


def save_activation_hook(layer_name: str, store_dict: dict):
    def hook(module, input, output):
        store_dict[layer_name] = output.detach().cpu()

    return hook


def attach_hooks(  # Renamed from attach_hooks to clarify it's for forward activations
    model: torch.nn.Module, target_paths: List[str], activation_dict: dict
):
    for name, module in model.named_modules():
        if name in target_paths:
            module.register_forward_hook(save_activation_hook(name, activation_dict))


def resolve_target_layers(selector_output: List[Dict[str, Any]]) -> List[str]:
    return list(
        set(sel["model_path"] for sel in selector_output if "model_path" in sel)
    )


_gradient_handles = []  # Global list to keep track of gradient hook handles


def save_gradient_hook(layer_name: str, store_dict: dict):
    """
    Creates a backward hook to store gradients of a specific layer's output.
    Note: PyTorch backward hooks operate on module *outputs*.
    """

    def hook(module, grad_input, grad_output):
        # grad_output is a tuple containing gradients w.r.t. module outputs
        # We usually want the first element if the module has a single output tensor
        if isinstance(grad_output, tuple) and len(grad_output) > 0:
            store_dict[layer_name] = grad_output[0].detach().cpu()
        # Handle cases where grad_output might not be a tuple (less common)
        elif torch.is_tensor(grad_output):
            store_dict[layer_name] = grad_output.detach().cpu()

    return hook


def attach_gradient_hooks(
    model: torch.nn.Module, target_paths: List[str], gradient_dict: dict
):
    """
    Attaches backward hooks to capture gradients for specified layers.
    Stores hook handles globally to allow removal later.
    """
    global _gradient_handles
    # Clear any previous handles before attaching new ones
    remove_hooks(_gradient_handles)
    _gradient_handles = []  # Reset the list

    for name, module in model.named_modules():
        if name in target_paths:
            handle = module.register_full_backward_hook(
                save_gradient_hook(name, gradient_dict)
            )
            _gradient_handles.append(handle)  # Store the handle


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]):
    """
    Removes hooks using their handles to prevent memory leaks.
    """
    for handle in handles:
        handle.remove()
    handles.clear()  # Clear the list after removing


# --- Modified prepare_model_with_hooks ---
# Now returns handles for activation hooks as well
def prepare_model_with_hooks(
    model: nn.Module, selector_output: List[Dict[str, Any]]
) -> Tuple[nn.Module, Dict[str, Any], List[torch.utils.hooks.RemovableHandle]]:
    """
    Orchestrates the FORWARD hook attachment process.
    Attaches an 'activations' dictionary to the model instance.
    Returns the model, the activation dict, and handles for removal.
    """
    activations = {}
    activation_handles = []  # Store activation hook handles
    resolved_layers = resolve_target_layers(selector_output)

    # Modified attach_hooks logic (inline or call modified function)
    for name, module in model.named_modules():
        if name in resolved_layers:
            handle = module.register_forward_hook(
                save_activation_hook(name, activations)
            )
            activation_handles.append(handle)  # Store handle

    # Attach the dictionary to the model itself for easy access
    model.activations = activations  # Keep this for convenience

    return model, activations, activation_handles


if __name__ == "__main__":

    # 引入梯度鉤子相關函數（確保它們在 hook_manager.py 裡）
    from app.core.mri_processing.pipelines.attach_hook import (
        attach_gradient_hooks,
        _gradient_handles,
        remove_hooks,
    )  # 假設已經在當前模組中

    # 1. Import the new central loader
    # 確保路徑正確：從 app.core.mri_processing.mri_model_loader 導入
    from app.core.mri_processing.mri_model_loader import get_model_and_input_shape

    # 2. Get the model and its input shape
    MODEL, input_shape = get_model_and_input_shape()

    if MODEL is None:
        print("Error: Model loading failed. Exiting hook manager test.")
        exit()

    # 3. Define a MOCK selector_output (simulating the LLM's choice)
    TARGET_PATH = "backbone.stage4.1.gconv2"  # <--- 修正: 使用單一變數
    mock_selector_output = [
        {
            "layer_name": "Final_Conv_Block",
            "layer_type": "Conv2d",
            "model_path": TARGET_PATH,  # <--- 修正: 使用正確的目標層
            "reason": "Final Conv2d layer, captures most abstract spatial features.",
        }
    ]

    print(f"--- Testing Hook Manager (Forward & Backward) ---")
    print(f"Model: {MODEL.__class__.__name__}")
    print(f"Target Hook Layer: {TARGET_PATH}")

    # --- Setup ---
    # 4. Call the function we are testing (attaches FORWARD hooks)
    model_with_hooks, activations_dict, forward_handles = prepare_model_with_hooks(
        MODEL, mock_selector_output
    )

    # 4b. Attach BACKWARD hooks manually for testing
    gradients_dict = {}
    attach_gradient_hooks(model_with_hooks, [TARGET_PATH], gradients_dict)

    # 5. Create a dummy input tensor
    dummy_input = torch.randn(1, *input_shape).to(next(MODEL.parameters()).device)
    # 確保輸入需要梯度
    dummy_input.requires_grad_(True)

    # 6. Run a forward pass to trigger the hooks
    print("\nRunning forward pass...")
    try:
        # 允許梯度追蹤
        outputs = model_with_hooks(dummy_input)
    except Exception as e:
        print(f"Error during forward pass: {e}")
        remove_hooks(forward_handles)
        remove_hooks(_gradient_handles)
        exit()

    # 6b. Run a backward pass to trigger gradient hooks
    # 假設我們計算 class 1 (AD) 的梯度
    class_score = outputs[0][0, 1]
    print("Running backward pass...")
    class_score.backward(retain_graph=True)

    # --- Verification ---
    print("\n--- Verification ---")

    # 7. Verify Forward Hooks (Activation)
    if TARGET_PATH in activations_dict:  # <--- 修正: 驗證正確的鍵
        print(f"✓ Activation hook SUCCESS: Captured key '{TARGET_PATH}'")
        print(f"  Captured Tensor Shape (Act): {activations_dict[TARGET_PATH].shape}")
        activation_success = True
    else:
        print(f"✗ Activation hook FAILED. Keys: {activations_dict.keys()}")
        activation_success = False

    # 7b. Verify Backward Hooks (Gradient)
    if TARGET_PATH in gradients_dict:  # <--- 修正: 驗證正確的鍵
        print(f"✓ Gradient hook SUCCESS: Captured key '{TARGET_PATH}'")
        print(f"  Captured Tensor Shape (Grad): {gradients_dict[TARGET_PATH].shape}")
        gradient_success = True
    else:
        print(f"✗ Gradient hook FAILED. Keys: {gradients_dict.keys()}")
        gradient_success = False

    # --- Cleanup ---
    # 清除鉤子以避免內存洩露
    remove_hooks(forward_handles)
    remove_hooks(_gradient_handles)

    if activation_success and gradient_success:
        print("\n--- [SUCCESS] All hooks verified and cleaned up. ---")
    else:
        print("\n--- [FAILURE] One or more hooks failed. ---")
