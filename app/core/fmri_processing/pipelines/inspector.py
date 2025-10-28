from torch.nn.modules.module import Module


import torch.nn as nn
from model.loader import get_model_and_input_shape 

def inspect_torch_model(model: nn.Module) -> list[dict]:
    """
    Returns layer information from a PyTorch model using named_modules.
    
    This simpler version does NOT use torchsummary and will not provide
    output_shape or param_count. 
    
    It is more robust and will not fail on complex models 
    with multiple outputs.
    
    :param model: PyTorch model instance
    :return: list of layers with their path and type
    """
    
    combined_info = []
    
    for name, module in model.named_modules():
        if name == "":  # Skip the top-level module
            continue

        # We are only interested in "leaf" modules (those that perform operations)
        # and not containers like nn.Sequential or nn.ModuleList.
        is_leaf = not list[Module](module.children())
        is_container = isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict))
        
        if is_leaf and not is_container:
            # This is a layer we can attach a hook to.
            combined_info.append({
                "model_path": name,                 # e.g. "backbone.conv1"
                "layer_type": module.__class__.__name__, # e.g. "Conv2d"
            })
            
    return combined_info


if __name__ == "__main__":
    
    # Call the central loader to get the model
    MODEL, input_shape = get_model_and_input_shape() 

    if MODEL is None:
        print("Error: Model loading failed. Exiting inspector.")
        exit() # Exit if import failed

    print(f"--- Analyzing {MODEL.__class__.__name__} (Simple Inspector) ---")
    print(f"--- Reference Input Shape (N_slices, C, H, W): {input_shape} ---")
    
    # Get the list of inspectable layers
    model_inspection = inspect_torch_model(MODEL) 
    
    if not model_inspection:
        print("\nERROR: Still could not find any layers. Please check model definition.")
    else:
        print("\n--- Model Layers (Available for XAI hooks) ---")
        for layer in model_inspection:
            print(layer)