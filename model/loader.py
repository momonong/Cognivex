import torch
import torch.nn as nn

def get_model_and_input_shape():
    """
    Imports the model and its specific input configuration.
    
    This function acts as the central "model registry" for the application.
    All other parts of the system (API, XAI, processing)
    should call this function to get model information.
    
    :return: (model_instance, input_shape_tuple) or (None, None) on failure
    """
    try:
        # Import the model AND the preprocessing constants
        # This path is based on your latest script
        from model.shufflenet.model import (
            PaperModel, 
            NUM_SLICES_PER_SUBJECT, 
            SLICE_IMG_SIZE
        )
    except ImportError as e:
        print(f"Error: Could not import PaperModel or constants from model.shufflenet.model")
        print(f"Details: {e}")
        return None, None

    MODEL = PaperModel()
    
    # Construct the input shape from the imported constants
    # (N_slices, C, H, W)
    # The '1' for channels (C) is standard for grayscale 2D CNNs
    input_shape = (NUM_SLICES_PER_SUBJECT, 1, SLICE_IMG_SIZE, SLICE_IMG_SIZE)
    
    return MODEL, input_shape


def load_model_for_inference(weights_path: str = "model/shufflenet/fold_3_best_model.pth"):
    """
    Helper function to get the model, input shape, and load weights.
    Sets model to eval() mode.
    
    Your main API should call this function.
    
    :param weights_path: Optional path to the .pth model weights file.
    :return: (model_instance, input_shape_tuple)
    """
    model, input_shape = get_model_and_input_shape()
    if model:
        if weights_path:
            try:
                # Load weights
                model.load_state_dict(torch.load(weights_path))
                print(f"Successfully loaded weights from {weights_path}")
            except Exception as e:
                print(f"Warning: Could not load weights from {weights_path}. Error: {e}")
                # Decide if this should be a fatal error depending on use case
                
        model.eval() # Set model to evaluation mode
        
    return model, input_shape