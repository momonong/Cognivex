import torch
import os
import json

from app.core.fmri_processing.pipelines.inspect_model import (
    inspect_torch_model,
)
from app.core.fmri_processing.pipelines.choose_layer import (
    select_visualization_layers,
)
from app.core.fmri_processing.pipelines.attach_hook import (
    prepare_model_with_hooks as attach_hooks_to_model,
)
from app.core.fmri_processing.pipelines.inference import (
    run_nii_inference,
)
from app.core.fmri_processing.pipelines.filter_layer import (
    filter_layers_by_llm,
)
from app.core.fmri_processing.pipelines.validate_layer import (
    validate_layers_by_llm,
)
from app.core.fmri_processing.pipelines.act_to_nii import (
    activation_to_nifti,
)
from app.core.fmri_processing.pipelines.resample import (
    resample_activation_to_atlas,
)
from app.core.fmri_processing.pipelines.brain_map import (
    analyze_brain_activation,
)
from app.core.fmri_processing.pipelines.visualize import (
    visualize_activation_map,
)

from scripts.capsnet.model import CapsNetRNN
from scripts.macadnnet.model import MCADNNet

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
MODEL = CapsNetRNN().to(DEVICE)
MODEL_TYPE = "3d" # '3d' or '2d'
# MODEL = MCADNNet().to(DEVICE)
# MODEL_TYPE = "2d"

INPUT_SHAPE = (1, 1, 91, 91, 109)
# INPUT_SHAPE = (1, 64, 64)

# --- Model-specific parameters ---
if MODEL_TYPE == "3d":
    WINDOW = 5 # Example for 3D sliding window
    STRIDE = 3
else: # 2D model
    WINDOW = 1 # Process slice by slice
    STRIDE = 1

# --- General parameters ---
STRIDE = 3
OUTPUT_DIR = "output/langraph"
SAVE_NAME = "langraph_test"
VIS_DIR_PREFIX = f"figures/langraph_test"
NORM_TYPE = "l2"
ACT_THRESHOLD_PERCENTILE = 99.0
ATLAS_PATH = "data/aal3/AAL3v1_1mm.nii.gz"
LABEL_PATH = "data/aal3/AAL3v1_1mm.nii.txt"
VIS_THRESHOLD_PERCENTILE = 0.1


def inspect_model_structure(model, input_shape, device):
    """
    Inspect model structure and return selected layers and their model paths

    Args:
        model: The model to inspect
        input_shape: The input shape of the model
        device: The device to use

    Returns:
        selected_layers: The selected layers
        selected_layer_names: The model paths of the selected layers
    """
    # print("\nStep 1: Inspect model structure")
    layers = inspect_torch_model(model, input_shape, device)
    response = select_visualization_layers(layers)
    selected_layers = json.loads(response)
    # print("Selected layers:", selected_layers)
    # Extract model paths only
    selected_layer_names = [item["model_path"] for item in selected_layers]

    return selected_layers, selected_layer_names


def prepare_model_for_inference(model, selected_layers, model_path, device):
    """
    Prepare model with hooks and load weights

    Args:
        model: The neural network model to attach hooks to
        selected_layers: List of selected layer information for hook attachment
        model_path: Path to the model weights file (.pth)
        device: The device to load the model on (cuda/mps/cpu)

    Returns:
        model: The model with hooks attached and weights loaded, ready for inference
    """
    model = attach_hooks_to_model(model, selected_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    return model


def real_data_inference(
    model, nii_path, save_dir, save_name, selected_layer_names, window, stride, device
):
    """
    Run NIfTI inference and save activation data

    Args:
        model: The prepared model with hooks for inference
        nii_path: Path to the input NIfTI file
        save_dir: Directory to save inference results and activations
        save_name: Base name for saved files
        selected_layer_names: List of layer names to extract activations from
        window: Window size for sliding window inference
        stride: Stride size for sliding window
        device: Device to run inference on

    Returns:
        prediction_result: The classification prediction result
    """
    prediction_result = run_nii_inference(
        model,
        nii_path,
        save_dir,
        save_name,
        selected_layer_names,
        window,
        stride,
        device,
    )

    return prediction_result


def dynamic_filtering(
    results,
    selected_layers,
    activation_dir,
    save_name_prefix,
    delete_rejected,
):
    """
    Dynamic filtering based on activation stats via LLM

    Args:
        results: Results dictionary to update with filtered layers
        selected_layers: List of initially selected layers
        activation_dir: Directory containing activation files
        save_name_prefix: Prefix for saved activation files
        delete_rejected: Boolean flag to delete rejected activation files

    Returns:
        keep_entries: List of layers that passed LLM filtering
        selected_layer_names: List of model paths for kept layers
        output_prefix: Output path prefix for further processing
    """
    keep_entries = filter_layers_by_llm(
        selected_layers, activation_dir, save_name_prefix, delete_rejected
    )
    results["final_layers"] = keep_entries
    selected_layer_names = [layer["model_path"] for layer in keep_entries]
    # print(f"[Summary] Final selected layers: {selected_layer_names}")
    output_prefix = os.path.join(activation_dir, save_name_prefix)

    return keep_entries, selected_layer_names, output_prefix


def validate_layers(selected_layers: list[dict], all_layers_info: list[dict]) -> list[dict]:
    """
    Validates that the layers selected by the LLM actually exist in the model.
    This uses the validate_layers_by_llm function for comprehensive LLM-based validation.
    This is a critical safety check to run before inference.

    Args:
        selected_layers (list[dict]): List of layers selected by the LLM with their metadata
        all_layers_info (list[dict]): Ground-truth list of all layers from inspect_torch_model

    Returns:
        list[dict]: List of validated layers that exist in the model with complete metadata
    """
    # print("\n--- Tool: Validating Selected Layers ---")
    
    # First do basic existence validation
    valid_layer_paths = {layer["model_path"] for layer in all_layers_info}
    basic_validated_layers = []

    for layer in selected_layers:
        model_path = layer.get("model_path")
        if model_path in valid_layer_paths:
            basic_validated_layers.append(layer)
            # print(f"✔  OK: {model_path}")
        # else:
        #     # print(f"✘ DROPPED (Invalid): {model_path} - Layer does not exist.")
    
    # Use LLM validation but preserve original metadata format
    if basic_validated_layers:
        # print("\n--- Running LLM-based Layer Validation ---")
        llm_validated_results = validate_layers_by_llm(basic_validated_layers)
        
        # Map LLM results back to original layer format with complete metadata
        validated_model_paths = {item["model_path"] for item in llm_validated_results}
        final_validated_layers = []
        
        for original_layer in basic_validated_layers:
            if original_layer["model_path"] in validated_model_paths:
                # Find the corresponding LLM result for updated reason
                llm_result = next(
                    (item for item in llm_validated_results if item["model_path"] == original_layer["model_path"]), 
                    None
                )
                # Preserve original format but update reason from LLM
                updated_layer = original_layer.copy()
                if llm_result:
                    updated_layer["reason"] = llm_result["reason"]
                final_validated_layers.append(updated_layer)
                
        # print(f"\n[Validation Summary] {len(final_validated_layers)}/{len(selected_layers)} layers passed validation")
        return final_validated_layers
    else:
        # print("\n[Warning] No valid layers found after basic validation")
        return []


def setup_layer_path(
    layer_name, output_prefix, reference_nii_path, norm_type, threshold_percentile
):
    """
    Setup file paths and convert activation to NIfTI format for a specific layer

    Args:
        layer_name: Name of the neural network layer
        output_prefix: Base output path prefix for files

    Returns:
        act_path: Path to the activation .pt file
        nii_output: Path to the converted NIfTI file
        vis_dir: Directory for visualization outputs
    """
    safe_layer_name = layer_name.replace(".", "_")
    act_path = f"{output_prefix}_{safe_layer_name}.pt"
    nii_output = f"{output_prefix}_{safe_layer_name}.nii.gz"
    vis_dir = f"{VIS_DIR_PREFIX}/{SAVE_NAME}_{safe_layer_name}"
    os.makedirs(vis_dir, exist_ok=True)
    activation_to_nifti(
        activation_path=act_path,
        reference_nii_path=reference_nii_path,
        output_path=nii_output,
        norm_type=norm_type,
        threshold_percentile=threshold_percentile,
    )
    return act_path, nii_output, vis_dir


def resample_to_atlas(act_path, atlas_path, output_dir):
    """
    Resample activation data to match atlas space

    Args:
        act_path: Path to the activation NIfTI file
        atlas_path: Path to the atlas NIfTI file for resampling
        output_dir: Directory to save resampled results

    Returns:
        resampled_path: Path to the resampled activation file
    """
    resampled_path = resample_activation_to_atlas(
        act_path=act_path,
        atlas_path=atlas_path,
        output_dir=output_dir,
    )
    return resampled_path


def analyze_brain_activation_data(activation_path, atlas_path, label_path):
    """
    Analyze brain activation patterns using atlas regions

    Args:
        activation_path: Path to the activation NIfTI file
        atlas_path: Path to the brain atlas NIfTI file
        label_path: Path to the atlas label text file

    Returns:
        df_result: DataFrame containing activation analysis results by brain region
    """
    df_result = analyze_brain_activation(
        activation_path=activation_path,
        atlas_path=atlas_path,
        label_path=label_path,
    )
    return df_result


def visualize_activation_map_data(activation_path, output_path, threshold, title):
    """
    Generate visualization of brain activation map

    Args:
        activation_path: Path to the activation NIfTI file
        output_path: Path to save the visualization image
        threshold: Threshold percentile for activation visualization
        title: Title for the visualization plot

    Returns:
        vis_output_path: Path to the generated visualization image
    """
    vis_output_path = visualize_activation_map(
        activation_path=activation_path,
        output_path=output_path,
        threshold=threshold,
        title=title,
    )
    return vis_output_path


def save_results(results, vis_output_path, df_result):
    """
    Save and update results with visualization and activation analysis data

    Args:
        results: Results dictionary to update
        vis_output_path: Path to the visualization output file
        df_result: DataFrame containing activation analysis results

    Returns:
        results: Updated results dictionary with visualization and activation data
    """
    results["visualization_results"] = vis_output_path
    results["activation_results"] = df_result.to_dict(orient="records")
    # print(df_result)
    # print("RESULTS:", results)
    return results
