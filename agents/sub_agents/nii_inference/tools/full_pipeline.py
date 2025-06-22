from typing import TypedDict, List
import os

class NiiInferenceInput(TypedDict):
    subject_id: str
    nii_path: str
    model_path: str
    output_name: str

class ActivationResult(TypedDict):
    layer: str
    summary: str
    visualization_path: str

class NiiInferenceOutput(TypedDict):
    classification: str
    final_layers: List[str]
    activation_results: List[ActivationResult]

def run_full_inference_pipeline(
    subject_id: str,
    nii_path: str,
    model_path: str = "model/capsnet/best_capsnet_rnn.pth",
    output_name: str = "module_test"
) -> NiiInferenceOutput:
    from scripts.capsnet.model import CapsNetRNN
    import torch
    import json
    from agents.sub_agents.nii_inference.tools.inspect_model import inspect_torch_model
    from agents.sub_agents.nii_inference.tools.choose_layer import select_visualization_layers
    from agents.sub_agents.nii_inference.tools.attach_hook import prepare_model_with_hooks
    from agents.sub_agents.nii_inference.tools.inference import run_inference
    from agents.sub_agents.nii_inference.tools.filter_layer import filter_layers_by_gemini
    from agents.sub_agents.nii_inference.tools.act_to_nii import activation_to_nifti
    from agents.sub_agents.nii_inference.tools.resample import resample_activation_to_atlas
    from agents.sub_agents.nii_inference.tools.brain_map import analyze_brain_activation
    from agents.sub_agents.nii_inference.tools.visualize import visualize_activation_map

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    output_dir = f"output/{output_name}"
    output_prefix = os.path.join(output_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)

    MODEL = CapsNetRNN()
    input_shape = (1, 1, 91, 91, 109)
    layers = inspect_torch_model(MODEL, input_shape)
    selected_layers = json.loads(select_visualization_layers(layers))
    selected_layer_names = [item["model_path"] for item in selected_layers]

    model = prepare_model_with_hooks(CapsNetRNN, selected_layers, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    run_inference(
        model=model,
        nii_path=nii_path,
        save_dir=output_dir,
        save_name=output_name,
        selected_layer_names=selected_layer_names,
        window=5,
        stride=3,
        device=device,
    )

    selected_layers = filter_layers_by_gemini(
        selected_layers=selected_layers,
        activation_dir=output_dir,
        save_name_prefix=output_name,
        delete_rejected=True,
    )

    if not selected_layers:
        raise ValueError("No valid layers selected after Gemini filtering.")

    selected_layer_names = [layer["model_path"] for layer in selected_layers]

    activation_results = []
    for layer_name in selected_layer_names:
        safe_layer_name = layer_name.replace(".", "_")
        act_path = f"{output_prefix}_{safe_layer_name}.pt"
        nii_output = f"{output_prefix}_{safe_layer_name}.nii.gz"
        vis_dir = f"figures/{output_name}_{safe_layer_name}"
        os.makedirs(vis_dir, exist_ok=True)

        activation_to_nifti(
            activation_path=act_path,
            reference_nii_path=nii_path,
            output_path=nii_output,
            norm_type="l2",
            threshold_percentile=99.0,
        )

        resampled_path = resample_activation_to_atlas(
            act_path=nii_output,
            atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
            output_dir=os.path.join(output_dir, "resampled", safe_layer_name),
        )

        df_result = analyze_brain_activation(
            activation_path=resampled_path,
            atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
            label_path="data/aal3/AAL3v1_1mm.nii.txt",
        )

        vis_path = os.path.join(vis_dir, "activation_map_mosaic.png")
        visualize_activation_map(
            activation_path=resampled_path,
            output_path=vis_path,
            threshold=0.1,
            title=f"Activation Map ({subject_id} {layer_name})",
        )

        activation_results.append({
            "layer": layer_name,
            "summary": df_result.to_string(index=False),
            "visualization_path": vis_path,
        })

    return {
        "classification": "AD",  # optional: 你可以根據 logits 真實分類
        "final_layers": selected_layer_names,
        "activation_results": activation_results,
    }
