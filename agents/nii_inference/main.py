import torch
import os
import json
from agents.nii_inference.inspect_model import inspect_torch_model
from agents.nii_inference.choose_layer import select_visualization_layers
from agents.nii_inference.attach_hook import prepare_model_with_hooks
from agents.nii_inference.inference import run_inference
from agents.nii_inference.filter_layer import filter_layers_by_gemini
from agents.nii_inference.act_to_nii import activation_to_nifti
from agents.nii_inference.resample import resample_activation_to_atlas
from agents.nii_inference.brain_map import analyze_brain_activation
from agents.nii_inference.visualize import visualize_activation_map
from scripts.capsnet.model import CapsNetRNN


def main():
    subject_id = "sub-14"
    model_path = "model/capsnet/best_capsnet_rnn.pth"
    nii_path = f"data/raw/AD/{subject_id}/dswausub-098_S_6601_task-rest_bold.nii.gz"
    save_name = "module_test"
    output_dir = f"output/{save_name}"
    output_prefix = os.path.join(output_dir, save_name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Inspect model structure
    print("\nStep 1: Inspect model structure")
    MODEL = CapsNetRNN()
    input_shape = (1, 1, 91, 91, 109)
    layers = inspect_torch_model(MODEL, input_shape)
    response = select_visualization_layers(layers)
    selected_layers = json.loads(response.text)
    print("Selected layers:", selected_layers)

    # Extract model paths only
    selected_layer_names = [item["model_path"] for item in selected_layers]

    # Step 2: Prepare model with hook + load weights
    print("\nStep 2: Attach hook and load weights")
    model = prepare_model_with_hooks(CapsNetRNN, selected_layers, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Step 3: Inference + save activation
    print("\nStep 3: Real data inference")
    run_inference(
        model=model,
        nii_path=nii_path,
        save_dir=output_dir,
        save_name=save_name,
        selected_layer_names=selected_layer_names,  
        window=5,
        stride=3,
        device=device,
    )

    # Step 4: Dynamic filtering based on activation stats via Gemini
    print("\nStep 4: Dynamic filtering based on activation stats (via Gemini)")
    selected_layers = filter_layers_by_gemini(
        selected_layers=selected_layers,
        activation_dir=output_dir,
        save_name_prefix=save_name,
        delete_rejected=True,  # 可選：是否刪除 activation
    )
    if not selected_layers:
        raise ValueError("No valid layers selected after Gemini filtering.")

    selected_layer_names = [layer["model_path"] for layer in selected_layers]
    print(f"[Summary] Final selected layers: {selected_layer_names}")


    # Step 5~7: For each selected layer, continue post-processing
    for layer_name in selected_layer_names:
        safe_layer_name = layer_name.replace(".", "_")
        act_path = f"{output_prefix}_{safe_layer_name}.pt"
        nii_output = f"{output_prefix}_{safe_layer_name}.nii.gz"
        vis_dir = f"figures/module_test/{save_name}_{safe_layer_name}"
        os.makedirs(vis_dir, exist_ok=True)

        print(f"\nStep 5: Convert activation to NIfTI for layer: {layer_name}")
        assert os.path.exists(act_path), f"Missing activation file: {act_path}"
        activation_to_nifti(
            activation_path=act_path,
            reference_nii_path=nii_path,
            output_path=nii_output,
            norm_type="l2",
            threshold_percentile=99.0,
        )

        print(f"\nStep 6: Resample to atlas for layer: {layer_name}")
        resampled_path = resample_activation_to_atlas(
            act_path=nii_output,
            atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
            output_dir=os.path.join(output_dir, "resampled", safe_layer_name),
        )

        print(f"\nStep 7: Analyze brain activation for layer: {layer_name}")
        df_result = analyze_brain_activation(
            activation_path=resampled_path,
            atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
            label_path="data/aal3/AAL3v1_1mm.nii.txt",
        )
        print(df_result)

        print(f"\nStep 8: Visualize activation for layer: {layer_name}")
        visualize_activation_map(
            activation_path=resampled_path,
            output_path=os.path.join(vis_dir, "activation_map_mosaic.png"),
            threshold=0.1,
            title=f"Activation Map ({subject_id} {layer_name})",
        )


if __name__ == "__main__":
    main()
