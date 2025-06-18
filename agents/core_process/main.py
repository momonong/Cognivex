import torch
import os
import json
from agents.core_process.inspect_model import inspect_torch_model
from agents.core_process.choose_layer import select_visualization_layers
from agents.core_process.attach_hook import prepare_model_with_hooks
from agents.core_process.inference import run_inference
from agents.core_process.act_to_nii import activation_to_nifti
from agents.core_process.resample import resample_activation_to_atlas
from agents.core_process.brain_map import analyze_brain_activation
from agents.core_process.visualize import visualize_activation_map
from scripts.capsnet.model import CapsNetRNN


def main():
    subject_id = "sub-14"
    layer_name = "conv3"
    save_name = "module_test"
    model_path = "model/capsnet/best_capsnet_rnn.pth"
    nii_path = f"data/raw/AD/{subject_id}/dswausub-098_S_6601_task-rest_bold.nii.gz"
    output_dir = f"output/capsnet"
    output_prefix = os.path.join(output_dir, save_name)

    print("\nStep 1: Inspect model structure")
    MODEL = CapsNetRNN()
    input_shape = (1, 1, 91, 91, 109)
    layers = inspect_torch_model(MODEL, input_shape)
    response = select_visualization_layers(layers)
    selected_layers = json.loads(response.text)
    print("Selected layers:", selected_layers)

    print("\nStep 2: Attach hook and infer")
    model = prepare_model_with_hooks(CapsNetRNN, selected_layers)

    print("\nStep 3: Real data inference")
    run_inference(
        model=model,
        model_path=model_path,
        nii_path=nii_path,
        save_dir=output_dir,
        save_name=save_name,
        window=5,
        stride=3,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    )

    print("\nStep 4: Convert activation to NIfTI")
    activation_path = f"{output_prefix}_{layer_name}.pt"
    assert os.path.exists(activation_path), f"Missing activation file: {activation_path}"
    activation_to_nifti(
        activation_path=activation_path,
        reference_nii_path=nii_path,
        output_path=f"{output_prefix}.nii.gz",
        norm_type="l2",
        threshold_percentile=99.0,
    )

    print("\nStep 5: Resample to atlas space")
    resampled_path = resample_activation_to_atlas(
        act_path=f"{output_prefix}.nii.gz",
        atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
        output_dir=os.path.join(output_dir, "resampled", save_name),
    )

    print("\nStep 6: Analyze brain activation")
    df_result = analyze_brain_activation(
        activation_path=resampled_path,
        atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
        label_path="data/aal3/AAL3v1_1mm.nii.txt",
    )
    print(df_result)

    print("\nStep 7: Visualize activation")
    vis_dir = f"figures/capsnet/{save_name}"
    os.makedirs(vis_dir, exist_ok=True)
    visualize_activation_map(
        activation_path=resampled_path,
        output_path=os.path.join(vis_dir, "activation_map_mosaic.png"),
        threshold=0.1,
        title=f"Activation Map ({subject_id} {layer_name})",
    )


if __name__ == "__main__":
    main()
