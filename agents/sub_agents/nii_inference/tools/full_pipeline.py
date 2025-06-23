import torch
import os
import json

from agents.sub_agents.nii_inference.tools.pipelines.inspect_model import (
    inspect_torch_model,
)
from agents.sub_agents.nii_inference.tools.pipelines.choose_layer import (
    select_visualization_layers,
)
from agents.sub_agents.nii_inference.tools.pipelines.attach_hook import (
    prepare_model_with_hooks,
)
from agents.sub_agents.nii_inference.tools.pipelines.inference import (
    run_nii_inference,
)
from agents.sub_agents.nii_inference.tools.pipelines.filter_layer import (
    filter_layers_by_gemini,
)
from agents.sub_agents.nii_inference.tools.pipelines.act_to_nii import (
    activation_to_nifti,
)
from agents.sub_agents.nii_inference.tools.pipelines.resample import (
    resample_activation_to_atlas,
)
from agents.sub_agents.nii_inference.tools.pipelines.brain_map import (
    analyze_brain_activation,
)
from agents.sub_agents.nii_inference.tools.pipelines.visualize import (
    visualize_activation_map,
)

from scripts.capsnet.model import CapsNetRNN

SUBJECT_ID = "sub-14"
MODEL_PATH = "model/capsnet/best_capsnet_rnn.pth"
NII_PATH = "data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
MODEL = CapsNetRNN().to(DEVICE)
INPUT_SHAPE = (1, 1, 91, 91, 109)
WINDOW = 5
STRIDE = 3
OUTPUT_DIR = "output/agent_test"
SAVE_NAME = "agent_test"
VIS_DIR_PREFIX = f"figures/agent_test"
NORM_TYPE = "l2"
ACT_THRESHOLD_PERCENTILE = 99.0
ATLAS_PATH = "data/aal3/AAL3v1_1mm.nii.gz"
LABEL_PATH = "data/aal3/AAL3v1_1mm.nii.txt"
VIS_FIG_NAME = "activation_map_mosaic.png"
VIS_THRESHOLD_PERCENTILE = 0.1


def pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "classification": None,
        "final_layers": [],
        "activation_results": [],
        "visualization_results": None,
    }

    # Step 1: Inspect model structure
    print("\nStep 1: Inspect model structure")
    layers = inspect_torch_model(MODEL, INPUT_SHAPE, DEVICE)
    response = select_visualization_layers(layers)
    selected_layers = json.loads(response)
    print("Selected layers:", selected_layers)

    # Extract model paths only
    selected_layer_names = [item["model_path"] for item in selected_layers]

    # Step 2: Prepare model with hook + load weights
    print("\nStep 2: Attach hook and load weights")
    model = prepare_model_with_hooks(MODEL, selected_layers)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    # Step 3: Inference + save activation
    print("\nStep 3: Real data inference")
    prediction_result = run_nii_inference(
        model=model,
        nii_path=NII_PATH,
        save_dir=OUTPUT_DIR,
        save_name=SAVE_NAME,
        selected_layer_names=selected_layer_names,
        window=WINDOW,
        stride=STRIDE,
        device=DEVICE,
    )
    results["classification"] = prediction_result

    # Step 4: Dynamic filtering based on activation stats via Gemini
    print("\nStep 4: Dynamic filtering based on activation stats (via Gemini)")
    keep_entries = filter_layers_by_gemini(
        selected_layers=selected_layers,
        activation_dir=OUTPUT_DIR,
        save_name_prefix=SAVE_NAME,
        delete_rejected=True,  # 可選：是否刪除 activation
    )
    if not keep_entries:
        raise ValueError("No valid layers selected after Gemini filtering.")

    results["final_layers"] = keep_entries
    selected_layer_names = [layer["model_path"] for layer in keep_entries]
    print(f"[Summary] Final selected layers: {selected_layer_names}")
    output_prefix = os.path.join(OUTPUT_DIR, SAVE_NAME)

    # Step 5~8: For each selected layer, continue post-processing
    for layer_name in selected_layer_names:
        safe_layer_name = layer_name.replace(".", "_")
        act_path = f"{output_prefix}_{safe_layer_name}.pt"
        nii_output = f"{output_prefix}_{safe_layer_name}.nii.gz"
        vis_dir = f"{VIS_DIR_PREFIX}/{SAVE_NAME}_{safe_layer_name}"
        os.makedirs(vis_dir, exist_ok=True)

        print(f"\nStep 5: Convert activation to NIfTI for layer: {layer_name}")
        assert os.path.exists(act_path), f"Missing activation file: {act_path}"
        activation_to_nifti(
            activation_path=act_path,
            reference_nii_path=NII_PATH,
            output_path=nii_output,
            norm_type=NORM_TYPE,
            threshold_percentile=ACT_THRESHOLD_PERCENTILE,
        )

        print(f"\nStep 6: Resample to atlas for layer: {layer_name}")
        resampled_path = resample_activation_to_atlas(
            act_path=nii_output,
            atlas_path=ATLAS_PATH,
            output_dir=os.path.join(OUTPUT_DIR, "resampled", safe_layer_name),
        )

        print(f"\nStep 7: Analyze brain activation for layer: {layer_name}")
        df_result = analyze_brain_activation(
            activation_path=resampled_path,
            atlas_path=ATLAS_PATH,
            label_path=LABEL_PATH,
        )
        results["activation_results"] = df_result.to_dict(orient="records")
        print(df_result)

        print(f"\nStep 8: Visualize activation for layer: {layer_name}")
        vis_output_path = visualize_activation_map(
            activation_path=resampled_path,
            output_path=os.path.join(vis_dir, VIS_FIG_NAME),
            threshold=VIS_THRESHOLD_PERCENTILE,
            title=f"Activation Map ({SUBJECT_ID} {layer_name})",
        )
        results["visualization_results"] = vis_output_path

        print("RESULTS:", results)
        return results


if __name__ == "__main__":
    pipeline()
