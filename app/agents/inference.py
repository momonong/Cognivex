# app/agents/1_run_inference.py
from app.graph.state import AgentState
# Import tools from our new core library
from app.core.fmri_processing.pipeline_steps import (
    inspect_model_structure,
    validate_layers,
    prepare_model_for_inference,
    real_data_inference,
)
# Import constants from the core library
from app.core.fmri_processing.pipeline_steps import (
    MODEL,
    DEVICE,
    INPUT_SHAPE,
    OUTPUT_DIR,
    WINDOW,
    STRIDE,
)

def run_inference_and_classification(state: AgentState) -> dict:
    """
    Node 1: Performs model prep, layer validation, and core inference.
    """
    print("\n--- Node: 1. Running Inference & Classification ---")
    subject_id = state['subject_id']
    model_path = state.get('model_path') # Assume model_path is passed in state
    nii_path = state['fmri_scan_path']
    save_name = f"{subject_id}"

    try:
        # Step 1 & 2: Inspect and Validate Layers
        all_layers_info, _ = inspect_model_structure(
            model=MODEL, input_shape=INPUT_SHAPE, device=DEVICE
        )
        validated_layers = validate_layers(
            selected_layers=all_layers_info, all_layers_info=all_layers_info
        )
        if not validated_layers:
            raise ValueError("No valid layers found after validation")
        
        validated_layer_names = [layer["model_path"] for layer in validated_layers]

        # Step 3: Prepare Model
        prepared_model = prepare_model_for_inference(
            model=MODEL,
            selected_layers=validated_layers,
            model_path=model_path,
            device=DEVICE,
        )
        # Step 4: Real Data Inference
        prediction_result = real_data_inference(
            model=prepared_model,
            nii_path=nii_path,
            save_dir=OUTPUT_DIR,
            save_name=save_name,
            selected_layer_names=validated_layer_names,
            window=WINDOW,
            stride=STRIDE,
            device=DEVICE,
        )

        trace = f"Node 1: Inference complete. Prediction: {prediction_result}"
        
        return {
            "classification_result": prediction_result,
            "validated_layers": validated_layers,
            "trace_log": state.get("trace_log", []) + [trace]
        }

    except Exception as e:
        error_message = f"Node 1 (Inference) Error: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}