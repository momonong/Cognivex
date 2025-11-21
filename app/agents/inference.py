# app/agents/1_run_inference.py
from app.graph.state import AgentState

# NEW: Import the generic pipeline system
from app.core.mri_processing.generic_pipeline_steps import (
    run_inference_and_classification as generic_run_inference,
    GenericInferencePipeline
)
from app.core.mri_processing.model_config import (
    get_config_by_name,
    ModelConfig,
    ModelType
)

# OLD: Keep original imports for comparison/fallback
# from app.core.fmri_processing.pipeline_steps import (
#     inspect_model_structure,
#     validate_layers, 
#     prepare_model_for_inference,
#     real_data_inference,
#     MODEL, DEVICE, INPUT_SHAPE, OUTPUT_DIR, WINDOW, STRIDE,
# )

def run_inference_and_classification(state: AgentState) -> dict:
    """
    Node 1: Performs model prep, layer validation, and core inference using the new generic system.
    
    Args:
        state: AgentState containing the workflow state (including model_name if specified)
    """
    # Get model name from state, default to shufflenet
    model_name = state.get('model_name', 'shufflenet')
    model_path = state.get('model_path')
    nii_path = state.get('fmri_scan_path')
    subject_id = state.get('subject_id', 'unknown_subject')
    
    print(f"\n--- Node: 1. Running Generic Inference & Classification ({model_name}) ---")
    
    if not nii_path:
        return {"error_log": state.get("error_log", []) + ["Missing fMRI scan path in state"]}
    
    try:
        # Create and run the full generic pipeline
        config = get_config_by_name(model_name)
        pipeline = GenericInferencePipeline(
            model_config=config,
            model_weights_path=model_path,
            output_dir=f"output/generic_pipeline/{subject_id}"
        )
        
        # Run the full pipeline with post-processing to get all XAI results
        results = pipeline.run_full_pipeline(
            nii_path=nii_path,
            save_name=subject_id,
            include_post_processing=True,
            target_class_index=1  # AD class
        )
        
        if "error" in results:
            return {"error_log": state.get("error_log", []) + [results["error"]]}
        
        trace = f"Node 1: Generic inference complete. Prediction: {results.get('prediction_result')}"
        
        # Return comprehensive results for the rest of the pipeline
        return {
            "classification_result": results.get("prediction_result"),
            "validated_layers": results.get("selected_layers", []),
            "activated_regions": results.get("activated_regions", []),
            "visualization_paths": results.get("visualization_paths", {}),
            "final_heatmap_paths": results.get("final_heatmap_paths", {}),
            "activation_gradient_files": results.get("activation_gradient_files", {}),
            "trace_log": state.get("trace_log", []) + [trace]
        }
        
    except Exception as e:
        error_message = f"Node 1 (Generic Inference) Error: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}

# Alternative: More flexible version that allows custom configurations
def run_inference_with_custom_config(state: AgentState, model_config: ModelConfig) -> dict:
    """
    Node 1 Alternative: Run inference with a custom model configuration.
    
    This version gives you full control over the model parameters.
    """
    print(f"\n--- Node: 1. Running Custom Inference ({model_config.model_type.value}) ---")
    
    subject_id = state['subject_id']
    model_path = state.get('model_path')
    nii_path = state['fmri_scan_path']
    save_name = f"{subject_id}"
    
    try:
        # Create pipeline with custom config
        pipeline = GenericInferencePipeline(
            model_config=model_config,
            model_path=model_path
        )
        
        # Run the full pipeline
        results = pipeline.run_full_pipeline(nii_path, save_name)
        
        if "error" in results:
            return {"error_log": state.get("error_log", []) + [results["error"]]}
        
        trace = f"Node 1: Custom inference complete. Prediction: {results['prediction_result']}"
        
        return {
            "classification_result": results["prediction_result"],
            "validated_layers": results["validated_layers"],
            "trace_log": state.get("trace_log", []) + [trace]
        }
        
    except Exception as e:
        error_message = f"Node 1 (Custom Inference) Error: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}

# Legacy function for backward compatibility
def run_inference_and_classification_legacy(state: AgentState) -> dict:
    """
    Legacy version using hardcoded CapsNet - kept for backward compatibility.
    
    This is the old implementation - consider migrating to the generic version.
    """
    print("\n--- Node: 1. Running Legacy Inference & Classification ---")
    print("WARNING: Using legacy hardcoded implementation. Consider upgrading to generic version.")
    
    # Use the generic version with CapsNet as default
    return run_inference_and_classification(state, model_name="capsnet")
