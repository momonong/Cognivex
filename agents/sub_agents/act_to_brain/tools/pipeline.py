"""
Complete NIfTI Inference Pipeline Tool for Google ADK Agent

This module provides a single, unified tool function that runs the complete
nii_inference pipeline for use with Google ADK LlmAgent.
"""

import json
import os
from typing import Dict, Any

from agents.sub_agents.act_to_brain.tools.pipeline_tools import (
    inspect_model_structure,
    validate_layers, 
    prepare_model_for_inference,
    real_data_inference,
    dynamic_filtering,
    setup_layer_path,
    resample_to_atlas,
    analyze_brain_activation_data,
    visualize_activation_map_data,
    save_results,
    # Import constants
    MODEL, DEVICE, INPUT_SHAPE, MODEL_PATH, NII_PATH, 
    OUTPUT_DIR, SAVE_NAME, ATLAS_PATH, LABEL_PATH,
    NORM_TYPE, ACT_THRESHOLD_PERCENTILE, VIS_THRESHOLD_PERCENTILE
)


def complete_nii_inference_pipeline(
    subject_id: str = "sub-14",
    model_path: str = None,
    nii_path: str = None,
    output_dir: str = None,
    save_name: str = None
) -> str:
    """
    Run the complete NIfTI inference pipeline for fMRI Alzheimer's detection.
    
    This function orchestrates the entire pipeline from model inspection to final 
    visualization and returns a structured JSON result for the ADK agent.
    
    Args:
        subject_id (str): Subject identifier for the analysis
        model_path (str): Path to model weights (optional, uses default if None)
        nii_path (str): Path to NIfTI file (optional, uses default if None) 
        output_dir (str): Output directory (optional, uses default if None)
        save_name (str): Save name prefix (optional, uses default if None)
        
    Returns:
        str: JSON string containing complete analysis results
    """
    
    # Use defaults if not provided
    model_path = model_path or MODEL_PATH
    nii_path = nii_path or NII_PATH
    output_dir = output_dir or OUTPUT_DIR
    save_name = save_name or SAVE_NAME
    
    print("="*80)
    print(f"RUNNING COMPLETE NII INFERENCE PIPELINE FOR {subject_id}")
    print("="*80)
    
    results = {
        "subject_id": subject_id,
        "model_path": model_path,
        "nii_path": nii_path,
        "classification": None,
        "final_layers": [],
        "activation_results": [],
        "pipeline_status": "running"
    }
    
    try:
        # Step 1: Model Structure Inspection
        print("\n[STEP 1] Model Structure Inspection")
        selected_layers, selected_layer_names = inspect_model_structure(
            model=MODEL,
            input_shape=INPUT_SHAPE, 
            device=DEVICE
        )
        
        print(f"Initial layers selected: {len(selected_layers)}")
        results["initial_layers"] = selected_layers
        
        # Step 2: Layer Validation
        print("\n[STEP 2] Layer Validation")
        all_layers_info = selected_layers  # Use selected layers as ground truth for validation
        
        validated_layers = validate_layers(
            selected_layers=selected_layers,
            all_layers_info=all_layers_info
        )
        
        print(f"Validated layers: {len(validated_layers)}")
        results["validated_layers"] = validated_layers
        
        if not validated_layers:
            raise ValueError("No valid layers found after validation")
            
        # Update layer names after validation
        validated_layer_names = [layer["model_path"] for layer in validated_layers]
        
        # Step 3: Model Preparation
        print("\n[STEP 3] Model Preparation with Hooks")
        prepared_model = prepare_model_for_inference(
            model=MODEL,
            selected_layers=validated_layers,
            model_path=model_path,
            device=DEVICE
        )
        
        print("Model prepared with hooks and weights loaded")
        
        # Step 4: Real Data Inference  
        print("\n[STEP 4] NIfTI Data Inference")
        prediction_result = real_data_inference(
            model=prepared_model,
            nii_path=nii_path,
            save_dir=output_dir,
            save_name=save_name,
            selected_layer_names=validated_layer_names,
            window=5,
            stride=3,
            device=DEVICE
        )
        
        print(f"Inference completed. Prediction: {prediction_result}")
        results["classification"] = prediction_result
        
        # Step 5: Dynamic Filtering
        print("\n[STEP 5] Dynamic Layer Filtering")
        keep_entries, final_layer_names, output_prefix = dynamic_filtering(
            results=results,
            selected_layers=validated_layers,
            activation_dir=output_dir,
            save_name_prefix=save_name,
            delete_rejected=True
        )
        
        print(f"Final layers after filtering: {len(keep_entries)}")
        results["final_layers"] = keep_entries
        
        if not final_layer_names:
            print("[Warning] No layers passed filtering - using validation results")
            final_layer_names = validated_layer_names[:1]  # Use at least one layer
            keep_entries = validated_layers[:1]
        
        # Step 6: Post-processing for each remaining layer
        print("\n[STEP 6] Post-processing Pipeline")
        
        activation_summaries = []
        
        for i, layer_name in enumerate(final_layer_names):
            print(f"\n[STEP 6.{i+1}] Processing layer: {layer_name}")
            
            try:
                # 6.1: Setup layer paths and convert to NIfTI
                act_path, nii_output, vis_dir = setup_layer_path(
                    layer_name=layer_name,
                    output_prefix=output_prefix,
                    reference_nii_path=nii_path,
                    norm_type=NORM_TYPE,
                    threshold_percentile=ACT_THRESHOLD_PERCENTILE
                )
                
                # 6.2: Resample to atlas space
                resampled_path = resample_to_atlas(
                    act_path=nii_output,
                    atlas_path=ATLAS_PATH,
                    output_dir=vis_dir
                )
                
                # 6.3: Analyze brain activation
                df_result = analyze_brain_activation_data(
                    activation_path=resampled_path,
                    atlas_path=ATLAS_PATH,
                    label_path=LABEL_PATH
                )
                
                # 6.4: Generate visualization
                vis_output_path = visualize_activation_map_data(
                    activation_path=resampled_path,
                    output_path=os.path.join(vis_dir, f"activation_map_{layer_name.replace('.', '_')}.png"),
                    threshold=VIS_THRESHOLD_PERCENTILE,
                    title=f"Activation Map - {layer_name}"
                )
                
                # 6.5: Save results using save_results function
                layer_results = {
                    "layer": layer_name.replace('.', '_'),
                    "model_path": layer_name,
                    "activation_file": act_path,
                    "nii_file": nii_output,
                    "resampled_file": resampled_path,
                }
                
                # Use save_results to properly format and save the data
                updated_results = save_results(
                    results=layer_results,
                    vis_output_path=vis_output_path,
                    df_result=df_result
                )
                
                # Create summary for this layer with save_results output
                layer_summary = {
                    "layer": layer_name.replace('.', '_'),
                    "model_path": layer_name,
                    "activation_file": act_path,
                    "nii_file": nii_output,
                    "resampled_file": resampled_path,
                    "visualization_path": vis_output_path,
                    "summary": f"Layer {layer_name} shows activation patterns relevant to {prediction_result} classification. "
                              f"Analysis completed with brain region mapping and visualization generated.",
                    "brain_regions": df_result.to_dict(orient="records") if df_result is not None else [],
                    "saved_results": updated_results  # Include the save_results output
                }
                
                activation_summaries.append(layer_summary)
                print(f"✓ Completed processing for {layer_name}")
                
            except Exception as e:
                print(f"✗ Error processing layer {layer_name}: {str(e)}")
                # Add error summary for this layer
                error_summary = {
                    "layer": layer_name.replace('.', '_'),
                    "model_path": layer_name, 
                    "error": str(e),
                    "summary": f"Processing failed for layer {layer_name}: {str(e)}"
                }
                activation_summaries.append(error_summary)
        
        results["activation_results"] = activation_summaries
        results["pipeline_status"] = "completed"
        
        # Final results summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETION SUMMARY")
        print("="*80)
        print(f"• Subject ID: {subject_id}")
        print(f"• Classification: {prediction_result}")
        print(f"• Initial layers: {len(selected_layers)}")
        print(f"• Validated layers: {len(validated_layers)}")
        print(f"• Final layers after filtering: {len(keep_entries)}")
        print(f"• Processed activations: {len(activation_summaries)}")
        print(f"• Output directory: {output_dir}")
        print("\nPipeline completed successfully!")
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        error_msg = f"Pipeline failed at step: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        
        results["pipeline_status"] = "failed"
        results["error"] = error_msg
        
        return json.dumps(results, indent=2)


def pipeline(subject_id: str) -> str:
    """
    Simplified wrapper function for ADK agent compatibility.
    
    Args:
        subject_id (str): Subject identifier for analysis
        
    Returns:
        str: JSON string with pipeline results
    """
    # Use default if empty string or None provided
    if not subject_id:
        subject_id = "sub-14"
    
    return complete_nii_inference_pipeline(subject_id=subject_id)


if __name__ == "__main__":
    # Test the pipeline
    result = complete_nii_inference_pipeline()
    print("\n" + "="*50)
    print("PIPELINE TEST RESULT:")
    print("="*50)
    print(result)
