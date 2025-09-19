# app/agents/3_post_processing.py
import os
from app.graph.state import AgentState, BrainRegionInfo
# Import all necessary step functions and constants
from app.core.fmri_processing.pipeline_steps import (
    setup_layer_path, resample_to_atlas, analyze_brain_activation_data,
    visualize_activation_map_data,
    OUTPUT_DIR, ATLAS_PATH, LABEL_PATH, NORM_TYPE,
    ACT_THRESHOLD_PERCENTILE, VIS_THRESHOLD_PERCENTILE,
)

def run_post_processing(state: AgentState) -> dict:
    """
    Node 3: Iterates through final layers for post-processing and visualization.
    """
    print("\n--- Node: 3. Post-processing Layers ---")
    subject_id = state['subject_id']
    nii_path = state['fmri_scan_path']
    final_layers = state.get('final_layers', [])
    save_name_prefix = f"{subject_id}"
    output_prefix = os.path.join(OUTPUT_DIR, save_name_prefix)
    
    final_visualization_paths = []
    # This will be a list of BrainRegionInfo objects, aggregated from all layers
    all_regions_info: list[BrainRegionInfo] = []
    
    # Use a dictionary to aggregate the highest activation score for each region
    region_max_activations = {}

    for layer in final_layers:
        layer_name = layer["model_path"]
        print(f"  - Processing layer: {layer_name}")
        try:
            # Step 6.1: Setup and Convert
            _, nii_output, vis_dir = setup_layer_path(
                layer_name=layer_name, output_prefix=output_prefix,
                reference_nii_path=nii_path, norm_type=NORM_TYPE,
                threshold_percentile=ACT_THRESHOLD_PERCENTILE,
            )
            # Step 6.2: Resample
            resampled_path = resample_to_atlas(nii_output, ATLAS_PATH, vis_dir)
            
            # Step 6.3: Analyze
            df_result = analyze_brain_activation_data(resampled_path, ATLAS_PATH, LABEL_PATH)
            
            # Step 6.4: Visualize
            vis_output_path = visualize_activation_map_data(
                activation_path=resampled_path,
                output_path=os.path.join(vis_dir, f"map_{layer_name.replace('.', '_')}.png"),
                threshold=VIS_THRESHOLD_PERCENTILE,
                title=f"Activation Map - {layer_name}"
            )
            final_visualization_paths.append(vis_output_path)
            
            # Aggregate results into our structured BrainRegionInfo
            for _, row in df_result.iterrows():
                region_name = row['Region Name']
                activation = row['Mean Activation']
                
                # If region is new or current activation is higher, update it
                if region_name not in region_max_activations or activation > region_max_activations[region_name]['activation_score']:
                    region_max_activations[region_name] = {
                        "region_name": region_name,
                        "activation_score": float(activation),
                        "hemisphere": "Unknown", # You might need to parse this from the name
                    }

        except Exception as e:
            print(f"  - Error processing layer {layer_name}: {e}")
    
    # Convert the aggregated dictionary to a list of BrainRegionInfo
    all_regions_info = list(region_max_activations.values())
    # Sort by activation score, descending
    all_regions_info.sort(key=lambda x: x['activation_score'], reverse=True)

    trace = f"Node 3: Post-processing complete for {len(final_layers)} layers."

    return {
        "activated_regions": all_regions_info,
        "visualization_paths": final_visualization_paths,
        "trace_log": state.get("trace_log", []) + [trace]
    }