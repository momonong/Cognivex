# app/agents/postprocessing.py
import os
from app.graph.state import AgentState, BrainRegionInfo
# Import all necessary step functions and constants
from app.core.fmri_processing.pipeline_steps import (
    setup_layer_path, resample_to_atlas, analyze_brain_activation_data,
    visualize_activation_map_data,
    OUTPUT_DIR, ATLAS_PATH, LABEL_PATH, NORM_TYPE,
    ACT_THRESHOLD_PERCENTILE, VIS_THRESHOLD_PERCENTILE,
)

# ---### NEW HELPER FUNCTION ###---
# This helper function is co-located in the same file that uses it.
def parse_hemisphere(region_name: str) -> str:
    """
    Parses a brain region name string to determine the hemisphere.
    Example: 'Precuneus_L 71' -> 'Left'
    """
    name_upper = region_name.upper()
    if '_L' in name_upper:
        return 'Left'
    if '_R' in name_upper:
        return 'Right'
    # Default fallback if no hemisphere marker is found
    return 'Bilateral / Unknown'
# ---### END HELPER FUNCTION ###---


def run_post_processing(state: AgentState) -> dict:
    """
    Node 3: Iterates through final layers for post-processing and visualization.
    This version now also enriches the output with hemisphere information.
    """
    print("\n--- Node: 3. Post-processing Layers ---")
    subject_id = state['subject_id']
    nii_path = state['fmri_scan_path']
    # Try to get final_layers first, fallback to validated_layers if filtering wasn't run
    final_layers = state.get('final_layers', state.get('validated_layers', []))
    save_name_prefix = subject_id  # Use the clean subject_id
    output_prefix = os.path.join(OUTPUT_DIR, save_name_prefix)
    
    final_visualization_paths = []
    region_max_activations = {}

    for layer in final_layers:
        layer_name = layer["model_path"]
        print(f"  - Processing layer: {layer_name}")
        try:
            # (Steps 6.1, 6.2, 6.3 - Unchanged)
            _, nii_output, vis_dir = setup_layer_path(
                layer_name=layer_name, output_prefix=output_prefix,
                reference_nii_path=nii_path, norm_type=NORM_TYPE,
                threshold_percentile=ACT_THRESHOLD_PERCENTILE,
            )
            resampled_path = resample_to_atlas(nii_output, ATLAS_PATH, vis_dir)
            df_result = analyze_brain_activation_data(resampled_path, ATLAS_PATH, LABEL_PATH)
            
            # (Step 6.4 - Unchanged)
            vis_output_path = visualize_activation_map_data(
                activation_path=resampled_path,
                output_path=os.path.join(vis_dir, f"map_{layer_name.replace('.', '_')}.png"),
                threshold=VIS_THRESHOLD_PERCENTILE,
                title=f"Activation Map - {layer_name}"
            )
            final_visualization_paths.append(vis_output_path)
            
            # ---### MODIFIED LOGIC ###---
            # Aggregate results, now with hemisphere parsing
            for _, row in df_result.iterrows():
                region_name = row['Region Name']
                activation = row['Mean Activation']
                
                # Call the local helper function to get the hemisphere
                hemisphere = parse_hemisphere(region_name)
                
                if region_name not in region_max_activations or activation > region_max_activations[region_name]['activation_score']:
                    region_max_activations[region_name] = {
                        "region_name": region_name,
                        "activation_score": float(activation),
                        "hemisphere": hemisphere, # Use the parsed value
                    }
            # ---### END MODIFIED LOGIC ###---

        except Exception as e:
            print(f"  - Error processing layer {layer_name}: {e}")
    
    all_regions_info = list(region_max_activations.values())
    all_regions_info.sort(key=lambda x: x['activation_score'], reverse=True)

    trace = f"Node 3: Post-processing complete for {len(final_layers)} layers."

    return {
        "activated_regions": all_regions_info,
        "visualization_paths": final_visualization_paths,
        "trace_log": state.get("trace_log", []) + [trace]
    }