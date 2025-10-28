import torch
import os
import json
import shutil  # For moving files in post-processing
from typing import Dict, Any, List, Tuple, Optional, Union

# --- Configuration ---
from app.core.fmri_processing.model_config import (
    ModelConfig,
    ModelFactory,
    get_config_by_name,
)

# --- Pipeline Component Imports ---
# Step 1: Inspector
from app.core.fmri_processing.pipelines.inspector import inspect_torch_model

# Step 2: Selector
from app.core.fmri_processing.pipelines.choose_layer import select_visualization_layers

# Step 3: Hook Manager (Updated with gradient functions)
from app.core.fmri_processing.pipelines.attach_hook import (
    prepare_model_with_hooks,
    attach_gradient_hooks,
    remove_hooks,
    _gradient_handles,  # Access global list
)

# Step 6: Native Heatmap Generation (Grad-CAM version)
from app.core.fmri_processing.pipelines.act_to_nii import (
    activation_and_gradient_to_nifti,
)

# Step 7: Spatial Normalization (New)
from app.core.fmri_processing.pipelines.spatial_normalizer import normalize_native_heatmap_to_mni

# Step 8: Resample to Atlas (Original version)
from app.core.fmri_processing.pipelines.resample import resample_activation_to_atlas

# Step 9: ROI Analysis
from app.core.fmri_processing.pipelines.brain_map import analyze_brain_activation

# Step 10: Visualization
from app.core.fmri_processing.pipelines.visualize import visualize_gradcam_2d

# Global constants (can be overridden by model config)
DEFAULT_OUTPUT_DIR = "output/generic_pipeline"  # Changed default


class GenericInferencePipeline:
    """
    A generic inference pipeline with Grad-CAM, ANTs normalization,
    and modular post-processing.
    """

    def __init__(
        self,
        model_config: Union[ModelConfig, str],
        model_weights_path: Optional[str] = None,  # Renamed for clarity
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        """
        Initialize the pipeline with a model configuration.
        """
        if isinstance(model_config, str):
            # Assumes get_config_by_name loads necessary paths (template, atlas etc.)
            self.config = get_config_by_name(model_config)
        else:
            self.config = model_config

        self.adapter = ModelFactory.create_adapter(self.config)
        self.model_weights_path = model_weights_path
        self.output_dir = output_dir

        # Initialize model components
        self.model = None
        self.prepared_model = None  # Model with weights loaded
        self.activation_handles = []  # Store forward hook handles

        # Basic validation of required config paths
        required_paths = ["mni_template_path", "atlas_path", "atlas_label_path"]
        for path_key in required_paths:
            if not hasattr(self.config, path_key) or not getattr(self.config, path_key):
                raise ValueError(f"ModelConfig is missing required path: {path_key}")

    def inspect_and_select_layers(
        self,
    ) -> Tuple[List[Dict], List[str]]:  # Renamed method
        """
        Inspect model structure and use LLM to select layers.
        """
        print(
            f"\n--- Step 1: Inspecting {self.config.model_type.value} & Selecting Layers ---"
        )

        if self.model is None:
            self.model = self.adapter.create_model()

        # Inspect without input_shape or device
        layers = inspect_torch_model(self.model)
        if not layers:
            raise RuntimeError("inspect_torch_model returned no layers.")

        print(f"Inspected {len(layers)} potential layers.")

        # Use model-specific layer selection strategy
        strategy = self.adapter.get_layer_selection_strategy()
        print(f"Using selection strategy: {strategy}")
        try:
            # Assumes select_visualization_layers returns a JSON *string*
            response_str = select_visualization_layers(layers, strategy=strategy)
            selected_layers = json.loads(response_str)
        except json.JSONDecodeError:
            print(f"Error: LLM selector response was not valid JSON: {response_str}")
            raise ValueError("Layer selection failed due to invalid LLM response.")
        except Exception as e:
            print(f"Error during layer selection: {e}")
            raise ValueError(f"Layer selection failed: {e}")

        if not selected_layers:
            raise ValueError("LLM selector did not select any layers.")

        # Basic validation (check if selected paths exist)
        all_layer_paths = {layer["model_path"] for layer in layers}
        validated_selection = []
        selected_layer_names = []
        for layer in selected_layers:
            path = layer.get("model_path")
            if path in all_layer_paths:
                validated_selection.append(layer)
                selected_layer_names.append(path)
            else:
                print(f"Warning: Selector chose non-existent layer '{path}'. Ignoring.")

        if not validated_selection:
            raise ValueError("None of the layers selected by the LLM were valid.")

        print(f"LLM selected {len(validated_selection)} valid layers:")
        for layer in validated_selection:
            print(f"  - {layer['model_path']} (Reason: {layer.get('reason', 'N/A')})")

        self.selected_layers = validated_selection  # Store for later use
        return validated_selection, selected_layer_names

    def prepare_model(self) -> torch.nn.Module:  # Simplified method
        """
        Loads model weights and sets to evaluation mode.
        """
        print(f"\n--- Step 2: Preparing {self.config.model_type.value} Model ---")

        if self.model is None:
            self.model = self.adapter.create_model()

        # Load model weights if provided
        if self.model_weights_path:
            if not os.path.exists(self.model_weights_path):
                raise FileNotFoundError(
                    f"Model weights not found at: {self.model_weights_path}"
                )
            print(f"Loading weights from: {self.model_weights_path}")
            try:
                self.model.load_state_dict(
                    torch.load(self.model_weights_path, map_location=self.config.device)
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model weights: {e}")
        else:
            print("Warning: No model weights path provided. Using initialized model.")

        self.model.to(self.config.device).eval()
        self.prepared_model = self.model
        print("Model prepared and set to evaluation mode.")
        return self.prepared_model

    def run_inference_with_hooks(
        self,  # Renamed method
        nii_path: str,
        save_name: str,
        target_layers_info: List[Dict],  # Use the selected layer dicts
        target_class_index: int = 1,
    ) -> Tuple[Any, Dict[str, str]]:
        """
        Run inference, capture activations AND gradients for target layers.

        Returns:
            Tuple (prediction_result, dict mapping layer_path to saved act/grad file path)
        """
        print(f"\n--- Step 3: Running Inference & Capturing Gradients ---")

        if self.prepared_model is None:
            raise ValueError("Model not prepared. Call prepare_model first.")

        model = self.prepared_model
        target_layer_paths = [layer["model_path"] for layer in target_layers_info]

        # --- Preprocessing ---
        inputs = self.adapter.preprocess_data(nii_path)
        print(f"Preprocessed input shape: {inputs.shape}")
        inputs = inputs.to(self.config.device)
        inputs.requires_grad_(True)  # Enable gradient computation

        # --- Attach Hooks ---
        # Forward hooks (store handles for removal)
        _, activations, self.activation_handles = prepare_model_with_hooks(
            model, target_layers_info
        )
        # Backward hooks (handles managed globally in hook_manager)
        gradients = {}
        attach_gradient_hooks(model, target_layer_paths, gradients)

        # --- Run Inference (Forward + Backward) ---
        print("Running forward pass...")
        outputs = model(inputs)

        # Postprocess prediction to get logits
        prediction_result, logits = self.adapter.postprocess_prediction(
            outputs, return_logits=True
        )
        print(f"Prediction result: {prediction_result}")

        print(f"Running backward pass for class index {target_class_index}...")
        model.zero_grad()
        target_score = logits[0, target_class_index]
        target_score.backward()

        # --- Save Activations and Gradients ---
        os.makedirs(self.output_dir, exist_ok=True)
        saved_files_dict = {}  # Map layer_path to file_path

        for layer_path in target_layer_paths:
            safe_name = layer_path.replace(".", "_")
            filename = f"{save_name}_{safe_name}_act_grad.pt"
            save_path = os.path.join(self.output_dir, filename)

            if layer_path in activations and layer_path in gradients:
                data_to_save = {
                    "activation": activations[layer_path],
                    "gradient": gradients[layer_path],
                }
                torch.save(data_to_save, save_path)
                print(f"Saved activation and gradient: {save_path}")
                saved_files_dict[layer_path] = save_path
            else:
                print(
                    f"Warning: Missing activation or gradient for {layer_path}. Cannot save."
                )

        # --- Cleanup Hooks ---
        remove_hooks(self.activation_handles)  # Remove forward hooks
        remove_hooks(_gradient_handles)  # Remove backward hooks

        if not saved_files_dict:
            print("Warning: No activation/gradient files were saved.")

        return prediction_result, saved_files_dict

    def run_post_processing(self,
                            target_layers_info: List[Dict], # Use the selected layer dicts
                            saved_files_dict: Dict[str, str], # Dict from run_inference
                            nii_path: str, # Original T1 path 
                            save_name: str) -> Dict[str, Any]:
            """
            Run post-processing: Native heatmap -> MNI heatmap -> Resample -> Analyze -> Visualize (2D).
            """
            print(f"\n--- Step 4: Running Post-Processing ---")
            
            post_results = {
                "activated_regions": [],
                "visualization_paths": {}, # Map layer_path to *directory* of PNGs
                "final_heatmap_paths": {} # Map layer_path to final resampled nii path
            }
            region_max_activations = {} 
            
            for layer_info in target_layers_info:
                layer_path = layer_info["model_path"]
                print(f"  Processing layer: {layer_path}")
                
                act_grad_pt_path = saved_files_dict.get(layer_path)
                if not act_grad_pt_path:
                    print(f"    Warning: Act/Grad file path missing for {layer_path}. Skipping.")
                    continue

                try:
                    # --- Define Paths for this layer ---
                    safe_layer_name = layer_path.replace(".", "_")
                    native_heatmap_nii = os.path.join(self.output_dir, f"{save_name}_{safe_layer_name}_native_heatmap.nii.gz") 
                    mni_heatmap_nii = os.path.join(self.output_dir, f"{save_name}_{safe_layer_name}_mni_heatmap.nii.gz")
                    resampled_nii = os.path.join(self.output_dir, f"{save_name}_{safe_layer_name}_resampled_heatmap.nii.gz")
                    # Directory for BOTH resampled NIfTI and PNGs
                    vis_dir = os.path.join(self.output_dir, "visualizations", f"{save_name}_{safe_layer_name}") 
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # --- Step 6 Call: Generate Native Space Heatmap (GradCAM) ---
                    print(f"    Generating native heatmap...")
                    success_native = activation_and_gradient_to_nifti(
                        data_path=act_grad_pt_path, 
                        reference_nii_path=nii_path, 
                        output_path=native_heatmap_nii,
                    )
                    if not success_native: raise RuntimeError("Native heatmap generation failed")

                    # --- Step 7 Call: Normalize to MNI ---
                    print(f"    Normalizing heatmap to MNI space...")
                    success_norm = normalize_native_heatmap_to_mni( 
                        t1_native_path=nii_path, 
                        heatmap_native_path=native_heatmap_nii,
                        mni_template_path=self.config.mni_template_path, 
                        output_path=mni_heatmap_nii,
                        transform_type='SyN', 
                        interpolator='linear'
                    )
                    if not success_norm: raise RuntimeError("Normalization failed")
                    
                    # --- Step 8 Call: Resample to Atlas ---
                    print(f"    Resampling heatmap to atlas grid...")
                    # Save resampled file directly to final path within vis_dir
                    final_resampled_path_in_vis_dir = os.path.join(vis_dir, os.path.basename(resampled_nii))
                    resampled_success_path = resample_activation_to_atlas( # Function returns path on success
                        act_path=mni_heatmap_nii, 
                        atlas_path=self.config.atlas_path, 
                        output_dir=vis_dir, # Output directly to vis_dir
                        interpolation='linear'
                    )
                    if not resampled_success_path: raise RuntimeError("Resampling failed")
                    # Rename if needed (resample function might create its own name)
                    if os.path.abspath(resampled_success_path) != os.path.abspath(final_resampled_path_in_vis_dir):
                        print(f"    Renaming resampled file to {os.path.basename(final_resampled_path_in_vis_dir)}")
                        shutil.move(resampled_success_path, final_resampled_path_in_vis_dir)
                    else:
                        final_resampled_path_in_vis_dir = resampled_success_path # Use the path returned

                    post_results["final_heatmap_paths"][layer_path] = final_resampled_path_in_vis_dir
                    
                    # --- Step 9 Call: Analyze Brain Activation ---
                    print(f"    Analyzing brain regions...")
                    df_result = analyze_brain_activation(
                        activation_path=final_resampled_path_in_vis_dir, # Use the resampled map
                        atlas_path=self.config.atlas_path,
                        label_path=self.config.atlas_label_path, 
                    )
                    
                    # --- !!! CHANGE HERE !!! ---
                    # --- Step 10 Call: Generate 2D Visualization ---
                    print(f"    Generating 2D Grad-CAM visualizations...")
                    success_vis = visualize_gradcam_2d(
                        data_path=act_grad_pt_path, # Needs the original act/grad data
                        reference_nii_path=nii_path, # Needs the original T1
                        output_dir=vis_dir # Save PNGs in the vis_dir
                        # colormap and alpha use defaults or add to config
                    )
                    if success_vis:
                        # Store the DIRECTORY where PNGs are saved
                        post_results["visualization_paths"][layer_path] = vis_dir 
                    else:
                        print(f"    Warning: 2D visualization generation failed for {layer_path}")
                    # --- !!! END CHANGE !!! ---
                    
                    # --- Aggregate region activations (unchanged) ---
                    for _, row in df_result.iterrows():
                        # ... (aggregation logic unchanged) ...
                        region_name = row['Region Name']; activation = row['Mean Activation'] # etc.
                        # ... (update region_max_activations) ...
                    
                except Exception as e:
                    print(f"    Error processing layer {layer_path}: {e}")
                    continue
            
            # --- Final aggregation and return (unchanged) ---
            all_regions_info = list(region_max_activations.values())
            all_regions_info.sort(key=lambda x: x['activation_score'], reverse=True)
            post_results["activated_regions"] = all_regions_info
            print(f"  Post-processing complete.")
            return post_results

    def _parse_hemisphere(self, region_name: str) -> str:
        # ... (function content unchanged) ...
        name_upper = region_name.upper()
        # ... rest of logic ...
        if "_L" in name_upper:
            return "Left"
        elif "_R" in name_upper:
            return "Right"
        else:
            return "Bilateral / Unknown"

    def run_full_pipeline(
        self,
        nii_path: str,
        save_name: str,
        include_post_processing: bool = True,  # Default to True now
        target_class_index: int = 1,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline: Inspect -> Select -> Prepare -> Infer/Hook -> PostProcess.
        """
        results = {}

        try:
            # Step 1: Inspect model and select layers
            selected_layers, selected_layer_names = self.inspect_and_select_layers()
            results["selected_layers"] = selected_layers

            # Step 2: Prepare model (load weights)
            prepared_model = self.prepare_model()

            # Step 3: Run inference (forward/backward, save act/grad)
            prediction_result, saved_files_dict = self.run_inference_with_hooks(
                nii_path, save_name, selected_layers, target_class_index
            )
            results["prediction_result"] = prediction_result
            results["activation_gradient_files"] = saved_files_dict

            # Step 4: Post-processing
            if include_post_processing:
                if not saved_files_dict:
                    print(
                        "Warning: No activation/gradient files saved, skipping post-processing."
                    )
                else:
                    post_processing_results = self.run_post_processing(
                        selected_layers,  # Pass the layers selected in Step 1
                        saved_files_dict,
                        nii_path,
                        save_name,
                    )
                    results.update(post_processing_results)

            print(f"\n--- Pipeline Complete for {self.config.model_type.value} ---")
            print(f"Prediction: {prediction_result}")

            return results

        except Exception as e:
            error_message = f"Pipeline error: {e}"
            print(f"ERROR: {error_message}")
            results["error"] = error_message
            # Optional: Add traceback details
            import traceback

            results["traceback"] = traceback.format_exc()
            return results


# --- Backward Compatibility Function (Simplified) ---
# This function might need more context on how 'state' is used in LangGraph
def run_inference_and_classification(
    state: Dict[str, Any], model_config: Union[ModelConfig, str] = "papermodel"
) -> Dict[str, Any]:  # Changed default
    """
    Simplified backward-compatible function for inference only.
    Use GenericInferencePipeline.run_full_pipeline for the complete workflow.
    """
    print("\n--- Node: Generic Inference & Classification (Inference Only) ---")

    subject_id = state.get("subject_id", "unknown_subject")
    model_weights = state.get(
        "model_path"
    )  # Assuming 'model_path' in state means weights path
    nii_path = state.get("fmri_scan_path")  # Assuming this is the T1 path now
    save_name = f"{subject_id}_inference_only"  # Different save name
    output_dir = state.get(
        "output_dir", DEFAULT_OUTPUT_DIR
    )  # Allow override from state

    if not nii_path:
        return {
            "error_log": state.get("error_log", [])
            + ["Missing 'fmri_scan_path' in state."]
        }

    try:
        # Create pipeline instance
        pipeline = GenericInferencePipeline(
            model_config=model_config,
            model_weights_path=model_weights,
            output_dir=output_dir,
        )

        # --- Run only the necessary steps for prediction ---
        # 1. Prepare model (loads weights)
        pipeline.prepare_model()

        # 2. Preprocess data (needed for inference)
        inputs = pipeline.adapter.preprocess_data(nii_path).to(pipeline.config.device)

        # 3. Run basic inference (NO hooks, NO gradients)
        with torch.no_grad():
            outputs = pipeline.prepared_model(inputs)

        # 4. Postprocess prediction
        prediction_result, _ = pipeline.adapter.postprocess_prediction(
            outputs, return_logits=True
        )

        trace = f"Node: Generic inference complete. Prediction: {prediction_result}"

        # Return only prediction and trace
        return {
            "classification_result": prediction_result,
            # "validated_layers": [], # No layers selected/validated in this path
            "trace_log": state.get("trace_log", []) + [trace],
        }

    except Exception as e:
        error_message = f"Node (Generic Inference Only) Error: {e}"
        print(f"\n[ERROR] {error_message}")
        import traceback

        print(traceback.format_exc())  # Print full traceback for debugging
        return {"error_log": state.get("error_log", []) + [error_message]}


# Export necessary components
__all__ = [
    "GenericInferencePipeline",
    "run_inference_and_classification",  # Kept for backward compatibility
    "ModelConfig",
    "ModelFactory",
    "get_config_by_name",
    # We might want to export the individual step functions too if needed elsewhere
    "inspect_torch_model",
    "select_visualization_layers",
    "prepare_model_with_hooks",
    "activation_and_gradient_to_nifti",
    "normalize_native_heatmap_to_mni",  # Export the new function
    "resample_activation_to_atlas",
    "analyze_brain_activation",
    "visualize_activation_map",
]


if __name__ == "__main__":
    
    print("--- Starting GenericInferencePipeline Test ---")
    
    # --- Configuration for the Test ---
    
    # 1. Specify the model configuration name
    CONFIG_NAME = "papermodel" # Use the name defined in model_config.py
    
    # 2. Specify the input T1 NIfTI file
    #    (Using the same example subject as before)
    INPUT_NIFTI_PATH = "/Volumes/3T-disk/fMRI/Model/sMRI_data/AD/T1_3D_MPRAGE_SAG_0003_008/T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii"
    
    # 3. Specify the path to model weights (OPTIONAL)
    #    If you don't have trained weights yet, set this to None.
    #    The pipeline will run with the initialized model (useful for testing flow).
    MODEL_WEIGHTS = "model/shufflenet/fold_3_best_model.pth" # Or "path/to/your/fold_X_best_model.pth"
    
    # 4. Define a base name for output files related to this run
    SAVE_NAME_PREFIX = "test_subject_008" 
    
    # 5. Define the main output directory for this test run
    TEST_OUTPUT_DIR = f"output/pipeline_test_run_{SAVE_NAME_PREFIX}"
    
    # --- Check required files ---
    if not os.path.exists(INPUT_NIFTI_PATH):
        print(f"Error: Input T1 NIfTI not found at: {INPUT_NIFTI_PATH}")
        exit()
        
    # Check if config exists (get_config_by_name will raise error if not)
    try:
        test_config = get_config_by_name(CONFIG_NAME)
        print(f"Loaded config: {CONFIG_NAME}")
        # Check essential paths defined in the config
        if not os.path.exists(test_config.mni_template_path):
             print(f"Error: MNI template specified in config not found: {test_config.mni_template_path}")
             exit()
        if not os.path.exists(test_config.atlas_path):
             print(f"Error: Atlas NIfTI specified in config not found: {test_config.atlas_path}")
             exit()
        if not os.path.exists(test_config.atlas_label_path):
             print(f"Error: Atlas Label file specified in config not found: {test_config.atlas_label_path}")
             exit()
             
    except ValueError as e:
        print(f"Error loading config '{CONFIG_NAME}': {e}")
        exit()

    # --- Instantiate and Run the Pipeline ---
    try:
        print("\nInstantiating GenericInferencePipeline...")
        pipeline = GenericInferencePipeline(
            model_config=CONFIG_NAME,
            model_weights_path=MODEL_WEIGHTS,
            output_dir=TEST_OUTPUT_DIR # Use the specific test output directory
        )
        
        print("\nRunning full pipeline...")
        results = pipeline.run_full_pipeline(
            nii_path=INPUT_NIFTI_PATH,
            save_name=SAVE_NAME_PREFIX,
            include_post_processing=True, # Run all steps including post-processing
            target_class_index=1 # Assume we want to explain class 1 (AD)
        )
        
        print("\n--- Pipeline Finished ---")
        
        # --- Print Summary of Results ---
        if "error" in results:
            print(f"Pipeline failed with error: {results['error']}")
            if "traceback" in results:
                 print("\nTraceback:")
                 print(results["traceback"])
        else:
            print("Pipeline completed successfully!")
            print(f"Prediction Result: {results.get('prediction_result')}")
            
            if "selected_layers" in results:
                print(f"Selected Layers: {[layer['model_path'] for layer in results['selected_layers']]}")
                
            if "final_heatmap_paths" in results:
                 print("\nFinal Heatmap NIfTI Files (aligned to atlas):")
                 for layer, path in results["final_heatmap_paths"].items():
                     print(f"  - {layer}: {path}")
                     
            if "visualization_paths" in results:
                 print("\nVisualization PNG Files:")
                 for layer, path in results["visualization_paths"].items():
                     print(f"  - {layer}: {path}")

            if "activated_regions" in results:
                print("\nTop 5 Activated Regions:")
                # Convert list of dicts to DataFrame for nice printing
                try:
                    import pandas as pd
                    df_regions = pd.DataFrame(results["activated_regions"])
                    print(df_regions.head(5).to_string(index=False))
                except ImportError:
                     print(results["activated_regions"][:5]) # Print raw list if pandas not available
                     
    except Exception as e:
        print(f"\n--- Critical Error Running Pipeline ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()