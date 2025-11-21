# File: app/core/fmri_processing/generic_pipeline_steps.py
"""
Generic Pipeline Steps for fMRI Processing with XAI capabilities.

Provides a modular pipeline class `GenericInferencePipeline` that integrates
model inference, Grad-CAM heatmap generation, spatial normalization (ANTs),
atlas alignment, ROI analysis, and 2D visualization.
"""

import torch
import os
import json
import shutil  # For moving/renaming files in post-processing
import time    # For timing steps if needed
import traceback # For detailed error reporting
from typing import Dict, Any, List, Tuple, Optional, Union

# --- Configuration ---
from app.core.mri_processing.model_config import (
    ModelConfig,
    ModelFactory,
    get_config_by_name,
)

# --- Pipeline Component Imports ---
# Step 1: Inspector (Simple version, no torchsummary)
from app.core.mri_processing.pipelines.inspector import inspect_torch_model 
# Step 2: Selector (Uses LLM)
from app.core.mri_processing.pipelines.choose_layer import select_visualization_layers 
# Step 3: Hook Manager (Includes gradient hooks)
from app.core.mri_processing.pipelines.attach_hook import (
    prepare_model_with_hooks, 
    attach_gradient_hooks, 
    remove_hooks, 
    _gradient_handles # Access global list for cleanup
)
# Step 6: Native Heatmap Generation (Grad-CAM version)
from app.core.mri_processing.pipelines.act_to_nii import activation_and_gradient_to_nifti 
# Step 7: Spatial Normalization (ANTs)
from app.core.mri_processing.pipelines.spatial_normalizer import normalize_native_heatmap_to_mni_accurate_masked # Use the accurate masked version
# Step 8: Resample to Atlas (Original version)
from app.core.mri_processing.pipelines.resample import resample_activation_to_atlas 
# Step 9: ROI Analysis
from app.core.mri_processing.pipelines.brain_map import analyze_brain_activation 
# Step 10: Visualization (New 2D version)
from app.core.mri_processing.pipelines.visualize import visualize_gradcam_2d 

# Global constants (can be overridden by model config or pipeline init)
DEFAULT_OUTPUT_DIR = "output/generic_pipeline" 

class GenericInferencePipeline:
    """
    A generic inference pipeline integrating XAI steps:
    1. Inspect Model & Select Layer (LLM)
    2. Prepare Model (Load Weights)
    3. Run Inference & Capture Activations/Gradients (Hooks)
    4. Post-Processing:
        a. Generate Native Heatmap (Grad-CAM)
        b. Normalize Heatmap to MNI (ANTs)
        c. Resample Heatmap to Atlas Grid
        d. Analyze ROI Activations
        e. Visualize 2D Overlays
    """

    def __init__(
        self,
        model_config: Union[ModelConfig, str],
        model_weights_path: Optional[str] = None,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        layer_selection_strategy: Optional[str] = None, # Allow overriding strategy
    ):
        """
        Initialize the pipeline with a model configuration.
        """
        print("\n--- Initializing GenericInferencePipeline ---")
        if isinstance(model_config, str):
            print(f"Loading configuration: '{model_config}'")
            try:
                self.config = get_config_by_name(model_config)
            except ValueError as e:
                 print(f"Error: {e}")
                 raise
        else:
            self.config = model_config
            print(f"Using provided ModelConfig for type: {self.config.model_type}")

        # Validate required paths in the loaded config
        print("Validating configuration paths...")
        required_paths = ["mni_template_path", "atlas_path", "atlas_label_path"]
        missing_paths = []
        for path_key in required_paths:
            path_val = getattr(self.config, path_key, None)
            if not path_val or not os.path.exists(path_val):
                missing_paths.append(f"{path_key} (value: {path_val})")
        if missing_paths:
             raise FileNotFoundError(f"Missing or invalid required file paths in config: {', '.join(missing_paths)}")
        print("Configuration paths validated.")

        self.adapter = ModelFactory.create_adapter(self.config)
        self.model_weights_path = model_weights_path
        self.output_dir = output_dir
        self.layer_selection_strategy = layer_selection_strategy # Store the strategy
        os.makedirs(self.output_dir, exist_ok=True) # Ensure base output dir exists

        # Initialize model components
        self.model: Optional[torch.nn.Module] = None
        self.prepared_model: Optional[torch.nn.Module] = None
        self.activation_handles: List[torch.utils.hooks.RemovableHandle] = [] 

    def inspect_and_select_layers(self) -> Tuple[List[Dict], List[str]]:
        """
        Step 1: Inspect model structure and use LLM to select layers.
        (With custom override for PaperModel/CNN_2D)
        """
        print(f"\n--- Step 1: Inspecting {self.config.model_type.value} & Selecting Layers ---")

        if self.model is None:
            self.model = self.adapter.create_model()

        layers = inspect_torch_model(self.model)
        if not layers:
            raise RuntimeError("inspect_torch_model returned no layers.")
        print(f"Inspected {len(layers)} potential layers.")

        # Determine the selection strategy
        strategy = self.layer_selection_strategy or self.adapter.get_layer_selection_strategy()
        print(f"Using selection strategy: '{strategy}'")

        # --- CUSTOM MODIFICATION FOR SHUFFLENET/PAPERMODEL ---
        from app.core.mri_processing.model_config import ModelType
        if strategy == "force_select_stage2" and self.config.model_type == ModelType.CNN_2D:
            print("INFO: PaperModel (CNN_2D) detected with 'force_select_stage2' strategy. Bypassing LLM.")
            forced_layer_path = "backbone.stage2"
            selected_layers = [{"model_path": forced_layer_path, "output_shape": "N/A", "reason": "Forced selection for better resolution."}]
            print(f"Force-selected layer: {forced_layer_path}")
        else:
            # --- ORIGINAL LLM-BASED SELECTION FOR OTHER MODELS ---
            print("Using LLM-based selection...")
            try:
                response_str = select_visualization_layers(layers, strategy=strategy)
                selected_layers = json.loads(str(response_str))
            except json.JSONDecodeError:
                print(f"Error: LLM selector response was not valid JSON: {response_str}")
                raise ValueError("Layer selection failed: Invalid LLM JSON response.")
            except Exception as e:
                print(f"Error during layer selection LLM call: {e}")
                raise ValueError(f"Layer selection failed: {e}")

        if not selected_layers or not isinstance(selected_layers, list):
            raise ValueError(f"Selector did not return a valid list of layers. Response: {selected_layers}")

        # Basic validation
        all_layer_paths = {layer["model_path"] for layer in layers}
        validated_selection = []
        selected_layer_names = []
        for layer in selected_layers:
            path = layer.get("model_path")
            if path in all_layer_paths:
                validated_selection.append(layer)
                selected_layer_names.append(path)
            else:
                # If we forced selection, this error is critical.
                if self.config.model_type == ModelType.CNN_2D:
                    raise ValueError(f"Forced layer '{path}' not found in model's layers.")
                print(f"Warning: Selector chose non-existent layer '{path}'. Ignoring.")

        if not validated_selection:
            raise ValueError("None of the selected layers were valid or found in the model.")

        print(f"Final selected {len(validated_selection)} valid layer(s):")
        for layer in validated_selection:
            print(f"  - {layer['model_path']} (Reason: {layer.get('reason', 'N/A')})")

        return validated_selection, selected_layer_names

    def prepare_model(self) -> torch.nn.Module: 
        """
        Step 2: Loads model weights and sets to evaluation mode.
        """
        print(f"\n--- Step 2: Preparing {self.config.model_type.value} Model ---")
        
        if self.model is None:
            self.model = self.adapter.create_model()
            
        # Load model weights if provided
        if self.model_weights_path:
            if not os.path.exists(self.model_weights_path):
                 raise FileNotFoundError(f"Model weights not found at: {self.model_weights_path}")
            print(f"Loading weights from: {self.model_weights_path}")
            try:
                # Ensure weights are loaded to the correct device specified in config
                state_dict = torch.load(self.model_weights_path, map_location=self.config.device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                 raise RuntimeError(f"Failed to load model weights from {self.model_weights_path}: {e}")
        else:
             print("Warning: No model weights path provided. Using initialized model (likely random predictions).")
             
        self.model.to(self.config.device).eval() # Move to device and set to eval mode
        self.prepared_model = self.model
        print(f"Model prepared on device '{self.config.device}' and set to evaluation mode.")
        return self.prepared_model

    def run_inference_with_hooks(self, 
                     nii_path: str, 
                     save_name: str, 
                     target_layers_info: List[Dict], 
                     target_class_index: int = 1) -> Tuple[Any, Dict[str, str]]:
        """
        Step 3: Run inference, capture activations AND gradients for target layers.
        """
        print(f"\n--- Step 3: Running Inference & Capturing Gradients ---")
        
        if self.prepared_model is None:
            raise ValueError("Model not prepared. Call prepare_model first.")
        
        model = self.prepared_model 
        target_layer_paths = [layer["model_path"] for layer in target_layers_info]
        
        # --- Preprocessing ---
        print(f"Preprocessing input NIfTI: {nii_path}")
        try:
            inputs = self.adapter.preprocess_data(nii_path)
            print(f"Preprocessed input shape: {inputs.shape}")
            inputs = inputs.to(self.config.device)
            inputs.requires_grad_(True) 
        except Exception as e:
             raise RuntimeError(f"Data preprocessing failed for {nii_path}: {e}")

        # --- Attach Hooks ---
        print(f"Attaching hooks to layers: {target_layer_paths}")
        # Forward hooks (store handles for removal)
        _ , activations, self.activation_handles = prepare_model_with_hooks(model, target_layers_info) 
        # Backward hooks (handles managed globally in hook_manager)
        gradients = {} 
        attach_gradient_hooks(model, target_layer_paths, gradients) 

        # --- Run Inference (Forward + Backward) ---
        prediction_result = None
        logits = None
        try:
            print("Running forward pass...")
            outputs = model(inputs)
            
            # Postprocess prediction to get result and logits
            prediction_result, logits = self.adapter.postprocess_prediction(outputs, return_logits=True)
            print(f"Prediction result: {prediction_result}")
            
            if logits is None:
                 raise RuntimeError("Adapter did not return logits needed for backward pass.")
                 
            print(f"Running backward pass for class index {target_class_index}...")
            model.zero_grad() 
            # Ensure target score selection is robust
            if logits.ndim > 1 and logits.shape[0] == 1 and logits.shape[1] > target_class_index:
                 target_score = logits[0, target_class_index] 
            elif logits.ndim == 1 and logits.shape[0] > target_class_index: # Handle 1D output if needed
                 target_score = logits[target_class_index]
            else:
                 raise ValueError(f"Cannot select target_class_index {target_class_index} from logits with shape {logits.shape}")
                 
            target_score.backward() 
            print("Backward pass complete.")
            
        except Exception as e:
            # Ensure hooks are removed even if inference fails
            remove_hooks(self.activation_handles) 
            remove_hooks(_gradient_handles)
            raise RuntimeError(f"Error during model inference or backpropagation: {e}")

        # --- Save Activations and Gradients ---
        saved_files_dict = {} 
        save_error = False
        for layer_path in target_layer_paths:
            safe_name = layer_path.replace('.', '_')
            filename = f"{save_name}_{safe_name}_act_grad.pt" 
            save_path = os.path.join(self.output_dir, filename)
            
            if layer_path in activations and layer_path in gradients:
                try:
                    data_to_save = {'activation': activations[layer_path], 'gradient': gradients[layer_path]}
                    torch.save(data_to_save, save_path)
                    print(f"Saved activation and gradient: {save_path}")
                    saved_files_dict[layer_path] = save_path
                except Exception as e:
                     print(f"Error saving data for layer {layer_path} to {save_path}: {e}")
                     save_error = True
            else:
                print(f"Warning: Missing activation or gradient for {layer_path}. Cannot save.")
                if layer_path not in activations: print("  Activation missing.")
                if layer_path not in gradients: print("  Gradient missing.")
                save_error = True # Treat missing data as an error for this step

        # --- Cleanup Hooks ---
        remove_hooks(self.activation_handles) 
        remove_hooks(_gradient_handles)
        print("Hooks removed.")

        if not saved_files_dict and not save_error:
             print("Warning: No target layers specified or found, no files saved.")
        elif not saved_files_dict and save_error:
             raise RuntimeError("Failed to save any activation/gradient files.")
             
        return prediction_result, saved_files_dict
        
    def run_post_processing(self,
                            target_layers_info: List[Dict], 
                            saved_files_dict: Dict[str, str], 
                            nii_path: str, # Original T1 path 
                            save_name: str) -> Dict[str, Any]:
            """
            Step 4: Run post-processing: Native heatmap -> MNI heatmap -> Resample -> Analyze -> Visualize (2D).
            (V2: Organized outputs into layer-specific subfolders)
            """
            print(f"\n--- Step 4: Running Post-Processing ---")
            
            post_results = {
                "activated_regions": [],
                "visualization_paths": {}, # Map layer_path to *directory* of PNGs
                "final_heatmap_paths": {} # Map layer_path to final resampled nii path
            }
            region_max_activations = {} 
            
            if not target_layers_info:
                print("  No target layers provided for post-processing. Skipping.")
                return post_results
                
            for layer_info in target_layers_info:
                layer_path = layer_info["model_path"]
                print(f"\n  Processing layer: {layer_path}")
                
                act_grad_pt_path = saved_files_dict.get(layer_path)
                if not act_grad_pt_path:
                    print(f"    Warning: Act/Grad file path missing for {layer_path}. Skipping post-processing for this layer.")
                    continue

                try:
                    # --- [NEW] Organized Path Definitions ---
                    safe_layer_name = layer_path.replace(".", "_")
                    
                    # Create a single, organized directory for this layer's outputs
                    layer_output_dir = os.path.join(self.output_dir, f"layer_{safe_layer_name}")
                    os.makedirs(layer_output_dir, exist_ok=True)
                    
                    # Step 6 Output: Native heatmap
                    native_heatmap_nii = os.path.join(layer_output_dir, f"{save_name}_01_native_heatmap.nii.gz") 
                    
                    # Step 7 Output: ANTs normalization files
                    # All ANTs files (transforms, warped T1, normalized heatmap) will go here
                    ants_output_dir = os.path.join(layer_output_dir, "02_ants_normalization")
                    os.makedirs(ants_output_dir, exist_ok=True)
                    # Use a simple prefix; ANTs function will add its own suffixes
                    ants_output_prefix = os.path.join(ants_output_dir, f"{save_name}_") 
                    
                    # Step 8 Output: Final resampled heatmap
                    resampled_nii = os.path.join(layer_output_dir, f"{save_name}_03_final_heatmap_resampled_to_atlas.nii.gz")
                    
                    # Step 10 Output: Visualization directory
                    vis_dir = os.path.join(layer_output_dir, "04_visualizations_2d_native") 
                    os.makedirs(vis_dir, exist_ok=True)
                    # --- [END NEW] Path Definitions ---

                    
                    # --- Step 6 Call: Generate Native Space Heatmap (GradCAM) ---
                    print(f"    6. Generating native heatmap...")
                    success_native = activation_and_gradient_to_nifti(
                        data_path=act_grad_pt_path, 
                        reference_nii_path=nii_path, 
                        output_path=native_heatmap_nii,
                    )
                    if not success_native: raise RuntimeError("Native heatmap generation failed")

                    # --- Step 7 Call: Normalize to MNI (with fallback) ---
                    print(f"    7. Normalizing heatmap to MNI space...")
                    
                    mni_heatmap_nii = "" # Initialize path
                    try:
                        # Try the accurate masked version
                        normalized_heatmap_path_or_none = normalize_native_heatmap_to_mni_accurate_masked( 
                            t1_native_path=nii_path, 
                            heatmap_native_path=native_heatmap_nii,
                            mni_template_path=self.config.mni_template_path, 
                            output_prefix=ants_output_prefix, # Pass new prefix
                            transform_type='SyN', 
                            interpolator='linear'
                        )
                        
                        if normalized_heatmap_path_or_none and os.path.exists(normalized_heatmap_path_or_none):
                            mni_heatmap_nii = normalized_heatmap_path_or_none
                            print(f"   ✅ ANTs normalization successful: {mni_heatmap_nii}")
                        else:
                            # Check expected path
                            expected_mni_path = f"{ants_output_prefix}_heatmap_MNI_masked_accurate.nii.gz"
                            if os.path.exists(expected_mni_path):
                                mni_heatmap_nii = expected_mni_path
                                print(f"   ✅ ANTs output found at expected path: {mni_heatmap_nii}")
                            else:
                                raise RuntimeError("ANTs normalization failed")
                                
                    except Exception as e:
                        print(f"   ⚠️  ANTs normalization failed: {e}")
                        print(f"   🔄 Falling back to native space analysis...")
                        # Use native heatmap directly for atlas analysis
                        mni_heatmap_nii = native_heatmap_nii

                    # --- Step 8 Call: Resample to Atlas ---
                    print(f"    8. Resampling heatmap to atlas grid...")
                    resampled_success_path = resample_activation_to_atlas( 
                        act_path=mni_heatmap_nii, # Use the normalized heatmap
                        atlas_path=self.config.atlas_path, 
                        output_dir=layer_output_dir, # Output to the layer's main dir
                        interpolation='linear'
                    )
                    if not resampled_success_path: raise RuntimeError("Resampling failed")
                    
                    # Rename the output file to our new, clean name
                    if os.path.abspath(resampled_success_path) != os.path.abspath(resampled_nii):
                        print(f"    Renaming resampled file to {os.path.basename(resampled_nii)}")
                        shutil.move(resampled_success_path, resampled_nii)
                    else:
                        resampled_nii = resampled_success_path # Use the path returned

                    post_results["final_heatmap_paths"][layer_path] = resampled_nii
                    
                    # --- Step 9 Call: Analyze Brain Activation ---
                    print(f"    9. Analyzing brain regions...")
                    df_result = analyze_brain_activation(
                        activation_path=resampled_nii, # Use the final resampled map
                        atlas_path=self.config.atlas_path,
                        label_path=self.config.atlas_label_path, 
                    )
                    if df_result is None or df_result.empty:
                        print("    Warning: Brain activation analysis returned empty results.")
                        # Store empty results to avoid errors later
                        post_results["activated_regions"] = [] 
                    else:
                        # Aggregate region activations
                        for _, row in df_result.iterrows():
                            region_name = row['Region Name']; activation = row['Mean Activation'] 
                            hemisphere = self._parse_hemisphere(region_name)
                            # Use Mean Activation for comparison
                            if (region_name not in region_max_activations or 
                                activation > region_max_activations[region_name]['activation_score']):
                                region_max_activations[region_name] = {
                                    "region_name": region_name,
                                    "activation_score": float(activation), # Use Mean Activation
                                    "hemisphere": hemisphere,
                                    "voxel_count": int(row['Voxel Count']), # Add voxel count for context
                                    "total_activation": float(row['Total Activation']) # Add total activation
                                }
                    
                    # --- Step 10 Call: Generate 2D Visualization ---
                    print(f"    10. Generating 2D Grad-CAM visualizations...")
                    success_vis = visualize_gradcam_2d(
                        data_path=act_grad_pt_path, 
                        reference_nii_path=nii_path, 
                        output_dir=vis_dir # Use new visualization dir
                    )
                    if success_vis:
                        post_results["visualization_paths"][layer_path] = vis_dir 
                    else:
                        print(f"    Warning: 2D visualization generation failed for {layer_path}")

                except Exception as e:
                    print(f"    ERROR processing layer {layer_path}: {e}")
                    # Optional: Log detailed traceback
                    # print(traceback.format_exc())
                    continue # Continue to the next layer if one fails
            
            # --- Final aggregation ---
            all_regions_info = list(region_max_activations.values())
            # Sort by Mean Activation Score
            all_regions_info.sort(key=lambda x: x['activation_score'], reverse=True) 
            post_results["activated_regions"] = all_regions_info
            print(f"  Post-processing finished.")
            return post_results

    def _parse_hemisphere(self, region_name: str) -> str:
        # ... (function content unchanged) ...
        name_upper = region_name.upper(); 
        if '_L' in name_upper: return 'Left'
        elif '_R' in name_upper: return 'Right'
        else: return 'Bilateral / Unknown'
    
    def run_full_pipeline(self, 
                         nii_path: str, 
                         save_name: str,
                         include_post_processing: bool = True, 
                         target_class_index: int = 1 
                         ) -> Dict[str, Any]:
        """
        Run the complete pipeline: Inspect -> Select -> Prepare -> Infer/Hook -> PostProcess.
        Removes redundant validation and filtering steps.
        """
        results = {}
        start_pipeline_time = time.time()
        
        try:
            # Step 1: Inspect model and select layers
            selected_layers, selected_layer_names = self.inspect_and_select_layers()
            results["selected_layers"] = selected_layers
            
            # --- Validation is implicitly done by selection ---
            # --- Dynamic filtering is removed ---
            
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
                    print("Warning: No activation/gradient files generated, skipping post-processing.")
                else:
                    post_processing_results = self.run_post_processing(
                        selected_layers, # Pass the layers selected in Step 1
                        saved_files_dict, 
                        nii_path, 
                        save_name 
                    )
                    results.update(post_processing_results)
            
            pipeline_time = time.time() - start_pipeline_time
            print(f"\n--- Pipeline Complete for {self.config.model_type.value} ({pipeline_time:.2f} seconds) ---")
            print(f"Prediction: {prediction_result}")
            
            return results
            
        except Exception as e:
            pipeline_time = time.time() - start_pipeline_time
            error_message = f"Pipeline error after {pipeline_time:.2f} seconds: {e}"
            print(f"ERROR: {error_message}")
            results["error"] = error_message
            results["traceback"] = traceback.format_exc()
            return results

# --- Backward Compatibility Function (Simplified) ---
def run_inference_and_classification(
    state: Dict[str, Any], model_config: Union[ModelConfig, str] = "papermodel"
) -> Dict[str, Any]: 
    """
    Simplified backward-compatible function for inference only.
    """
    # ... (Function content largely unchanged, but ensure paths and config name are correct) ...
    print("\n--- Node: Generic Inference & Classification (Inference Only) ---")
    subject_id = state.get("subject_id", "unknown_subject")
    model_weights = state.get("model_path") 
    nii_path = state.get("fmri_scan_path") or state.get("t1_native_path") # Support both key names
    save_name = f"{subject_id}_inference_only" 
    output_dir = state.get("output_dir", DEFAULT_OUTPUT_DIR) 

    if not nii_path: return {"error_log": state.get("error_log", []) + ["Missing input NIfTI path in state."]}
         
    try:
        pipeline = GenericInferencePipeline( model_config=model_config, model_weights_path=model_weights, output_dir=output_dir)
        pipeline.prepare_model() 
        inputs = pipeline.adapter.preprocess_data(nii_path).to(pipeline.config.device)
        with torch.no_grad(): outputs = pipeline.prepared_model(inputs)
        prediction_result, _ = pipeline.adapter.postprocess_prediction(outputs, return_logits=False) # Don't need logits here
        trace = f"Node: Generic inference complete. Prediction: {prediction_result}"
        return { "classification_result": prediction_result, "trace_log": state.get("trace_log", []) + [trace] }
    except Exception as e:
        # ... (Error handling unchanged) ...
        error_message = f"Node (Generic Inference Only) Error: {e}" # ... rest ...
        return {"error_log": state.get("error_log", []) + [error_message]}


# Export necessary components
__all__ = [
    'GenericInferencePipeline',
    'run_inference_and_classification', 
    'ModelConfig',
    'ModelFactory',
    'get_config_by_name',
    'inspect_torch_model',
    'select_visualization_layers',
    'prepare_model_with_hooks', 
    'activation_and_gradient_to_nifti',
    'normalize_native_heatmap_to_mni_accurate_masked', # Export the specific accurate function
    'resample_activation_to_atlas',
    'analyze_brain_activation',
    'visualize_gradcam_2d' # Export the 2D visualizer
]


# --- Main execution block for testing the full pipeline ---
if __name__ == "__main__":
    
    print("--- Starting GenericInferencePipeline Full Test ---")
    start_full_time = time.time()
    
    # --- [NEW] Simplified Configuration ---
    CONFIG_NAME = "papermodel" 
    INPUT_NIFTI_PATH = "/Volumes/3T-disk/fMRI/Model/sMRI_data/AD/T1_3D_MPRAGE_SAG_0003_008/T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii"
    MODEL_WEIGHTS = "model/shufflenet/fold_3_best_model.pth" 
    
    SUBJECT_ID = "test_subject_008" # Use a clean subject ID
    SAVE_NAME_PREFIX = SUBJECT_ID    # This will prefix files (e.g., "test_subject_008_01_native.nii.gz")
    TEST_OUTPUT_DIR = f"output/pipeline_run_{SUBJECT_ID}" # A clean, top-level folder for this subject
    # --- [END NEW] Configuration ---
    
    # --- Check required files ---
    print("Checking input files...")
    if not os.path.exists(INPUT_NIFTI_PATH):
        print(f"Error: Input T1 NIfTI not found at: {INPUT_NIFTI_PATH}"); exit()
    if MODEL_WEIGHTS and not os.path.exists(MODEL_WEIGHTS):
         print(f"Error: Model weights not found at: {MODEL_WEIGHTS}"); exit()
         
    try:
        test_config = get_config_by_name(CONFIG_NAME)
        print(f"Loaded config: {CONFIG_NAME}")
        if not os.path.exists(test_config.mni_template_path): raise FileNotFoundError(f"MNI template not found: {test_config.mni_template_path}")
        if not os.path.exists(test_config.atlas_path): raise FileNotFoundError(f"Atlas NIfTI not found: {test_config.atlas_path}")
        if not os.path.exists(test_config.atlas_label_path): raise FileNotFoundError(f"Atlas Label file not found: {test_config.atlas_label_path}")
    except Exception as e:
        print(f"Error loading config or checking config paths: {e}"); exit()

    # --- Instantiate and Run ---
    try:
        print("\nInstantiating GenericInferencePipeline...")
        pipeline = GenericInferencePipeline(
            model_config=CONFIG_NAME,
            model_weights_path=MODEL_WEIGHTS,
            output_dir=TEST_OUTPUT_DIR # Use new clean output dir
        )
        
        print("\nRunning full pipeline with post-processing...")
        results = pipeline.run_full_pipeline(
            nii_path=INPUT_NIFTI_PATH,
            save_name=SAVE_NAME_PREFIX, # Use new clean prefix
            include_post_processing=True, 
            target_class_index=1 # Explain AD class
        )
        
        print("\n--- Pipeline Finished ---")
        
        # --- Print Summary ---
        if "error" in results:
            print(f"Pipeline failed with error: {results['error']}")
            if "traceback" in results: print(f"\nTraceback:\n{results['traceback']}")
        else:
            print("Pipeline completed successfully!")
            print(f"Prediction Result: {results.get('prediction_result')}")
            if "selected_layers" in results: print(f"Selected Layers: {[layer['model_path'] for layer in results['selected_layers']]}")
            if "final_heatmap_paths" in results:
                 print("\nFinal Heatmap NIfTI Files (aligned to atlas):")
                 for layer, path in results["final_heatmap_paths"].items(): print(f"  - {layer}: {path}")
            if "visualization_paths" in results:
                 print("\nVisualization PNG Directories:")
                 for layer, path in results["visualization_paths"].items(): print(f"  - {layer}: {path}")
            if "activated_regions" in results:
                print("\nTop 5 Activated Regions (Sorted by Mean Activation):")
                try:
                    import pandas as pd
                    # Ensure the list is not empty before creating DataFrame
                    if results["activated_regions"]:
                        df_regions = pd.DataFrame(results["activated_regions"])
                        print(df_regions.head(5).to_string(index=False))
                    else:
                         print("  (No regions passed activation threshold or analysis failed)")
                except ImportError: print(results["activated_regions"][:5]) 
                     
    except Exception as e:
        print(f"\n--- Critical Error Running Pipeline ---")
        print(f"Error: {e}")
        traceback.print_exc()
        
    finally:
        end_full_time = time.time()
        print(f"\nTotal script execution time: {end_full_time - start_full_time:.2f} seconds.")