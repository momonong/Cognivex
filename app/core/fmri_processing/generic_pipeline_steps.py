"""
Generic Pipeline Steps for fMRI Processing

This module provides model-agnostic pipeline steps that work with any model
through the ModelAdapter interface. This replaces the hardcoded pipeline_steps.py
with a more flexible, extensible design.
"""

import torch
import os
import json
from typing import Dict, Any, List, Tuple, Optional, Union

from app.core.fmri_processing.model_config import (
    ModelConfig, 
    ModelFactory,
    get_config_by_name
)

# Import existing pipeline components
from app.core.fmri_processing.pipelines.inspect_model import inspect_torch_model
from app.core.fmri_processing.pipelines.choose_layer import select_visualization_layers
from app.core.fmri_processing.pipelines.attach_hook import prepare_model_with_hooks as attach_hooks_to_model
from app.core.fmri_processing.pipelines.validate_layer import validate_layers_by_llm
from app.core.fmri_processing.pipelines.filter_layer import filter_layers_by_llm

from app.core.fmri_processing.pipelines.act_to_nii import activation_to_nifti
from app.core.fmri_processing.pipelines.resample import resample_activation_to_atlas
from app.core.fmri_processing.pipelines.brain_map import analyze_brain_activation
from app.core.fmri_processing.pipelines.visualize import visualize_activation_map

# Global constants - these can be overridden by model config
DEFAULT_OUTPUT_DIR = "output/langraph"
DEFAULT_SAVE_NAME = "langraph_test"
DEFAULT_VIS_DIR_PREFIX = "figures/langraph_test"
DEFAULT_NORM_TYPE = "l2"
DEFAULT_ACT_THRESHOLD_PERCENTILE = 95.0
DEFAULT_ATLAS_PATH = "data/aal3/AAL3v1_1mm.nii.gz"
DEFAULT_LABEL_PATH = "data/aal3/AAL3v1_1mm.nii.txt"
DEFAULT_VIS_THRESHOLD_PERCENTILE = 0.1

class GenericInferencePipeline:
    """
    A generic inference pipeline that can work with any model type
    through the ModelAdapter interface.
    """
    
    def __init__(self, 
                 model_config: Union[ModelConfig, str],
                 model_path: Optional[str] = None,
                 output_dir: str = DEFAULT_OUTPUT_DIR):
        """
        Initialize the pipeline with a model configuration.
        
        Args:
            model_config: Either a ModelConfig object or string name of predefined config
            model_path: Path to trained model weights (optional)
            output_dir: Directory for saving outputs
        """
        if isinstance(model_config, str):
            self.config = get_config_by_name(model_config)
        else:
            self.config = model_config
            
        self.adapter = ModelFactory.create_adapter(self.config)
        self.model_path = model_path
        self.output_dir = output_dir
        
        # Initialize model components
        self.model = None
        self.prepared_model = None
        self.selected_layers = None
        
    def inspect_model_structure(self) -> Tuple[List[Dict], List[str]]:
        """
        Inspect model structure and return selected layers.
        
        Returns:
            Tuple of (selected_layers, selected_layer_names)
        """
        print(f"\n--- Inspecting {self.config.model_type.value} Model Structure ---")
        
        # Create model if not already created
        if self.model is None:
            self.model = self.adapter.create_model()
        
        # Get layer information using existing inspect function
        # For inspect_torch_model, we need to remove the batch dimension
        inspect_input_shape = self.config.input_shape[1:]  # Remove batch dimension
        print(f"Using input shape for inspection: {inspect_input_shape}")
        
        layers = inspect_torch_model(
            self.model, 
            inspect_input_shape,
            self.config.device
        )
        
        # Use model-specific layer selection strategy
        strategy = self.adapter.get_layer_selection_strategy()
        response = select_visualization_layers(layers, strategy=strategy)
        selected_layers = json.loads(response)
        
        # Extract model paths
        selected_layer_names = [item["model_path"] for item in selected_layers]
        
        print(f"Selected {len(selected_layers)} layers for {self.config.model_type.value} model")
        
        return selected_layers, selected_layer_names
    
    def validate_layers(self, selected_layers: List[Dict], all_layers_info: List[Dict]) -> List[Dict]:
        """
        Validate that selected layers exist in the model.
        
        Args:
            selected_layers: List of layers selected by LLM
            all_layers_info: Ground-truth list of all layers from model inspection
            
        Returns:
            List of validated layers
        """
        print("\n--- Validating Selected Layers ---")
        
        # Basic existence validation
        valid_layer_paths = {layer["model_path"] for layer in all_layers_info}
        basic_validated_layers = []
        
        for layer in selected_layers:
            model_path = layer.get("model_path")
            if model_path in valid_layer_paths:
                basic_validated_layers.append(layer)
                print(f"✓ Valid: {model_path}")
            else:
                print(f"✗ Invalid: {model_path} - Layer does not exist")
        
        # Use LLM validation for additional checks
        if basic_validated_layers:
            llm_validated_results = validate_layers_by_llm(basic_validated_layers)
            
            # Map results back to original format
            validated_model_paths = {item["model_path"] for item in llm_validated_results}
            final_validated_layers = []
            
            for original_layer in basic_validated_layers:
                if original_layer["model_path"] in validated_model_paths:
                    llm_result = next(
                        (item for item in llm_validated_results 
                         if item["model_path"] == original_layer["model_path"]), 
                        None
                    )
                    updated_layer = original_layer.copy()
                    if llm_result:
                        updated_layer["reason"] = llm_result["reason"]
                    final_validated_layers.append(updated_layer)
            
            print(f"Final validation: {len(final_validated_layers)}/{len(selected_layers)} layers passed")
            return final_validated_layers
        else:
            print("No valid layers found after basic validation")
            return []
    
    def prepare_model_for_inference(self, selected_layers: List[Dict]) -> torch.nn.Module:
        """
        Prepare model with hooks and load weights.
        
        Args:
            selected_layers: List of validated layers to hook
            
        Returns:
            Prepared model ready for inference
        """
        print(f"\n--- Preparing {self.config.model_type.value} Model for Inference ---")
        
        if self.model is None:
            self.model = self.adapter.create_model()
        
        # Attach hooks to selected layers
        model_with_hooks = attach_hooks_to_model(self.model, selected_layers)
        
        # Load model weights if provided
        if self.model_path:
            print(f"Loading weights from: {self.model_path}")
            model_with_hooks.load_state_dict(
                torch.load(self.model_path, map_location=self.config.device)
            )
        
        model_with_hooks.to(self.config.device).eval()
        self.prepared_model = model_with_hooks
        
        return model_with_hooks
    
    def run_inference(self, 
                     nii_path: str, 
                     save_name: str, 
                     selected_layer_names: List[str]) -> Union[str, int, float]:
        """
        Run inference using model-specific preprocessing and postprocessing.
        
        Args:
            nii_path: Path to input NIfTI file
            save_name: Base name for saved files
            selected_layer_names: List of layer names to extract activations from
            
        Returns:
            Processed prediction result
        """
        print(f"\n--- Running {self.config.model_type.value} Inference ---")
        
        if self.prepared_model is None:
            raise ValueError("Model not prepared. Call prepare_model_for_inference first.")
        
        # Use adapter for model-specific preprocessing
        inputs = self.adapter.preprocess_data(nii_path)
        print(f"Preprocessed input shape: {inputs.shape}")
        
        # Run inference
        with torch.no_grad():
            outputs = self.prepared_model(inputs)
        
        # Use adapter for model-specific postprocessing
        prediction_result = self.adapter.postprocess_prediction(outputs)
        print(f"Prediction result: {prediction_result}")
        
        # Save activations if model has them
        if hasattr(self.prepared_model, "activations") and isinstance(self.prepared_model.activations, dict):
            os.makedirs(self.output_dir, exist_ok=True)
            
            for layer_name in selected_layer_names:
                if layer_name in self.prepared_model.activations:
                    act = self.prepared_model.activations[layer_name]
                    filename = f"{save_name}_{layer_name.replace('.', '_')}.pt"
                    save_path = os.path.join(self.output_dir, filename)
                    torch.save(act.cpu(), save_path)
                    print(f"Saved activation: {save_path}")
        
        return prediction_result
    
    def run_dynamic_filtering(self,
                             validated_layers: List[Dict],
                             save_name: str) -> List[Dict]:
        """
        Run dynamic layer filtering based on activation statistics.
        
        Args:
            validated_layers: List of validated layers
            save_name: Base name for saved files
            
        Returns:
            List of filtered layers that passed LLM evaluation
        """
        print(f"\n--- Running Dynamic Layer Filtering ---")
        
        try:
            # Use the same filtering logic as in the original system
            keep_entries = filter_layers_by_llm(
                validated_layers, 
                self.output_dir, 
                save_name, 
                delete_rejected=True
            )
            
            if not keep_entries:
                print("  Warning: No layers passed filtering. Using first validated layer as fallback.")
                keep_entries = validated_layers[:1]
            
            print(f"  Filtering complete: {len(keep_entries)}/{len(validated_layers)} layers kept")
            return keep_entries
            
        except Exception as e:
            print(f"  Error in dynamic filtering: {e}")
            print(f"  Fallback: Using all validated layers")
            return validated_layers
    
    def run_post_processing(self,
                           validated_layers: List[Dict], 
                           nii_path: str, 
                           save_name: str) -> Dict[str, Any]:
        """
        Run post-processing: activation analysis and visualization.
        
        Args:
            validated_layers: List of validated layers to process
            nii_path: Path to original NIfTI file
            save_name: Base name for saved files
            
        Returns:
            Dictionary containing post-processing results
        """
        print(f"\n--- Running Post-Processing for {len(validated_layers)} layers ---")
        
        post_results = {
            "activated_regions": [],
            "visualization_paths": [],
            "final_layers": validated_layers
        }
        
        output_prefix = os.path.join(self.output_dir, save_name)
        region_max_activations = {}
        
        for layer in validated_layers:
            layer_name = layer["model_path"]
            print(f"  Processing layer: {layer_name}")
            
            try:
                # Setup paths and convert activation to NIfTI
                safe_layer_name = layer_name.replace(".", "_")
                act_path = f"{output_prefix}_{safe_layer_name}.pt"
                nii_output = f"{output_prefix}_{safe_layer_name}.nii.gz"
                vis_dir = f"{DEFAULT_VIS_DIR_PREFIX}/{save_name}_{safe_layer_name}"
                os.makedirs(vis_dir, exist_ok=True)
                
                # Check if activation file exists
                if not os.path.exists(act_path):
                    print(f"    Warning: Activation file not found: {act_path}")
                    continue
                
                # Convert activation to NIfTI
                activation_to_nifti(
                    activation_path=act_path,
                    reference_nii_path=nii_path,
                    output_path=nii_output,
                    norm_type=DEFAULT_NORM_TYPE,
                    threshold_percentile=DEFAULT_ACT_THRESHOLD_PERCENTILE,
                )
                
                # Resample to atlas space
                resampled_path = resample_activation_to_atlas(
                    act_path=nii_output,
                    atlas_path=DEFAULT_ATLAS_PATH,
                    output_dir=vis_dir,
                )
                
                # Analyze brain activation
                df_result = analyze_brain_activation(
                    activation_path=resampled_path,
                    atlas_path=DEFAULT_ATLAS_PATH,
                    label_path=DEFAULT_LABEL_PATH,
                )
                
                # Generate visualization
                vis_output_path = visualize_activation_map(
                    activation_path=resampled_path,
                    output_path=os.path.join(vis_dir, f"map_{safe_layer_name}.png"),
                    threshold=DEFAULT_VIS_THRESHOLD_PERCENTILE,
                    title=f"Activation Map - {layer_name}"
                )
                post_results["visualization_paths"].append(vis_output_path)
                
                # Aggregate region activations
                for _, row in df_result.iterrows():
                    region_name = row['Region Name']
                    activation = row['Mean Activation']
                    hemisphere = self._parse_hemisphere(region_name)
                    
                    if (region_name not in region_max_activations or 
                        activation > region_max_activations[region_name]['activation_score']):
                        region_max_activations[region_name] = {
                            "region_name": region_name,
                            "activation_score": float(activation),
                            "hemisphere": hemisphere,
                        }
                
            except Exception as e:
                print(f"    Error processing layer {layer_name}: {e}")
                continue
        
        # Sort regions by activation score
        all_regions_info = list(region_max_activations.values())
        all_regions_info.sort(key=lambda x: x['activation_score'], reverse=True)
        post_results["activated_regions"] = all_regions_info
        
        print(f"  Post-processing complete: {len(all_regions_info)} regions, {len(post_results['visualization_paths'])} visualizations")
        
        return post_results
    
    def _parse_hemisphere(self, region_name: str) -> str:
        """
        Parse hemisphere information from region name.
        
        Args:
            region_name: Brain region name
            
        Returns:
            Hemisphere string
        """
        name_upper = region_name.upper()
        if '_L' in name_upper:
            return 'Left'
        elif '_R' in name_upper:
            return 'Right'
        else:
            return 'Bilateral / Unknown'
    
    def run_full_pipeline(self, 
                         nii_path: str, 
                         save_name: str,
                         include_post_processing: bool = False,
                         include_dynamic_filtering: bool = False) -> Dict[str, Any]:
        """
        Run the complete inference pipeline from start to finish.
        
        Args:
            nii_path: Path to input NIfTI file  
            save_name: Base name for saved files
            include_post_processing: Whether to include activation analysis and visualization
            include_dynamic_filtering: Whether to apply LLM-based dynamic layer filtering
            
        Returns:
            Dictionary containing all pipeline results
        """
        results = {}
        
        try:
            # Step 1: Inspect model and select layers
            selected_layers, selected_layer_names = self.inspect_model_structure()
            results["selected_layers"] = selected_layers
            
            # Step 2: Validate layers
            if self.model is None:
                self.model = self.adapter.create_model()
            all_layers_info = inspect_torch_model(
                self.model, 
                self.config.input_shape[1:], 
                self.config.device
            )
            validated_layers = self.validate_layers(selected_layers, all_layers_info)
            
            if not validated_layers:
                raise ValueError("No valid layers found after validation")
            
            results["validated_layers"] = validated_layers
            validated_layer_names = [layer["model_path"] for layer in validated_layers]
            
            # Step 3: Prepare model for inference
            prepared_model = self.prepare_model_for_inference(validated_layers)
            
            # Step 4: Run inference
            prediction_result = self.run_inference(
                nii_path, save_name, validated_layer_names
            )
            results["prediction_result"] = prediction_result
            
            # Step 5: Optional dynamic filtering
            final_layers = validated_layers
            if include_dynamic_filtering:
                final_layers = self.run_dynamic_filtering(
                    validated_layers, save_name
                )
                results["final_layers"] = final_layers
            
            # Step 6: Optional post-processing
            if include_post_processing:
                post_processing_results = self.run_post_processing(
                    final_layers, nii_path, save_name
                )
                results.update(post_processing_results)
            
            print(f"\n--- Pipeline Complete for {self.config.model_type.value} ---")
            print(f"Prediction: {prediction_result}")
            
            return results
            
        except Exception as e:
            error_message = f"Pipeline error: {e}"
            print(f"ERROR: {error_message}")
            results["error"] = error_message
            return results

# Convenience functions for backward compatibility
def run_inference_and_classification(state: Dict[str, Any], 
                                     model_config: Union[ModelConfig, str] = "capsnet") -> Dict[str, Any]:
    """
    Backward-compatible function for running inference.
    This can replace the original function in inference.py
    
    Note: This function only runs inference. For the complete workflow including
    dynamic filtering and post-processing, use the individual workflow nodes.
    """
    print("\n--- Node: Generic Inference & Classification ---")
    
    subject_id = state['subject_id']
    model_path = state.get('model_path')
    nii_path = state['fmri_scan_path']
    save_name = f"{subject_id}"
    
    try:
        # Create pipeline
        pipeline = GenericInferencePipeline(
            model_config=model_config,
            model_path=model_path
        )
        
        # Run only the inference part (not the full pipeline)
        # The workflow handles filtering and post-processing in separate nodes
        results = pipeline.run_full_pipeline(nii_path, save_name)
        
        if "error" in results:
            return {"error_log": state.get("error_log", []) + [results["error"]]}
        
        trace = f"Node: Generic inference complete. Prediction: {results['prediction_result']}"
        
        return {
            "classification_result": results["prediction_result"],
            "validated_layers": results["validated_layers"],
            "trace_log": state.get("trace_log", []) + [trace]
        }
        
    except Exception as e:
        error_message = f"Node (Generic Inference) Error: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}

# Export the generic pipeline for easy import
__all__ = [
    'GenericInferencePipeline',
    'run_inference_and_classification',
    'ModelConfig',
    'ModelFactory',
    'get_config_by_name',
    # Note: filter_layers_by_llm and post-processing functions are now integrated
    # into GenericInferencePipeline and available through run_full_pipeline options
]
