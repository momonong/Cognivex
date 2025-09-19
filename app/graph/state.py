from typing import TypedDict, List, Dict, Any, Optional

class BrainRegionInfo(TypedDict):
    """Stores detailed information for a single brain region."""
    region_name: str
    activation_score: float
    hemisphere: str
    associated_networks: Optional[List[str]]
    known_functions: Optional[str]

class AgentState(TypedDict):
    """
    Defines the complete state for the fMRI analysis workflow.
    Includes inputs, intermediate results, and final outputs.
    """
    
    # === 1. Inputs ===
    # Data provided at the start of the workflow.
    subject_id: str
    fmri_scan_path: str
    model_path: Optional[str]

    # === 2. Intermediate Data ===
    # Data passed between internal nodes of the fMRI analysis pipeline.
    validated_layers: Optional[List[Dict[str, Any]]]
    final_layers: Optional[List[Dict[str, Any]]]
    post_processing_results: Optional[List[Dict[str, Any]]]
    clean_region_names: Optional[List[str]] 

    # === 3. Final Outputs ===
    # The primary, structured results of the entire pipeline.
    classification_result: Optional[str]
    activated_regions: Optional[List[BrainRegionInfo]]
    visualization_paths: Optional[List[str]]
    image_explanation: Optional[Dict[str, Any]]
    rag_summary: Optional[str]
    generated_reports: Optional[Dict[str, str]]

    # === 4. System & Tracing ===
    # For logging and error handling throughout the workflow.
    error_log: List[str]
    trace_log: List[str]