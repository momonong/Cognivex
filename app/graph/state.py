from typing import TypedDict, List, Dict, Any, Optional

class BrainRegionInfo(TypedDict):
    """Stores detailed information for a single brain region."""
    region_name: str
    activation_score: float
    hemisphere: str
    associated_networks: Optional[List[str]]
    known_functions: Optional[List[str]]  # <-- MODIFIED for consistency

class AgentState(TypedDict):
    """
    Defines the complete state for the fMRI analysis workflow.
    Includes inputs, intermediate results, and final outputs.
    """
    
    # === 1. Inputs ===
    subject_id: str
    fmri_scan_path: str
    model_path: Optional[str]
    model_name: Optional[str]
    llm_provider: Optional[str] # Add this to pass the provider choice through the state

    # === 2. Intermediate Data ===
    validated_layers: Optional[List[Dict[str, Any]]]
    final_layers: Optional[List[Dict[str, Any]]]
    post_processing_results: Optional[List[Dict[str, Any]]]
    clean_region_names: Optional[List[str]] 

    # === 3. Final Outputs ===
    classification_result: Optional[str]
    activated_regions: Optional[List[BrainRegionInfo]]
    visualization_paths: Optional[Dict[str, str]]  # <-- MODIFIED for robustness
    image_explanation: Optional[Dict[str, Any]]
    rag_summary: Optional[str] # This can likely be removed if final_report_json contains all info
    final_report_json: Optional[Dict[str, Any]] # <-- MODIFIED to match your dashboard output

    # === 4. System & Tracing ===
    error_log: List[str]
    trace_log: List[str]