from typing import TypedDict, List, Dict, Any, Optional, Literal

class BrainRegionInfo(TypedDict):
    """Stores detailed information for a single brain region."""
    region_name: str
    activation_score: float
    hemisphere: str
    associated_networks: Optional[List[str]]
    known_functions: Optional[str]
    
    # === Structural MRI specific fields ===
    feature_value: Optional[float]  # Original feature value (standardized)
    importance_rank: Optional[int]  # Ranking by importance
    clinical_relevance: Optional[str]  # Clinical significance from MODEL_OVERALL.md

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
    model_name: Optional[str]  # 模型名稱 ("capsnet", "mcadnnet", etc.)
    
    # === Analysis Mode Control ===
    analysis_mode: Optional[Literal["structural", "functional"]]  # Analysis type
    ml_model_type: Optional[str]  # ML model type ("random_forest", "svm", etc.)

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
    generated_reports: Optional[Dict[str, str]]  # For functional MRI text reports
    structured_report: Optional[Dict[str, Dict[str, Any]]]  # For structural MRI JSON reports
    
    # === Structural MRI Specific Outputs ===
    roi_features: Optional[Dict[str, float]]  # ROI name -> feature value
    feature_importances: Optional[Dict[str, float]]  # ROI name -> importance
    prediction_confidence: Optional[float]  # Prediction confidence (0-1)
    feature_importance_plot_path: Optional[str]  # Path to importance chart
    roi_visualization_path: Optional[str]  # Path to brain visualization

    # === 4. System & Tracing ===
    # For logging and error handling throughout the workflow.
    error_log: List[str]
    trace_log: List[str]