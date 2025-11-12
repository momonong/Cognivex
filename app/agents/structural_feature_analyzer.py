"""
Structural Feature Analyzer Agent

This agent analyzes feature importances from the ML model
and converts them into BrainRegionInfo format for downstream processing.
"""

from typing import Dict, List
from app.graph.state import AgentState, BrainRegionInfo


def analyze_feature_importance(state: AgentState) -> dict:
    """
    Analyze feature importances and identify key ROIs
    
    This agent:
    1. Retrieves feature importances from state
    2. Sorts ROIs by importance
    3. Selects Top N (default 10) most important ROIs
    4. Converts to BrainRegionInfo format
    5. Adds importance ranking
    
    Args:
        state: AgentState containing feature_importances
    
    Returns:
        Updated state dict with:
        - activated_regions: List[BrainRegionInfo] sorted by importance
        - trace_log: Updated with processing steps
    """
    print("\n" + "="*60)
    print("AGENT: Structural Feature Analyzer")
    print("="*60)
    
    feature_importances = state.get('feature_importances', {})
    
    if not feature_importances:
        error_msg = "No feature importances found in state"
        print(f"⚠️  {error_msg}")
        return {
            "activated_regions": [],
            "trace_log": state.get("trace_log", []) + [error_msg]
        }
    
    try:
        # Step 1: Sort features by importance
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"\n📊 Analyzing {len(sorted_features)} features")
        
        # Step 2: Select Top N features (default 10, but include all for completeness)
        top_n = 10
        top_features = sorted_features[:top_n]
        
        print(f"✓ Selected Top {top_n} most important features")
        
        # Step 3: Convert to BrainRegionInfo format
        activated_regions: List[BrainRegionInfo] = []
        
        for rank, (roi_name, importance) in enumerate(sorted_features, 1):
            # Determine hemisphere from ROI name
            if roi_name.endswith('_L'):
                hemisphere = "Left"
            elif roi_name.endswith('_R'):
                hemisphere = "Right"
            else:
                hemisphere = "Bilateral"
            
            # Get feature value if available
            roi_features = state.get('roi_features', {})
            feature_value = roi_features.get(roi_name)
            
            # Create BrainRegionInfo
            region_info: BrainRegionInfo = {
                "region_name": roi_name,
                "activation_score": float(importance),  # Use importance as activation score
                "hemisphere": hemisphere,
                "feature_value": float(feature_value) if feature_value is not None else None,
                "importance_rank": rank,
                "clinical_relevance": None,  # Will be filled by knowledge_reasoner
                "associated_networks": None,  # Will be filled by knowledge_reasoner
                "known_functions": None  # Will be filled by knowledge_reasoner
            }
            
            activated_regions.append(region_info)
        
        # Step 4: Display Top 10
        print(f"\n🎯 Top {top_n} Important Brain Regions:")
        for i, region in enumerate(activated_regions[:top_n], 1):
            print(f"   {i}. {region['region_name']}")
            print(f"      Importance: {region['activation_score']:.4f} ({region['activation_score']*100:.2f}%)")
            print(f"      Hemisphere: {region['hemisphere']}")
            if region['feature_value'] is not None:
                print(f"      Feature Value: {region['feature_value']:.3f}")
        
        trace_msg = f"Feature analysis complete: identified {len(activated_regions)} regions"
        print(f"\n✅ {trace_msg}")
        print("="*60 + "\n")
        
        return {
            "activated_regions": activated_regions,
            "trace_log": state.get("trace_log", []) + [trace_msg]
        }
        
    except Exception as e:
        error_msg = f"Feature analysis failed: {type(e).__name__}: {e}"
        print(f"❌ {error_msg}")
        print("="*60 + "\n")
        
        return {
            "activated_regions": [],
            "error_log": state.get("error_log", []) + [error_msg],
            "trace_log": state.get("trace_log", []) + ["Feature analysis failed"]
        }
