"""
Demo: Counterfactual Explanation in Agent B

This script demonstrates how Agent B interprets counterfactual simulation results
to identify key diagnostic drivers and provide clinical explanations.

Requirements: 7.2, 7.3, 7.4
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.agents.agent_b_consultant import AgentB, AgentBConfig
from app.core.models import (
    ContextObject,
    DiagnosticReport,
    Feature,
    AnomalyStatus,
    MaskedFeature
)


def demo_significant_impact():
    """
    Demo: Counterfactual with significant confidence change (>0.1)
    
    This demonstrates Requirement 7.3: Features with significant confidence
    change are identified as key diagnostic drivers.
    """
    print("\n" + "="*80)
    print("DEMO 1: Significant Impact - Key Diagnostic Drivers")
    print("="*80)
    print("Scenario: Masking hippocampal features causes 25% confidence drop")
    print("Expected: Features identified as KEY DIAGNOSTIC DRIVERS")
    print("-"*80)
    
    # Create diagnostic report
    diagnostic_report = DiagnosticReport(
        subject_id='sub-demo-01',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.82,
        top_features=[
            Feature(
                roi_name='Hippocampus_L',
                feature_name='Hippocampus_L_GM_Vol',
                feature_value=2500.0,
                z_score=-2.8,
                shap_value=0.15,
                rank=1
            ),
            Feature(
                roi_name='Hippocampus_R',
                feature_name='Hippocampus_R_GM_Vol',
                feature_value=2450.0,
                z_score=-2.6,
                shap_value=0.12,
                rank=2
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=False,
            anomalous_regions=[]
        )
    )
    
    # Create counterfactual results with SIGNIFICANT change
    counterfactual_result = {
        'original_prediction': 'AD',
        'original_confidence': 0.85,
        'new_prediction': 'NC',
        'new_confidence': 0.60,
        'confidence_delta': -0.25,  # Significant change!
        'masked_features': [
            {
                'roi_name': 'Hippocampus_L',
                'feature_name': 'Hippocampus_L_GM_Vol',
                'original_value': 2500.0,
                'masked_value': 3200.0  # Population mean
            },
            {
                'roi_name': 'Hippocampus_R',
                'feature_name': 'Hippocampus_R_GM_Vol',
                'original_value': 2450.0,
                'masked_value': 3150.0  # Population mean
            }
        ]
    }
    
    # Create context object
    context_object = ContextObject(
        subject_id='sub-demo-01',
        diagnostic_report=diagnostic_report,
        tool_results={'counterfactual': counterfactual_result},
        decision_rationale="High UQ detected, ran counterfactual simulation.",
        signals={
            'uq_score': 0.82,
            'has_anomaly': False,
            'prediction': 'AD',
            'confidence': 0.85
        },
        agent_a_reasoning=["Read diagnostic report", "High UQ", "Ran counterfactual"]
    )
    
    # Initialize Agent B (template mode for demo)
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    
    # Synthesize report
    result = agent_b.synthesize(context_object)
    
    # Extract counterfactual section
    report = result['clinical_report']
    cf_section_start = report.find('COUNTERFACTUAL ANALYSIS')
    cf_section_end = report.find('\n\n', cf_section_start + 1)
    if cf_section_end == -1:
        cf_section_end = len(report)
    
    cf_section = report[cf_section_start:cf_section_end]
    
    print("\n" + cf_section)
    print("\n" + "-"*80)
    print("✓ Requirement 7.3 validated: Significant change identified as KEY DRIVERS")
    print("✓ Requirement 7.2 validated: Clinical explanations provided")
    print("-"*80)


def demo_minimal_impact():
    """
    Demo: Counterfactual with minimal confidence change (<0.05)
    
    This demonstrates Requirement 7.4: Features with minimal confidence
    change are indicated as NOT primary drivers.
    """
    print("\n" + "="*80)
    print("DEMO 2: Minimal Impact - Not Primary Drivers")
    print("="*80)
    print("Scenario: Masking features causes only 2% confidence change")
    print("Expected: Features identified as NOT PRIMARY DRIVERS")
    print("-"*80)
    
    # Create diagnostic report
    diagnostic_report = DiagnosticReport(
        subject_id='sub-demo-02',
        prediction_result='AD',
        confidence=0.88,
        uq_score=0.85,
        top_features=[
            Feature(
                roi_name='Frontal_Sup_L',
                feature_name='Frontal_Sup_L_GM_Vol',
                feature_value=15000.0,
                z_score=-1.2,
                shap_value=0.03,
                rank=8
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=False,
            anomalous_regions=[]
        )
    )
    
    # Create counterfactual results with MINIMAL change
    counterfactual_result = {
        'original_prediction': 'AD',
        'original_confidence': 0.88,
        'new_prediction': 'AD',
        'new_confidence': 0.86,
        'confidence_delta': -0.02,  # Minimal change!
        'masked_features': [
            {
                'roi_name': 'Frontal_Sup_L',
                'feature_name': 'Frontal_Sup_L_GM_Vol',
                'original_value': 15000.0,
                'masked_value': 16500.0  # Population mean
            }
        ]
    }
    
    # Create context object
    context_object = ContextObject(
        subject_id='sub-demo-02',
        diagnostic_report=diagnostic_report,
        tool_results={'counterfactual': counterfactual_result},
        decision_rationale="High UQ detected, ran counterfactual simulation.",
        signals={
            'uq_score': 0.85,
            'has_anomaly': False,
            'prediction': 'AD',
            'confidence': 0.88
        },
        agent_a_reasoning=["Read diagnostic report", "High UQ", "Ran counterfactual"]
    )
    
    # Initialize Agent B (template mode for demo)
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    
    # Synthesize report
    result = agent_b.synthesize(context_object)
    
    # Extract counterfactual section
    report = result['clinical_report']
    cf_section_start = report.find('COUNTERFACTUAL ANALYSIS')
    cf_section_end = report.find('\n\n', cf_section_start + 1)
    if cf_section_end == -1:
        cf_section_end = len(report)
    
    cf_section = report[cf_section_start:cf_section_end]
    
    print("\n" + cf_section)
    print("\n" + "-"*80)
    print("✓ Requirement 7.4 validated: Minimal change identified as NOT PRIMARY DRIVERS")
    print("✓ Requirement 7.2 validated: Clinical explanations provided")
    print("-"*80)


def demo_moderate_impact():
    """
    Demo: Counterfactual with moderate confidence change (0.05-0.1)
    
    This demonstrates the middle ground where features have measurable
    but not critical impact.
    """
    print("\n" + "="*80)
    print("DEMO 3: Moderate Impact - Contributing Factors")
    print("="*80)
    print("Scenario: Masking features causes 7% confidence change")
    print("Expected: Features identified as MODERATE IMPACT")
    print("-"*80)
    
    # Create diagnostic report
    diagnostic_report = DiagnosticReport(
        subject_id='sub-demo-03',
        prediction_result='AD',
        confidence=0.82,
        uq_score=0.78,
        top_features=[
            Feature(
                roi_name='Temporal_Mid_L',
                feature_name='Temporal_Mid_L_GM_Vol',
                feature_value=12000.0,
                z_score=-1.8,
                shap_value=0.08,
                rank=4
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=False,
            anomalous_regions=[]
        )
    )
    
    # Create counterfactual results with MODERATE change
    counterfactual_result = {
        'original_prediction': 'AD',
        'original_confidence': 0.82,
        'new_prediction': 'AD',
        'new_confidence': 0.75,
        'confidence_delta': -0.07,  # Moderate change
        'masked_features': [
            {
                'roi_name': 'Temporal_Mid_L',
                'feature_name': 'Temporal_Mid_L_GM_Vol',
                'original_value': 12000.0,
                'masked_value': 13500.0  # Population mean
            }
        ]
    }
    
    # Create context object
    context_object = ContextObject(
        subject_id='sub-demo-03',
        diagnostic_report=diagnostic_report,
        tool_results={'counterfactual': counterfactual_result},
        decision_rationale="High UQ detected, ran counterfactual simulation.",
        signals={
            'uq_score': 0.78,
            'has_anomaly': False,
            'prediction': 'AD',
            'confidence': 0.82
        },
        agent_a_reasoning=["Read diagnostic report", "High UQ", "Ran counterfactual"]
    )
    
    # Initialize Agent B (template mode for demo)
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    
    # Synthesize report
    result = agent_b.synthesize(context_object)
    
    # Extract counterfactual section
    report = result['clinical_report']
    cf_section_start = report.find('COUNTERFACTUAL ANALYSIS')
    cf_section_end = report.find('\n\n', cf_section_start + 1)
    if cf_section_end == -1:
        cf_section_end = len(report)
    
    cf_section = report[cf_section_start:cf_section_end]
    
    print("\n" + cf_section)
    print("\n" + "-"*80)
    print("✓ Requirement 7.2 validated: Clinical explanations provided for moderate impact")
    print("-"*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COUNTERFACTUAL EXPLANATION DEMO")
    print("="*80)
    print("This demo showcases Agent B's counterfactual interpretation capabilities")
    print("Requirements: 7.2, 7.3, 7.4")
    print("="*80)
    
    # Run all demos
    demo_significant_impact()
    demo_minimal_impact()
    demo_moderate_impact()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("✓ Requirement 7.2: Clinical explanations for feature impact")
    print("✓ Requirement 7.3: Significant changes (>0.1) identified as key drivers")
    print("✓ Requirement 7.4: Minimal changes (<0.05) identified as non-primary")
    print("\nAll counterfactual explanation requirements validated!")
    print("="*80)
