"""
Demo: Agent B - Clinical Consultant

This script demonstrates Agent B's clinical synthesis capabilities.
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
    AnomalyStatus
)


def demo_standard_case():
    """Demo: Standard case with no anomalies"""
    print("\n" + "="*80)
    print("DEMO 1: Standard Case (No Anomalies)")
    print("="*80)
    
    # Create mock ContextObject
    diagnostic_report = DiagnosticReport(
        subject_id='sub-demo-001',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.65,
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
            ),
            Feature(
                roi_name='Entorhinal_Cortex_L',
                feature_name='Entorhinal_Cortex_L_GM_Vol',
                feature_value=1800.0,
                z_score=-2.3,
                shap_value=0.10,
                rank=3
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=False,
            anomalous_regions=[]
        )
    )
    
    context_object = ContextObject(
        subject_id='sub-demo-001',
        diagnostic_report=diagnostic_report,
        decision_rationale="Standard case: moderate uncertainty, no anomalies. Proceeding to synthesis.",
        signals={
            'uq_score': 0.65,
            'has_anomaly': False,
            'prediction': 'AD',
            'confidence': 0.85
        },
        agent_a_reasoning=[
            "[2024-01-15T10:00:00] Starting orchestration for sub-demo-001",
            "[2024-01-15T10:00:01] Read diagnostic report for sub-demo-001",
            "[2024-01-15T10:00:02] Evaluated signals: UQ=0.650, Anomaly=False",
            "[2024-01-15T10:00:03] Standard case: low uncertainty, no anomalies. Proceeding to synthesis.",
            "[2024-01-15T10:00:04] ContextObject compiled and validated"
        ]
    )
    
    # Initialize Agent B (template mode for demo)
    config = AgentBConfig(use_llm=False, verbose=True)
    agent_b = AgentB(config=config)
    
    # Synthesize report
    result = agent_b.synthesize(context_object)
    
    # Print report
    print("\n" + "-"*80)
    print("CLINICAL REPORT:")
    print("-"*80)
    print(result['clinical_report'])
    print("-"*80)


def demo_mixed_pathology():
    """Demo: Case with potential mixed pathology"""
    print("\n" + "="*80)
    print("DEMO 2: Mixed Pathology Case")
    print("="*80)
    
    # Create mock ContextObject with anomalies
    diagnostic_report = DiagnosticReport(
        subject_id='sub-demo-002',
        prediction_result='AD',
        confidence=0.88,
        uq_score=0.70,
        top_features=[
            Feature(
                roi_name='Frontal_Lobe_L',
                feature_name='Frontal_Lobe_L_GM_Vol',
                feature_value=3000.0,
                z_score=-3.5,
                shap_value=0.20,
                rank=1
            ),
            Feature(
                roi_name='Temporal_Lobe_L',
                feature_name='Temporal_Lobe_L_GM_Vol',
                feature_value=2800.0,
                z_score=-3.2,
                shap_value=0.18,
                rank=2
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=True,
            anomalous_regions=['Frontal_Lobe_L', 'Temporal_Lobe_L']
        )
    )
    
    # Add knowledge context showing non-AD associations
    tool_results = {
        'knowledge_context': {
            'query_regions': ['Frontal_Lobe_L', 'Temporal_Lobe_L'],
            'summary': 'Frontal and temporal lobe atrophy associated with FTD and vascular dementia',
            'contexts': [
                {
                    'region': 'Frontal_Lobe_L',
                    'context': {
                        'function': 'Executive function, decision making, personality',
                        'clinical_significance': 'Atrophy indicates executive dysfunction',
                        'related_conditions': ['Frontotemporal Dementia', 'Vascular Dementia', 'Progressive Supranuclear Palsy']
                    }
                },
                {
                    'region': 'Temporal_Lobe_L',
                    'context': {
                        'function': 'Memory, language, auditory processing',
                        'clinical_significance': 'Atrophy indicates memory and language deficits',
                        'related_conditions': ['Frontotemporal Dementia', 'Semantic Dementia', 'Alzheimer Disease']
                    }
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-demo-002',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="Anomalies detected in 2 regions. Retrieved clinical context from knowledge graph.",
        signals={
            'uq_score': 0.70,
            'has_anomaly': True,
            'anomalous_regions': ['Frontal_Lobe_L', 'Temporal_Lobe_L'],
            'prediction': 'AD',
            'confidence': 0.88
        },
        agent_a_reasoning=[
            "[2024-01-15T10:05:00] Starting orchestration for sub-demo-002",
            "[2024-01-15T10:05:01] Read diagnostic report for sub-demo-002",
            "[2024-01-15T10:05:02] Evaluated signals: UQ=0.700, Anomaly=True",
            "[2024-01-15T10:05:03] Anomalies detected in 2 regions. Querying knowledge graph.",
            "[2024-01-15T10:05:04] Retrieved knowledge context for Frontal_Lobe_L",
            "[2024-01-15T10:05:05] Retrieved knowledge context for Temporal_Lobe_L",
            "[2024-01-15T10:05:06] ContextObject compiled and validated"
        ]
    )
    
    # Initialize Agent B
    config = AgentBConfig(use_llm=False, verbose=True)
    agent_b = AgentB(config=config)
    
    # Synthesize report
    result = agent_b.synthesize(context_object)
    
    # Print report
    print("\n" + "-"*80)
    print("CLINICAL REPORT:")
    print("-"*80)
    print(result['clinical_report'])
    print("-"*80)


def demo_counterfactual():
    """Demo: Case with counterfactual analysis"""
    print("\n" + "="*80)
    print("DEMO 3: Counterfactual Analysis Case")
    print("="*80)
    
    # Create mock ContextObject with counterfactual
    diagnostic_report = DiagnosticReport(
        subject_id='sub-demo-003',
        prediction_result='AD',
        confidence=0.90,
        uq_score=0.85,
        top_features=[
            Feature(
                roi_name='Hippocampus_L',
                feature_name='Hippocampus_L_GM_Vol',
                feature_value=2300.0,
                z_score=-3.2,
                shap_value=0.18,
                rank=1
            ),
            Feature(
                roi_name='Hippocampus_R',
                feature_name='Hippocampus_R_GM_Vol',
                feature_value=2250.0,
                z_score=-3.0,
                shap_value=0.16,
                rank=2
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=False,
            anomalous_regions=[]
        )
    )
    
    # Add counterfactual results
    tool_results = {
        'counterfactual': {
            'original_prediction': 'AD',
            'original_confidence': 0.90,
            'new_prediction': 'NC',
            'new_confidence': 0.35,
            'confidence_delta': -0.55,
            'masked_features': [
                {
                    'roi_name': 'Hippocampus_L',
                    'feature_name': 'Hippocampus_L_GM_Vol',
                    'original_value': 2300.0,
                    'masked_value': 3000.0
                },
                {
                    'roi_name': 'Hippocampus_R',
                    'feature_name': 'Hippocampus_R_GM_Vol',
                    'original_value': 2250.0,
                    'masked_value': 2950.0
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-demo-003',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="High uncertainty detected (UQ=0.85). Invoked counterfactual simulation to identify key diagnostic drivers.",
        signals={
            'uq_score': 0.85,
            'has_anomaly': False,
            'prediction': 'AD',
            'confidence': 0.90
        },
        agent_a_reasoning=[
            "[2024-01-15T10:10:00] Starting orchestration for sub-demo-003",
            "[2024-01-15T10:10:01] Read diagnostic report for sub-demo-003",
            "[2024-01-15T10:10:02] Evaluated signals: UQ=0.850, Anomaly=False",
            "[2024-01-15T10:10:03] High UQ detected (0.850 > 0.800). Triggering counterfactual simulation.",
            "[2024-01-15T10:10:04] Simulated counterfactual: masked 2 features",
            "[2024-01-15T10:10:05] ContextObject compiled and validated"
        ]
    )
    
    # Initialize Agent B
    config = AgentBConfig(use_llm=False, verbose=True)
    agent_b = AgentB(config=config)
    
    # Synthesize report
    result = agent_b.synthesize(context_object)
    
    # Print report
    print("\n" + "-"*80)
    print("CLINICAL REPORT:")
    print("-"*80)
    print(result['clinical_report'])
    print("-"*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AGENT B - CLINICAL CONSULTANT DEMO")
    print("="*80)
    print("\nThis demo shows Agent B's clinical synthesis capabilities:")
    print("1. Standard case with no anomalies")
    print("2. Mixed pathology case with anomaly detection")
    print("3. Counterfactual analysis case with key driver identification")
    
    try:
        demo_standard_case()
        input("\nPress Enter to continue to next demo...")
        
        demo_mixed_pathology()
        input("\nPress Enter to continue to next demo...")
        
        demo_counterfactual()
        
        print("\n" + "="*80)
        print("DEMO COMPLETE")
        print("="*80)
        print("\n[SUCCESS] Agent B successfully synthesized clinical reports")
        print("[SUCCESS] Anomaly detection and mixed pathology flagging working")
        print("[SUCCESS] Counterfactual interpretation and key driver identification working")
        
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
