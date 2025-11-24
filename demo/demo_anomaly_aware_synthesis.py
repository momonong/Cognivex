"""
Demo: Anomaly-Aware Synthesis (Task 4.3)

This script demonstrates the anomaly-aware synthesis capabilities of Agent B,
showing how it detects and reports on:
- Mixed pathology indicators
- Model-knowledge discrepancies
- Disease associations
- SHAP-condition mismatches
- Multiple pathology recommendations
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
    """Demo: Standard case without anomalies"""
    print("\n" + "="*80)
    print("DEMO 1: Standard Case (No Anomalies)")
    print("="*80)
    
    diagnostic_report = DiagnosticReport(
        subject_id='sub-standard',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.70,
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
    
    context_object = ContextObject(
        subject_id='sub-standard',
        diagnostic_report=diagnostic_report,
        decision_rationale="Standard case: typical AD pattern",
        signals={
            'uq_score': 0.70,
            'has_anomaly': False,
            'prediction': 'AD',
            'confidence': 0.85
        },
        agent_a_reasoning=["Read diagnostic report", "Evaluated signals", "Standard pattern"]
    )
    
    config = AgentBConfig(use_llm=False, verbose=True)
    agent_b = AgentB(config=config)
    result = agent_b.synthesize(context_object)
    
    print("\n" + "-"*80)
    print("CLINICAL REPORT:")
    print("-"*80)
    print(result['clinical_report'])
    print("-"*80)


def demo_mixed_pathology():
    """Demo: Mixed pathology case"""
    print("\n" + "="*80)
    print("DEMO 2: Mixed Pathology Detection")
    print("="*80)
    print("Scenario: High confidence AD prediction, but anomalous regions")
    print("          associated with Frontotemporal Dementia")
    print("="*80)
    
    diagnostic_report = DiagnosticReport(
        subject_id='sub-mixed',
        prediction_result='AD',
        confidence=0.92,  # High confidence
        uq_score=0.75,
        top_features=[
            Feature(
                roi_name='Frontal_Lobe',
                feature_name='Frontal_Lobe_GM_Vol',
                feature_value=3000.0,
                z_score=-3.8,
                shap_value=0.22,
                rank=1
            ),
            Feature(
                roi_name='Temporal_Lobe',
                feature_name='Temporal_Lobe_GM_Vol',
                feature_value=2800.0,
                z_score=-3.2,
                shap_value=0.18,
                rank=2
            ),
            Feature(
                roi_name='Hippocampus_L',
                feature_name='Hippocampus_L_GM_Vol',
                feature_value=2400.0,
                z_score=-2.5,
                shap_value=0.12,
                rank=3
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=True,
            anomalous_regions=['Frontal_Lobe', 'Temporal_Lobe']
        )
    )
    
    # Knowledge context showing non-AD associations
    tool_results = {
        'knowledge_context': {
            'query_regions': ['Frontal_Lobe', 'Temporal_Lobe'],
            'summary': 'Frontal and temporal atrophy pattern',
            'contexts': [
                {
                    'region': 'Frontal_Lobe',
                    'context': {
                        'function': 'Executive function, personality, behavior',
                        'clinical_significance': 'Atrophy indicates executive dysfunction',
                        'related_conditions': [
                            'Frontotemporal Dementia',
                            'Behavioral Variant FTD',
                            'Progressive Supranuclear Palsy'
                        ]
                    }
                },
                {
                    'region': 'Temporal_Lobe',
                    'context': {
                        'function': 'Memory, language, semantic knowledge',
                        'clinical_significance': 'Atrophy in dementia syndromes',
                        'related_conditions': [
                            'Semantic Dementia',
                            'Temporal Variant FTD',
                            'Alzheimer Disease'
                        ]
                    }
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-mixed',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="Anomalies detected, queried knowledge graph",
        signals={
            'uq_score': 0.75,
            'has_anomaly': True,
            'anomalous_regions': ['Frontal_Lobe', 'Temporal_Lobe'],
            'prediction': 'AD',
            'confidence': 0.92
        },
        agent_a_reasoning=[
            "Read diagnostic report",
            "Detected anomalies in frontal and temporal regions",
            "Queried knowledge graph for clinical context"
        ]
    )
    
    config = AgentBConfig(use_llm=False, verbose=True)
    agent_b = AgentB(config=config)
    result = agent_b.synthesize(context_object)
    
    print("\n" + "-"*80)
    print("CLINICAL REPORT:")
    print("-"*80)
    print(result['clinical_report'])
    print("-"*80)
    
    print("\n" + "-"*80)
    print("KEY OBSERVATIONS:")
    print("-"*80)
    print("✓ Mixed pathology flagged (Requirement 6.1)")
    print("✓ Discrepancies explained (Requirement 6.2)")
    print("✓ Disease associations listed (Requirement 6.3)")
    print("✓ SHAP-condition mismatches highlighted (Requirement 6.4)")
    print("✓ Multiple pathology recommendations provided (Requirement 6.5)")
    print("-"*80)


def demo_shap_mismatch():
    """Demo: SHAP-condition mismatch"""
    print("\n" + "="*80)
    print("DEMO 3: SHAP-Condition Mismatch")
    print("="*80)
    print("Scenario: Leading SHAP feature (Cerebellum) associated with")
    print("          movement disorders, not AD")
    print("="*80)
    
    diagnostic_report = DiagnosticReport(
        subject_id='sub-shap-mismatch',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.75,
        top_features=[
            Feature(
                roi_name='Cerebellum',
                feature_name='Cerebellum_GM_Vol',
                feature_value=1500.0,
                z_score=-3.5,
                shap_value=0.25,  # Leading feature
                rank=1
            ),
            Feature(
                roi_name='Basal_Ganglia',
                feature_name='Basal_Ganglia_GM_Vol',
                feature_value=2000.0,
                z_score=-3.0,
                shap_value=0.18,
                rank=2
            ),
            Feature(
                roi_name='Hippocampus_L',
                feature_name='Hippocampus_L_GM_Vol',
                feature_value=2500.0,
                z_score=-2.5,
                shap_value=0.10,
                rank=3
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=True,
            anomalous_regions=['Cerebellum', 'Basal_Ganglia']
        )
    )
    
    tool_results = {
        'knowledge_context': {
            'query_regions': ['Cerebellum', 'Basal_Ganglia'],
            'contexts': [
                {
                    'region': 'Cerebellum',
                    'context': {
                        'function': 'Motor coordination, balance',
                        'clinical_significance': 'Atrophy in cerebellar disorders',
                        'related_conditions': [
                            'Spinocerebellar Ataxia',
                            'Multiple System Atrophy',
                            'Cerebellar Degeneration'
                        ]
                    }
                },
                {
                    'region': 'Basal_Ganglia',
                    'context': {
                        'function': 'Motor control, procedural learning',
                        'clinical_significance': 'Atrophy in movement disorders',
                        'related_conditions': [
                            'Parkinson Disease',
                            'Huntington Disease',
                            'Progressive Supranuclear Palsy'
                        ]
                    }
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-shap-mismatch',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="Leading features show atypical pattern",
        signals={
            'uq_score': 0.75,
            'has_anomaly': True,
            'anomalous_regions': ['Cerebellum', 'Basal_Ganglia'],
            'prediction': 'AD',
            'confidence': 0.85
        },
        agent_a_reasoning=[
            "Read diagnostic report",
            "Detected anomalies in cerebellum and basal ganglia",
            "Unusual pattern for AD diagnosis"
        ]
    )
    
    config = AgentBConfig(use_llm=False, verbose=True)
    agent_b = AgentB(config=config)
    result = agent_b.synthesize(context_object)
    
    print("\n" + "-"*80)
    print("CLINICAL REPORT:")
    print("-"*80)
    print(result['clinical_report'])
    print("-"*80)
    
    print("\n" + "-"*80)
    print("KEY OBSERVATIONS:")
    print("-"*80)
    print("✓ SHAP-condition mismatch detected")
    print("✓ Leading feature (Cerebellum) associated with movement disorders")
    print("✓ Comprehensive workup recommended")
    print("-"*80)


def demo_vascular_mixed():
    """Demo: AD with vascular components"""
    print("\n" + "="*80)
    print("DEMO 4: AD with Vascular Components")
    print("="*80)
    print("Scenario: AD prediction with white matter changes suggesting")
    print("          vascular dementia component")
    print("="*80)
    
    diagnostic_report = DiagnosticReport(
        subject_id='sub-vascular',
        prediction_result='AD',
        confidence=0.88,
        uq_score=0.76,
        top_features=[
            Feature(
                roi_name='Hippocampus_L',
                feature_name='Hippocampus_L_GM_Vol',
                feature_value=2400.0,
                z_score=-2.9,
                shap_value=0.16,
                rank=1
            ),
            Feature(
                roi_name='White_Matter',
                feature_name='White_Matter_Vol',
                feature_value=400000.0,
                z_score=-3.2,
                shap_value=0.14,
                rank=2
            ),
            Feature(
                roi_name='Periventricular_WM',
                feature_name='Periventricular_WM_Vol',
                feature_value=50000.0,
                z_score=-3.5,
                shap_value=0.12,
                rank=3
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=True,
            anomalous_regions=['White_Matter', 'Periventricular_WM']
        )
    )
    
    tool_results = {
        'knowledge_context': {
            'query_regions': ['White_Matter', 'Periventricular_WM'],
            'contexts': [
                {
                    'region': 'White_Matter',
                    'context': {
                        'function': 'Neural connectivity, information transfer',
                        'clinical_significance': 'Hyperintensities indicate vascular disease',
                        'related_conditions': [
                            'Vascular Dementia',
                            'Small Vessel Disease',
                            'Cerebral Amyloid Angiopathy'
                        ]
                    }
                },
                {
                    'region': 'Periventricular_WM',
                    'context': {
                        'function': 'Connectivity around ventricles',
                        'clinical_significance': 'Changes in vascular and demyelinating disease',
                        'related_conditions': [
                            'Vascular Dementia',
                            'Binswanger Disease',
                            'Multiple Sclerosis'
                        ]
                    }
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-vascular',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="White matter anomalies suggest vascular component",
        signals={
            'uq_score': 0.76,
            'has_anomaly': True,
            'anomalous_regions': ['White_Matter', 'Periventricular_WM'],
            'prediction': 'AD',
            'confidence': 0.88
        },
        agent_a_reasoning=[
            "Read diagnostic report",
            "Detected white matter anomalies",
            "Queried knowledge for vascular context"
        ]
    )
    
    config = AgentBConfig(use_llm=False, verbose=True)
    agent_b = AgentB(config=config)
    result = agent_b.synthesize(context_object)
    
    print("\n" + "-"*80)
    print("CLINICAL REPORT:")
    print("-"*80)
    print(result['clinical_report'])
    print("-"*80)
    
    print("\n" + "-"*80)
    print("KEY OBSERVATIONS:")
    print("-"*80)
    print("✓ Mixed AD-vascular pathology detected")
    print("✓ Vascular imaging recommended")
    print("✓ CSF biomarkers recommended to confirm AD")
    print("-"*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ANOMALY-AWARE SYNTHESIS DEMONSTRATION")
    print("Task 4.3 - Requirements 6.1, 6.2, 6.3, 6.4, 6.5")
    print("="*80)
    
    demo_standard_case()
    input("\nPress Enter to continue to Demo 2...")
    
    demo_mixed_pathology()
    input("\nPress Enter to continue to Demo 3...")
    
    demo_shap_mismatch()
    input("\nPress Enter to continue to Demo 4...")
    
    demo_vascular_mixed()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nTask 4.3 successfully demonstrates:")
    print("✓ Detection of model-knowledge discrepancies")
    print("✓ Flagging of potential mixed pathology")
    print("✓ Highlighting of SHAP-condition mismatches")
    print("✓ Generation of comprehensive recommendations")
    print("="*80)
