"""
Tests for Anomaly-Aware Synthesis (Task 4.3)

This module tests the specific requirements for anomaly-aware synthesis:
- Requirement 6.1: Flag potential mixed pathology
- Requirement 6.2: Explain discrepancies
- Requirement 6.3: List disease associations
- Requirement 6.4: Highlight SHAP-condition mismatches
- Requirement 6.5: Recommend additional clinical correlation for multiple pathologies
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.agent_b_consultant import AgentB, AgentBConfig
from app.core.models import (
    ContextObject,
    DiagnosticReport,
    Feature,
    AnomalyStatus
)


def test_requirement_6_1_mixed_pathology_flagging():
    """
    Test Requirement 6.1: Flag potential mixed pathology when model predicts AD 
    with high confidence AND anomalous regions are associated with non-AD conditions
    """
    print("\n" + "="*80)
    print("TEST: Requirement 6.1 - Mixed Pathology Flagging")
    print("="*80)
    
    # Create scenario: High confidence AD prediction with non-AD associated regions
    diagnostic_report = DiagnosticReport(
        subject_id='sub-6.1',
        prediction_result='AD',
        confidence=0.90,  # High confidence
        uq_score=0.75,
        top_features=[
            Feature(
                roi_name='Frontal_Lobe',
                feature_name='Frontal_Lobe_GM_Vol',
                feature_value=3000.0,
                z_score=-3.5,
                shap_value=0.20,
                rank=1
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=True,
            anomalous_regions=['Frontal_Lobe']
        )
    )
    
    # Knowledge context with non-AD conditions
    tool_results = {
        'knowledge_context': {
            'query_regions': ['Frontal_Lobe'],
            'summary': 'Frontal lobe atrophy',
            'contexts': [
                {
                    'region': 'Frontal_Lobe',
                    'context': {
                        'function': 'Executive function',
                        'clinical_significance': 'Atrophy indicates cognitive decline',
                        'related_conditions': ['Frontotemporal Dementia', 'Vascular Dementia']
                    }
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-6.1',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="High confidence AD with anomalies",
        signals={
            'uq_score': 0.75,
            'has_anomaly': True,
            'anomalous_regions': ['Frontal_Lobe'],
            'prediction': 'AD',
            'confidence': 0.90
        },
        agent_a_reasoning=[]
    )
    
    # Synthesize report
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    result = agent_b.synthesize(context_object)
    
    report = result['clinical_report']
    
    # Verify mixed pathology is flagged
    assert 'MIXED PATHOLOGY' in report or 'mixed pathology' in report.lower(), \
        "Report should flag potential mixed pathology"
    
    # Verify it mentions the discrepancy
    assert 'Frontotemporal Dementia' in report or 'FTD' in report or 'Vascular' in report, \
        "Report should mention non-AD conditions"
    
    print("[OK] Requirement 6.1: Mixed pathology flagged correctly")
    print(f"[OK] Report contains mixed pathology warning")


def test_requirement_6_2_explain_discrepancies():
    """
    Test Requirement 6.2: Explain discrepancies between model prediction 
    and knowledge context using medical reasoning
    """
    print("\n" + "="*80)
    print("TEST: Requirement 6.2 - Explain Discrepancies")
    print("="*80)
    
    diagnostic_report = DiagnosticReport(
        subject_id='sub-6.2',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.75,
        top_features=[
            Feature(
                roi_name='Basal_Ganglia',
                feature_name='Basal_Ganglia_GM_Vol',
                feature_value=2000.0,
                z_score=-3.0,
                shap_value=0.18,
                rank=1
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=True,
            anomalous_regions=['Basal_Ganglia']
        )
    )
    
    tool_results = {
        'knowledge_context': {
            'query_regions': ['Basal_Ganglia'],
            'contexts': [
                {
                    'region': 'Basal_Ganglia',
                    'context': {
                        'function': 'Motor control',
                        'clinical_significance': 'Atrophy in movement disorders',
                        'related_conditions': ['Parkinson Disease', 'Huntington Disease']
                    }
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-6.2',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="Discrepancy detected",
        signals={
            'uq_score': 0.75,
            'has_anomaly': True,
            'anomalous_regions': ['Basal_Ganglia'],
            'prediction': 'AD',
            'confidence': 0.85
        },
        agent_a_reasoning=[]
    )
    
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    result = agent_b.synthesize(context_object)
    
    report = result['clinical_report']
    
    # Verify explanation is present
    assert 'discrepancy' in report.lower() or 'differ' in report.lower(), \
        "Report should explain the discrepancy"
    
    # Verify medical reasoning is provided
    assert 'atypical' in report.lower() or 'mixed' in report.lower() or 'co-existing' in report.lower(), \
        "Report should provide medical reasoning for discrepancy"
    
    print("[OK] Requirement 6.2: Discrepancies explained with medical reasoning")


def test_requirement_6_3_list_disease_associations():
    """
    Test Requirement 6.3: List disease associations for anomalous regions
    """
    print("\n" + "="*80)
    print("TEST: Requirement 6.3 - List Disease Associations")
    print("="*80)
    
    diagnostic_report = DiagnosticReport(
        subject_id='sub-6.3',
        prediction_result='AD',
        confidence=0.80,
        uq_score=0.75,
        top_features=[
            Feature(
                roi_name='Temporal_Lobe',
                feature_name='Temporal_Lobe_GM_Vol',
                feature_value=2800.0,
                z_score=-2.5,
                shap_value=0.12,
                rank=1
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=True,
            anomalous_regions=['Temporal_Lobe', 'Parietal_Lobe']
        )
    )
    
    tool_results = {
        'knowledge_context': {
            'query_regions': ['Temporal_Lobe', 'Parietal_Lobe'],
            'contexts': [
                {
                    'region': 'Temporal_Lobe',
                    'context': {
                        'function': 'Memory and language',
                        'clinical_significance': 'Atrophy in dementia',
                        'related_conditions': ['Alzheimer Disease', 'Semantic Dementia', 'Temporal Lobe Epilepsy']
                    }
                },
                {
                    'region': 'Parietal_Lobe',
                    'context': {
                        'function': 'Spatial processing',
                        'clinical_significance': 'Atrophy in cognitive decline',
                        'related_conditions': ['Alzheimer Disease', 'Posterior Cortical Atrophy']
                    }
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-6.3',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="Multiple anomalies",
        signals={
            'uq_score': 0.75,
            'has_anomaly': True,
            'anomalous_regions': ['Temporal_Lobe', 'Parietal_Lobe'],
            'prediction': 'AD',
            'confidence': 0.80
        },
        agent_a_reasoning=[]
    )
    
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    result = agent_b.synthesize(context_object)
    
    report = result['clinical_report']
    
    # Verify disease associations are listed
    assert 'DISEASE ASSOCIATIONS' in report, \
        "Report should have a disease associations section"
    
    # Verify specific conditions are mentioned
    assert 'Temporal_Lobe' in report and 'Parietal_Lobe' in report, \
        "Report should list the anomalous regions"
    
    # Verify at least some conditions are mentioned
    condition_count = sum([
        'Semantic Dementia' in report,
        'Epilepsy' in report,
        'Posterior Cortical Atrophy' in report
    ])
    assert condition_count > 0, \
        "Report should list disease associations"
    
    print("[OK] Requirement 6.3: Disease associations listed correctly")


def test_requirement_6_4_shap_condition_mismatch():
    """
    Test Requirement 6.4: Highlight when leading SHAP feature is associated 
    with a different condition than predicted
    """
    print("\n" + "="*80)
    print("TEST: Requirement 6.4 - SHAP-Condition Mismatch")
    print("="*80)
    
    diagnostic_report = DiagnosticReport(
        subject_id='sub-6.4',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.75,
        top_features=[
            Feature(
                roi_name='Cerebellum',
                feature_name='Cerebellum_GM_Vol',
                feature_value=1500.0,
                z_score=-3.2,
                shap_value=0.25,  # Leading feature
                rank=1
            ),
            Feature(
                roi_name='Hippocampus_L',
                feature_name='Hippocampus_L_GM_Vol',
                feature_value=2500.0,
                z_score=-2.5,
                shap_value=0.10,
                rank=2
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=True,
            anomalous_regions=['Cerebellum']
        )
    )
    
    tool_results = {
        'knowledge_context': {
            'query_regions': ['Cerebellum'],
            'contexts': [
                {
                    'region': 'Cerebellum',
                    'context': {
                        'function': 'Motor coordination',
                        'clinical_significance': 'Atrophy in cerebellar disorders',
                        'related_conditions': ['Spinocerebellar Ataxia', 'Multiple System Atrophy']
                    }
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-6.4',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="Leading feature mismatch",
        signals={
            'uq_score': 0.75,
            'has_anomaly': True,
            'anomalous_regions': ['Cerebellum'],
            'prediction': 'AD',
            'confidence': 0.85
        },
        agent_a_reasoning=[]
    )
    
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    result = agent_b.synthesize(context_object)
    
    report = result['clinical_report']
    
    # Verify SHAP mismatch is highlighted
    assert 'SHAP-CONDITION MISMATCHES' in report or 'SHAP' in report, \
        "Report should highlight SHAP-condition mismatches"
    
    # Verify the leading feature is mentioned
    assert 'Cerebellum' in report, \
        "Report should mention the leading feature"
    
    # Verify the mismatch is explained
    assert 'differ' in report.lower() or 'mismatch' in report.lower() or 'mixed pathology' in report.lower(), \
        "Report should explain the mismatch"
    
    print("[OK] Requirement 6.4: SHAP-condition mismatch highlighted")


def test_requirement_6_5_multiple_pathology_recommendations():
    """
    Test Requirement 6.5: Recommend additional clinical correlation 
    when multiple pathologies are suggested
    """
    print("\n" + "="*80)
    print("TEST: Requirement 6.5 - Multiple Pathology Recommendations")
    print("="*80)
    
    diagnostic_report = DiagnosticReport(
        subject_id='sub-6.5',
        prediction_result='AD',
        confidence=0.88,
        uq_score=0.75,
        top_features=[
            Feature(
                roi_name='Frontal_Lobe',
                feature_name='Frontal_Lobe_GM_Vol',
                feature_value=3000.0,
                z_score=-3.5,
                shap_value=0.20,
                rank=1
            ),
            Feature(
                roi_name='White_Matter',
                feature_name='White_Matter_Vol',
                feature_value=400000.0,
                z_score=-2.8,
                shap_value=0.15,
                rank=2
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=True,
            anomalous_regions=['Frontal_Lobe', 'White_Matter']
        )
    )
    
    tool_results = {
        'knowledge_context': {
            'query_regions': ['Frontal_Lobe', 'White_Matter'],
            'contexts': [
                {
                    'region': 'Frontal_Lobe',
                    'context': {
                        'function': 'Executive function',
                        'clinical_significance': 'Atrophy in FTD',
                        'related_conditions': ['Frontotemporal Dementia', 'Behavioral Variant FTD']
                    }
                },
                {
                    'region': 'White_Matter',
                    'context': {
                        'function': 'Neural connectivity',
                        'clinical_significance': 'Hyperintensities in vascular disease',
                        'related_conditions': ['Vascular Dementia', 'Small Vessel Disease']
                    }
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-6.5',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="Multiple pathologies suggested",
        signals={
            'uq_score': 0.75,
            'has_anomaly': True,
            'anomalous_regions': ['Frontal_Lobe', 'White_Matter'],
            'prediction': 'AD',
            'confidence': 0.88
        },
        agent_a_reasoning=[]
    )
    
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    result = agent_b.synthesize(context_object)
    
    report = result['clinical_report']
    
    # Verify recommendations section exists
    assert 'RECOMMENDATIONS' in report, \
        "Report should have recommendations section"
    
    # Verify additional clinical correlation is recommended
    assert 'clinical correlation' in report.lower(), \
        "Report should recommend additional clinical correlation"
    
    # Verify comprehensive workup is recommended for mixed pathology
    assert 'comprehensive' in report.lower() or 'workup' in report.lower(), \
        "Report should recommend comprehensive workup"
    
    # Verify specific tests are mentioned
    test_mentions = sum([
        'vascular' in report.lower(),
        'CSF' in report or 'biomarker' in report.lower(),
        'PET' in report or 'imaging' in report.lower()
    ])
    assert test_mentions >= 2, \
        "Report should recommend specific diagnostic tests"
    
    print("[OK] Requirement 6.5: Multiple pathology recommendations provided")


def test_integration_all_requirements():
    """
    Integration test: Verify all requirements work together
    """
    print("\n" + "="*80)
    print("TEST: Integration - All Requirements Together")
    print("="*80)
    
    # Complex scenario with all features
    diagnostic_report = DiagnosticReport(
        subject_id='sub-integration',
        prediction_result='AD',
        confidence=0.92,  # High confidence
        uq_score=0.78,
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
                roi_name='Basal_Ganglia',
                feature_name='Basal_Ganglia_GM_Vol',
                feature_value=2000.0,
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
            anomalous_regions=['Frontal_Lobe', 'Basal_Ganglia']
        )
    )
    
    tool_results = {
        'knowledge_context': {
            'query_regions': ['Frontal_Lobe', 'Basal_Ganglia'],
            'contexts': [
                {
                    'region': 'Frontal_Lobe',
                    'context': {
                        'function': 'Executive function',
                        'clinical_significance': 'Atrophy in FTD',
                        'related_conditions': ['Frontotemporal Dementia', 'Progressive Supranuclear Palsy']
                    }
                },
                {
                    'region': 'Basal_Ganglia',
                    'context': {
                        'function': 'Motor control',
                        'clinical_significance': 'Atrophy in movement disorders',
                        'related_conditions': ['Parkinson Disease', 'Huntington Disease']
                    }
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-integration',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="Complex case with multiple anomalies",
        signals={
            'uq_score': 0.78,
            'has_anomaly': True,
            'anomalous_regions': ['Frontal_Lobe', 'Basal_Ganglia'],
            'prediction': 'AD',
            'confidence': 0.92
        },
        agent_a_reasoning=[]
    )
    
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    result = agent_b.synthesize(context_object)
    
    report = result['clinical_report']
    
    # Verify all key sections are present
    assert 'ANOMALY ANALYSIS' in report
    assert 'DISEASE ASSOCIATIONS' in report
    assert 'MIXED PATHOLOGY' in report or 'mixed pathology' in report.lower()
    assert 'SHAP-CONDITION MISMATCHES' in report or 'SHAP' in report
    assert 'RECOMMENDATIONS' in report
    assert 'clinical correlation' in report.lower()
    
    # Verify reasoning chain captures the complexity
    reasoning = result['reasoning_chain']
    assert len(reasoning) > 0
    
    print("[OK] Integration test: All requirements working together")
    print(f"[OK] Report sections: ANOMALY, DISEASE, MIXED PATHOLOGY, SHAP, RECOMMENDATIONS")
    print(f"[OK] Reasoning chain: {len(reasoning)} steps")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ANOMALY-AWARE SYNTHESIS - REQUIREMENTS TEST SUITE")
    print("Task 4.3 - Requirements 6.1, 6.2, 6.3, 6.4, 6.5")
    print("="*80)
    
    try:
        test_requirement_6_1_mixed_pathology_flagging()
        test_requirement_6_2_explain_discrepancies()
        test_requirement_6_3_list_disease_associations()
        test_requirement_6_4_shap_condition_mismatch()
        test_requirement_6_5_multiple_pathology_recommendations()
        test_integration_all_requirements()
        
        print("\n" + "="*80)
        print("ALL REQUIREMENTS TESTS PASSED")
        print("="*80)
        print("\nTask 4.3 Implementation Complete:")
        print("✓ Requirement 6.1: Mixed pathology flagging")
        print("✓ Requirement 6.2: Discrepancy explanation")
        print("✓ Requirement 6.3: Disease association listing")
        print("✓ Requirement 6.4: SHAP-condition mismatch highlighting")
        print("✓ Requirement 6.5: Multiple pathology recommendations")
        
    except AssertionError as e:
        print(f"\n[FAILED] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
