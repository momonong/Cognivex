"""
Tests for Agent B - Clinical Consultant

This module tests the Agent B implementation including:
- Template-based synthesis
- LLM integration
- Anomaly-aware synthesis
- Counterfactual explanation
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


def test_agent_b_initialization():
    """Test Agent B initialization"""
    print("\n" + "="*80)
    print("TEST: Agent B Initialization")
    print("="*80)
    
    # Test with default config
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    
    assert agent_b is not None
    assert agent_b.config.model == "medgemma-27b"
    assert agent_b.config.use_llm == False
    assert agent_b.system_prompt is not None
    
    print("[OK] Agent B initialized successfully")


def test_template_synthesis():
    """Test template-based synthesis"""
    print("\n" + "="*80)
    print("TEST: Template-Based Synthesis")
    print("="*80)
    
    # Create mock ContextObject
    diagnostic_report = DiagnosticReport(
        subject_id='sub-test',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.75,
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
        subject_id='sub-test',
        diagnostic_report=diagnostic_report,
        decision_rationale="Standard case: low uncertainty, no anomalies.",
        signals={
            'uq_score': 0.75,
            'has_anomaly': False,
            'prediction': 'AD',
            'confidence': 0.85
        },
        agent_a_reasoning=["Read diagnostic report", "Evaluated signals"]
    )
    
    # Initialize Agent B (template mode)
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    
    # Synthesize report
    result = agent_b.synthesize(context_object)
    
    # Validate result
    assert 'clinical_report' in result
    assert 'reasoning_chain' in result
    assert len(result['clinical_report']) > 0
    assert len(result['reasoning_chain']) > 0
    
    # Check report contains key sections
    report = result['clinical_report']
    assert 'DIAGNOSTIC SUMMARY' in report
    assert 'KEY FINDINGS' in report
    assert 'CLINICAL INTERPRETATION' in report
    assert 'RECOMMENDATIONS' in report
    assert 'sub-test' in report
    assert 'AD' in report
    
    print("[OK] Template synthesis completed")
    print(f"[OK] Report length: {len(report)} chars")
    print(f"[OK] Reasoning steps: {len(result['reasoning_chain'])}")


def test_anomaly_synthesis():
    """Test anomaly-aware synthesis"""
    print("\n" + "="*80)
    print("TEST: Anomaly-Aware Synthesis")
    print("="*80)
    
    # Create mock ContextObject with anomalies
    diagnostic_report = DiagnosticReport(
        subject_id='sub-test-anomaly',
        prediction_result='AD',
        confidence=0.85,
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
            anomalous_regions=['Frontal_Lobe', 'Temporal_Lobe']
        )
    )
    
    # Add knowledge context
    tool_results = {
        'knowledge_context': {
            'query_regions': ['Frontal_Lobe'],
            'summary': 'Frontal lobe atrophy associated with FTD',
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
        subject_id='sub-test-anomaly',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="Anomalies detected, queried knowledge graph.",
        signals={
            'uq_score': 0.75,
            'has_anomaly': True,
            'anomalous_regions': ['Frontal_Lobe', 'Temporal_Lobe'],
            'prediction': 'AD',
            'confidence': 0.85
        },
        agent_a_reasoning=["Read diagnostic report", "Detected anomalies", "Queried knowledge"]
    )
    
    # Initialize Agent B
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    
    # Synthesize report
    result = agent_b.synthesize(context_object)
    
    # Validate result
    report = result['clinical_report']
    assert 'ANOMALY ANALYSIS' in report
    assert 'Frontal_Lobe' in report
    assert 'CLINICAL CONTEXT' in report
    
    # Check for mixed pathology detection
    assert 'MIXED PATHOLOGY' in report or 'mixed pathology' in report.lower()
    
    print("[OK] Anomaly synthesis completed")
    print(f"[OK] Report contains anomaly analysis")
    print(f"[OK] Mixed pathology detection working")


def test_counterfactual_synthesis():
    """Test counterfactual explanation"""
    print("\n" + "="*80)
    print("TEST: Counterfactual Explanation")
    print("="*80)
    
    # Create mock ContextObject with counterfactual
    diagnostic_report = DiagnosticReport(
        subject_id='sub-test-cf',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.85,
        top_features=[
            Feature(
                roi_name='Hippocampus_L',
                feature_name='Hippocampus_L_GM_Vol',
                feature_value=2500.0,
                z_score=-2.8,
                shap_value=0.15,
                rank=1
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
            'original_confidence': 0.85,
            'new_prediction': 'NC',
            'new_confidence': 0.45,
            'confidence_delta': -0.40,
            'masked_features': [
                {
                    'roi_name': 'Hippocampus_L',
                    'feature_name': 'Hippocampus_L_GM_Vol',
                    'original_value': 2500.0,
                    'masked_value': 3000.0
                }
            ]
        }
    }
    
    context_object = ContextObject(
        subject_id='sub-test-cf',
        diagnostic_report=diagnostic_report,
        tool_results=tool_results,
        decision_rationale="High UQ detected, ran counterfactual simulation.",
        signals={
            'uq_score': 0.85,
            'has_anomaly': False,
            'prediction': 'AD',
            'confidence': 0.85
        },
        agent_a_reasoning=["Read diagnostic report", "High UQ", "Ran counterfactual"]
    )
    
    # Initialize Agent B
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    
    # Synthesize report
    result = agent_b.synthesize(context_object)
    
    # Validate result
    report = result['clinical_report']
    assert 'COUNTERFACTUAL ANALYSIS' in report
    assert 'Hippocampus_L' in report
    assert 'KEY DIAGNOSTIC DRIVERS' in report  # Significant change
    
    print("[OK] Counterfactual synthesis completed")
    print(f"[OK] Report contains counterfactual analysis")
    print(f"[OK] Key driver identification working")


def test_context_formatting():
    """Test ContextObject formatting for LLM"""
    print("\n" + "="*80)
    print("TEST: Context Formatting for LLM")
    print("="*80)
    
    # Create mock ContextObject
    diagnostic_report = DiagnosticReport(
        subject_id='sub-test',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.75,
        top_features=[
            Feature(
                roi_name='Hippocampus_L',
                feature_name='Hippocampus_L_GM_Vol',
                feature_value=2500.0,
                z_score=-2.8,
                shap_value=0.15,
                rank=1
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=False,
            anomalous_regions=[]
        )
    )
    
    context_object = ContextObject(
        subject_id='sub-test',
        diagnostic_report=diagnostic_report,
        decision_rationale="Standard case",
        signals={'uq_score': 0.75, 'has_anomaly': False}
    )
    
    # Initialize Agent B
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    
    # Format context
    formatted = agent_b._format_context_for_llm(context_object)
    
    # Validate formatting
    assert formatted is not None
    assert len(formatted) > 0
    assert 'subject_id' in formatted
    assert 'prediction' in formatted
    assert 'top_features' in formatted
    
    # Should be valid JSON
    import json
    parsed = json.loads(formatted)
    assert parsed['subject_id'] == 'sub-test'
    assert parsed['prediction'] == 'AD'
    
    print("[OK] Context formatting successful")
    print(f"[OK] Formatted context length: {len(formatted)} chars")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AGENT B CONSULTANT - TEST SUITE")
    print("="*80)
    
    try:
        test_agent_b_initialization()
        test_template_synthesis()
        test_anomaly_synthesis()
        test_counterfactual_synthesis()
        test_context_formatting()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n[FAILED] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
