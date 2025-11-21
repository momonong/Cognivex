"""
Tests for CDDA data models

This module tests the MCP and context data models to ensure:
- Proper serialization/deserialization
- Validation logic works correctly
- Builder pattern functions as expected
"""

import pytest
from datetime import datetime

from app.core.models import (
    # MCP Models
    ResourceMetadata,
    ToolMetadata,
    MCPAction,
    MCPActionList,
    
    # Context Models
    ContextObject,
    DiagnosticReport,
    Feature,
    AnomalyStatus,
    CounterfactualResult,
    MaskedFeature,
    KnowledgeContext,
    RegionContext,
    AgentResult,
    
    # Builder
    ContextObjectBuilder,
    build_context_from_diagnostic_report,
    build_context_with_counterfactual,
    build_context_with_knowledge
)


# ============================================================================
# MCP Model Tests
# ============================================================================

def test_resource_metadata_creation():
    """Test ResourceMetadata creation and serialization"""
    resource = ResourceMetadata(
        uri="diagnosis://sub-0005/report",
        name="Diagnostic Report",
        description="Complete diagnostic data",
        mime_type="application/json"
    )
    
    assert resource.uri == "diagnosis://sub-0005/report"
    assert resource.name == "Diagnostic Report"
    
    # Test serialization
    data = resource.to_dict()
    assert data['uri'] == "diagnosis://sub-0005/report"
    assert data['mime_type'] == "application/json"


def test_tool_metadata_creation():
    """Test ToolMetadata creation and serialization"""
    tool = ToolMetadata(
        name="simulate_counterfactual",
        description="What-if analysis",
        input_schema={
            "type": "object",
            "properties": {
                "subject_id": {"type": "string"}
            }
        }
    )
    
    assert tool.name == "simulate_counterfactual"
    assert "subject_id" in tool.input_schema["properties"]
    
    # Test serialization
    data = tool.to_dict()
    assert data['name'] == "simulate_counterfactual"


def test_mcp_action_lifecycle():
    """Test MCPAction creation and status updates"""
    action = MCPAction(
        type="read_resource",
        target="diagnosis://sub-0005/report"
    )
    
    assert action.type == "read_resource"
    assert action.status == "pending"
    
    # Mark as success
    action.mark_success({"data": "test"})
    assert action.status == "success"
    assert action.result == {"data": "test"}
    
    # Test error marking
    action2 = MCPAction(type="call_tool", target="test_tool")
    action2.mark_error("Tool failed")
    assert action2.status == "error"
    assert action2.error == "Tool failed"


def test_mcp_action_list():
    """Test MCPActionList utility methods"""
    action_list = MCPActionList()
    
    # Add actions
    action1 = MCPAction(type="read_resource", target="test1")
    action1.mark_success({"data": "test1"})
    action_list.add_action(action1)
    
    action2 = MCPAction(type="call_tool", target="test2")
    action2.mark_error("Failed")
    action_list.add_action(action2)
    
    # Test queries
    assert len(action_list.actions) == 2
    assert len(action_list.get_successful_actions()) == 1
    assert len(action_list.get_failed_actions()) == 1
    assert len(action_list.get_resource_reads()) == 1
    assert len(action_list.get_tool_calls()) == 1


# ============================================================================
# Context Model Tests
# ============================================================================

def test_feature_creation():
    """Test Feature creation and anomaly detection"""
    feature = Feature(
        roi_name="Hippocampus_L",
        feature_name="Hippocampus_L_GM_Vol",
        feature_value=1500.0,
        z_score=-3.2,
        shap_value=0.15,
        rank=1
    )
    
    assert feature.roi_name == "Hippocampus_L"
    assert feature.is_anomalous(threshold=2.5) == True
    assert feature.is_anomalous(threshold=4.0) == False


def test_anomaly_status_creation():
    """Test AnomalyStatus creation"""
    status = AnomalyStatus(
        has_anomaly=True,
        anomalous_regions=["Hippocampus_L", "Hippocampus_R"],
        anomaly_type="statistical_outlier"
    )
    
    assert status.has_anomaly == True
    assert len(status.anomalous_regions) == 2


def test_diagnostic_report_creation():
    """Test DiagnosticReport creation and serialization"""
    features = [
        Feature(
            roi_name="Hippocampus_L",
            feature_name="Hippocampus_L_GM_Vol",
            feature_value=1500.0,
            z_score=-3.2,
            shap_value=0.15,
            rank=1
        )
    ]
    
    anomaly_status = AnomalyStatus(
        has_anomaly=True,
        anomalous_regions=["Hippocampus_L"]
    )
    
    report = DiagnosticReport(
        subject_id="sub-0005",
        prediction_result="AD",
        confidence=0.85,
        uq_score=0.75,
        top_features=features,
        anomaly_status=anomaly_status
    )
    
    assert report.subject_id == "sub-0005"
    assert report.prediction_result == "AD"
    assert len(report.top_features) == 1
    
    # Test serialization
    data = report.to_dict()
    assert data['subject_id'] == "sub-0005"
    assert data['confidence'] == 0.85


def test_counterfactual_result_creation():
    """Test CounterfactualResult creation"""
    masked_features = [
        MaskedFeature(
            roi_name="Hippocampus_L",
            feature_name="Hippocampus_L_GM_Vol",
            original_value=1500.0,
            masked_value=2000.0
        )
    ]
    
    result = CounterfactualResult(
        subject_id="sub-0005",
        original_prediction="AD",
        original_confidence=0.85,
        new_prediction="NC",
        new_confidence=0.55,
        confidence_delta=-0.30,
        masked_features=masked_features,
        interpretation="Significant impact"
    )
    
    assert result.confidence_delta == -0.30
    assert len(result.masked_features) == 1


def test_knowledge_context_creation():
    """Test KnowledgeContext creation"""
    contexts = [
        RegionContext(
            region_name="Hippocampus_L",
            full_name="Left Hippocampus",
            function="Memory formation",
            clinical_significance="Critical for AD diagnosis",
            related_conditions=["Alzheimer's Disease"],
            is_ad_hotspot=True
        )
    ]
    
    knowledge = KnowledgeContext(
        query_regions=["Hippocampus_L"],
        contexts=contexts,
        summary="Hippocampus shows atrophy"
    )
    
    assert len(knowledge.contexts) == 1
    assert knowledge.contexts[0].is_ad_hotspot == True


# ============================================================================
# ContextObject Tests
# ============================================================================

def test_context_object_validation():
    """Test ContextObject validation"""
    # Create minimal valid context
    report = DiagnosticReport(
        subject_id="sub-0005",
        prediction_result="AD",
        confidence=0.85,
        uq_score=0.75,
        top_features=[],
        anomaly_status=AnomalyStatus(has_anomaly=False, anomalous_regions=[])
    )
    
    context = ContextObject(
        subject_id="sub-0005",
        diagnostic_report=report,
        signals={"uq_score": 0.75}
    )
    
    assert context.validate() == True
    
    # Test invalid context (missing subject_id)
    invalid_context = ContextObject(
        subject_id="",
        diagnostic_report=report,
        signals={}
    )
    
    assert invalid_context.validate() == False


def test_context_object_serialization():
    """Test ContextObject serialization for Agent B"""
    report = DiagnosticReport(
        subject_id="sub-0005",
        prediction_result="AD",
        confidence=0.85,
        uq_score=0.75,
        top_features=[],
        anomaly_status=AnomalyStatus(has_anomaly=False, anomalous_regions=[])
    )
    
    context = ContextObject(
        subject_id="sub-0005",
        diagnostic_report=report,
        signals={"uq_score": 0.75},
        decision_rationale="Standard case"
    )
    
    # Test JSON serialization
    json_str = context.serialize_for_agent_b()
    assert "sub-0005" in json_str
    assert "Standard case" in json_str


# ============================================================================
# ContextObjectBuilder Tests
# ============================================================================

def test_context_builder_basic():
    """Test ContextObjectBuilder basic functionality"""
    builder = ContextObjectBuilder()
    
    report = DiagnosticReport(
        subject_id="sub-0005",
        prediction_result="AD",
        confidence=0.85,
        uq_score=0.75,
        top_features=[],
        anomaly_status=AnomalyStatus(has_anomaly=False, anomalous_regions=[])
    )
    
    context = (builder
               .set_subject_id("sub-0005")
               .set_diagnostic_report(report)
               .add_signal("uq_score", 0.75)
               .add_reasoning_step("Step 1: Got report")
               .build())
    
    assert context.subject_id == "sub-0005"
    assert len(context.agent_a_reasoning) == 1
    assert context.validate() == True


def test_context_builder_validation():
    """Test ContextObjectBuilder validation"""
    builder = ContextObjectBuilder()
    
    # Try to build without required fields
    with pytest.raises(ValueError, match="subject_id"):
        builder.build()
    
    # Add subject_id but still missing report
    builder.set_subject_id("sub-0005")
    with pytest.raises(ValueError, match="diagnostic_report"):
        builder.build()


def test_context_builder_from_dict():
    """Test building context from dictionary data"""
    report_dict = {
        'subject_id': 'sub-0005',
        'prediction_result': 'AD',
        'confidence': 0.85,
        'uq_score': 0.75,
        'top_features': [],
        'anomaly_status': {
            'has_anomaly': False,
            'anomalous_regions': []
        }
    }
    
    builder = ContextObjectBuilder()
    context = (builder
               .set_subject_id("sub-0005")
               .set_diagnostic_report_from_dict(report_dict)
               .add_signal("uq_score", 0.75)
               .build())
    
    assert context.diagnostic_report.prediction_result == "AD"


def test_convenience_builders():
    """Test convenience builder functions"""
    report_dict = {
        'subject_id': 'sub-0005',
        'prediction_result': 'AD',
        'confidence': 0.85,
        'uq_score': 0.75,
        'top_features': [],
        'anomaly_status': {
            'has_anomaly': False,
            'anomalous_regions': []
        }
    }
    
    # Test standard builder
    context = build_context_from_diagnostic_report(
        subject_id="sub-0005",
        report_dict=report_dict,
        decision_rationale="Standard case"
    )
    
    assert context.subject_id == "sub-0005"
    assert context.decision_rationale == "Standard case"


# ============================================================================
# AgentResult Tests
# ============================================================================

def test_agent_result_creation():
    """Test AgentResult creation"""
    report = DiagnosticReport(
        subject_id="sub-0005",
        prediction_result="AD",
        confidence=0.85,
        uq_score=0.75,
        top_features=[],
        anomaly_status=AnomalyStatus(has_anomaly=False, anomalous_regions=[])
    )
    
    context = ContextObject(
        subject_id="sub-0005",
        diagnostic_report=report,
        signals={"uq_score": 0.75}
    )
    
    result = AgentResult(
        subject_id="sub-0005",
        agent_decision="STANDARD_REPORT",
        prediction="AD",
        confidence=0.85,
        uq_score=0.75,
        context_object=context,
        clinical_report="Patient shows signs of AD",
        reasoning_chain=["Step 1", "Step 2"]
    )
    
    assert result.subject_id == "sub-0005"
    assert result.agent_decision == "STANDARD_REPORT"
    assert len(result.reasoning_chain) == 2


def test_agent_result_serialization():
    """Test AgentResult serialization"""
    report = DiagnosticReport(
        subject_id="sub-0005",
        prediction_result="AD",
        confidence=0.85,
        uq_score=0.75,
        top_features=[],
        anomaly_status=AnomalyStatus(has_anomaly=False, anomalous_regions=[])
    )
    
    context = ContextObject(
        subject_id="sub-0005",
        diagnostic_report=report,
        signals={"uq_score": 0.75}
    )
    
    result = AgentResult(
        subject_id="sub-0005",
        agent_decision="STANDARD_REPORT",
        prediction="AD",
        confidence=0.85,
        uq_score=0.75,
        context_object=context,
        clinical_report="Patient shows signs of AD",
        reasoning_chain=["Step 1"]
    )
    
    data = result.to_dict()
    assert data['subject_id'] == "sub-0005"
    assert 'context_object' in data
    assert 'clinical_report' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
