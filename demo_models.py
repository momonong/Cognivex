"""
Demo: CDDA Phase 4 Data Models

This script demonstrates the new MCP and A2A data models:
- MCP protocol models (ResourceMetadata, ToolMetadata, MCPAction)
- Context models (DiagnosticReport, ContextObject, etc.)
- ContextObjectBuilder for Agent A → Agent B handoff
"""

from app.core.models import (
    # MCP Models
    ResourceMetadata,
    ToolMetadata,
    MCPAction,
    MCPActionList,
    
    # Context Models
    DiagnosticReport,
    Feature,
    AnomalyStatus,
    ContextObject,
    AgentResult,
    
    # Builder
    ContextObjectBuilder,
    build_context_from_diagnostic_report
)


def demo_mcp_models():
    """Demo: MCP protocol models"""
    print("\n" + "="*80)
    print("DEMO 1: MCP Protocol Models")
    print("="*80)
    
    # Create resource metadata
    resource = ResourceMetadata(
        uri="diagnosis://sub-0005/report",
        name="Diagnostic Report",
        description="Complete diagnostic data including prediction, SHAP, UQ, anomalies"
    )
    print(f"\n[Resource Metadata]")
    print(f"  URI: {resource.uri}")
    print(f"  Name: {resource.name}")
    
    # Create tool metadata
    tool = ToolMetadata(
        name="simulate_counterfactual",
        description="What-if analysis by masking features",
        input_schema={
            "type": "object",
            "properties": {
                "subject_id": {"type": "string"},
                "features_to_mask": {"type": "array"}
            }
        }
    )
    print(f"\n[Tool Metadata]")
    print(f"  Name: {tool.name}")
    print(f"  Description: {tool.description}")
    
    # Create MCP actions
    action1 = MCPAction(type="read_resource", target="diagnosis://sub-0005/report")
    action1.mark_success({"data": "diagnostic report"})
    
    action2 = MCPAction(type="call_tool", target="simulate_counterfactual")
    action2.mark_success({"confidence_delta": -0.25})
    
    # Create action list
    action_list = MCPActionList()
    action_list.add_action(action1)
    action_list.add_action(action2)
    
    print(f"\n[MCP Action List]")
    print(f"  Total actions: {len(action_list.actions)}")
    print(f"  Resource reads: {len(action_list.get_resource_reads())}")
    print(f"  Tool calls: {len(action_list.get_tool_calls())}")
    print(f"  Successful: {len(action_list.get_successful_actions())}")


def demo_context_models():
    """Demo: Context data models"""
    print("\n" + "="*80)
    print("DEMO 2: Context Data Models")
    print("="*80)
    
    # Create feature
    feature = Feature(
        roi_name="Hippocampus_L",
        feature_name="Hippocampus_L_GM_Vol",
        feature_value=1500.0,
        z_score=-3.2,
        shap_value=0.15,
        rank=1
    )
    print(f"\n[Feature]")
    print(f"  ROI: {feature.roi_name}")
    print(f"  Z-score: {feature.z_score}")
    print(f"  SHAP: {feature.shap_value}")
    print(f"  Is anomalous: {feature.is_anomalous()}")
    
    # Create anomaly status
    anomaly_status = AnomalyStatus(
        has_anomaly=True,
        anomalous_regions=["Hippocampus_L", "Hippocampus_R"]
    )
    print(f"\n[Anomaly Status]")
    print(f"  Has anomaly: {anomaly_status.has_anomaly}")
    print(f"  Anomalous regions: {', '.join(anomaly_status.anomalous_regions)}")
    
    # Create diagnostic report
    report = DiagnosticReport(
        subject_id="sub-0005",
        prediction_result="AD",
        confidence=0.85,
        uq_score=0.75,
        top_features=[feature],
        anomaly_status=anomaly_status
    )
    print(f"\n[Diagnostic Report]")
    print(f"  Subject: {report.subject_id}")
    print(f"  Prediction: {report.prediction_result} ({report.confidence:.1%})")
    print(f"  UQ Score: {report.uq_score:.3f}")
    print(f"  Top features: {len(report.top_features)}")


def demo_context_builder():
    """Demo: ContextObjectBuilder"""
    print("\n" + "="*80)
    print("DEMO 3: ContextObjectBuilder (Agent A → Agent B Handoff)")
    print("="*80)
    
    # Create diagnostic report
    report = DiagnosticReport(
        subject_id="sub-0005",
        prediction_result="AD",
        confidence=0.85,
        uq_score=0.75,
        top_features=[],
        anomaly_status=AnomalyStatus(has_anomaly=False, anomalous_regions=[])
    )
    
    # Build context using builder pattern
    builder = ContextObjectBuilder()
    context = (builder
               .set_subject_id("sub-0005")
               .set_diagnostic_report(report)
               .add_signal("uq_score", 0.75)
               .add_signal("has_anomaly", False)
               .set_decision_rationale("Standard case - low uncertainty, no anomalies")
               .add_reasoning_step("1. Retrieved diagnostic report")
               .add_reasoning_step("2. Evaluated UQ score: 0.75 < 0.8")
               .add_reasoning_step("3. No anomalies detected")
               .add_reasoning_step("4. Proceeding to standard synthesis")
               .build())
    
    print(f"\n[ContextObject]")
    print(f"  Subject: {context.subject_id}")
    print(f"  Decision: {context.decision_rationale}")
    print(f"  Signals: {context.signals}")
    print(f"  Reasoning steps: {len(context.agent_a_reasoning)}")
    print(f"  Valid: {context.validate()}")
    
    # Show reasoning chain
    print(f"\n[Agent A Reasoning Chain]")
    for step in context.agent_a_reasoning:
        print(f"  {step}")
    
    # Serialize for Agent B
    print(f"\n[Serialization for Agent B]")
    json_str = context.serialize_for_agent_b()
    print(f"  JSON length: {len(json_str)} characters")
    print(f"  First 200 chars: {json_str[:200]}...")


def demo_convenience_builders():
    """Demo: Convenience builder functions"""
    print("\n" + "="*80)
    print("DEMO 4: Convenience Builder Functions")
    print("="*80)
    
    # Mock report dictionary (as returned by toolkit)
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
    
    # Use convenience builder
    context = build_context_from_diagnostic_report(
        subject_id="sub-0005",
        report_dict=report_dict,
        decision_rationale="Standard case",
        reasoning_chain=[
            "1. Retrieved diagnostic report",
            "2. Low uncertainty detected",
            "3. No anomalies found"
        ]
    )
    
    print(f"\n[Quick Build Result]")
    print(f"  Subject: {context.subject_id}")
    print(f"  Prediction: {context.signals['prediction']}")
    print(f"  Confidence: {context.signals['confidence']:.1%}")
    print(f"  UQ Score: {context.signals['uq_score']:.3f}")
    print(f"  Reasoning steps: {len(context.agent_a_reasoning)}")


def demo_agent_result():
    """Demo: AgentResult (final output)"""
    print("\n" + "="*80)
    print("DEMO 5: AgentResult (Final A2A Output)")
    print("="*80)
    
    # Create context object
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
        signals={"uq_score": 0.75, "has_anomaly": False},
        decision_rationale="Standard case",
        agent_a_reasoning=["Step 1", "Step 2"]
    )
    
    # Create agent result
    result = AgentResult(
        subject_id="sub-0005",
        agent_decision="STANDARD_REPORT",
        prediction="AD",
        confidence=0.85,
        uq_score=0.75,
        context_object=context,
        clinical_report="Patient shows typical AD presentation with hippocampal atrophy.",
        reasoning_chain=[
            "Agent A: Retrieved diagnostic data",
            "Agent A: Evaluated signals",
            "Agent A: Compiled context",
            "Agent B: Synthesized clinical narrative"
        ]
    )
    
    print(f"\n[AgentResult]")
    print(f"  Subject: {result.subject_id}")
    print(f"  Decision: {result.agent_decision}")
    print(f"  Prediction: {result.prediction} ({result.confidence:.1%})")
    print(f"  UQ Score: {result.uq_score:.3f}")
    print(f"\n[Clinical Report]")
    print(f"  {result.clinical_report}")
    print(f"\n[Complete Reasoning Chain]")
    for step in result.reasoning_chain:
        print(f"  {step}")


if __name__ == "__main__":
    demo_mcp_models()
    demo_context_models()
    demo_context_builder()
    demo_convenience_builders()
    demo_agent_result()
    
    print("\n" + "="*80)
    print("All demos completed successfully!")
    print("="*80 + "\n")
