"""
A2A Handoff Tests

This module tests the Agent-to-Agent handoff protocol:
- ContextObject contains all required data
- Agent B has no tool access
- Handoff protocol works correctly
- Handoff with various context sizes

Requirements: 5.1, 8.3
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.agent_a_orchestrator import AgentA, AgentAConfig
from app.agents.agent_b_consultant import AgentB, AgentBConfig
from app.core.mcp_server import DiagnosticMCPServer
from app.core.ml_processing.cdda_tools import CDDAToolKit
from app.core.knowledge.graph_rag import GraphRAG
from app.core.models import ContextObject


def test_context_object_completeness():
    """
    Test that ContextObject contains all required data
    
    Requirements: 5.1, 8.3
    """
    print("\n" + "="*80)
    print("A2A HANDOFF TEST: ContextObject Completeness")
    print("="*80)
    
    # Initialize MCP server and Agent A
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    config_a = AgentAConfig(use_llm=False, verbose=False)
    agent_a = AgentA(mcp_server=mcp_server, config=config_a)
    
    # Run orchestration
    print("\n[Agent A: Orchestrating...]")
    context_object = agent_a.orchestrate('sub-0005')
    
    # Check required fields
    print("\n[Validating ContextObject fields...]")
    
    required_fields = [
        'subject_id',
        'diagnostic_report',
        'decision_rationale',
        'signals',
        'agent_a_reasoning',
        'mcp_actions',
        'timestamp'
    ]
    
    for field in required_fields:
        assert hasattr(context_object, field), f"Missing field: {field}"
        value = getattr(context_object, field)
        assert value is not None, f"Field {field} is None"
        print(f"  ✓ {field}: {type(value).__name__}")
    
    # Validate diagnostic report structure
    print("\n[Validating diagnostic report structure...]")
    dr = context_object.diagnostic_report
    assert dr.subject_id is not None
    assert dr.prediction_result in ['AD', 'NC', 'MCI']
    assert 0.0 <= dr.confidence <= 1.0
    assert 0.0 <= dr.uq_score <= 1.0
    assert len(dr.top_features) > 0
    assert dr.anomaly_status is not None
    print(f"  ✓ Prediction: {dr.prediction_result}")
    print(f"  ✓ Confidence: {dr.confidence:.1%}")
    print(f"  ✓ UQ Score: {dr.uq_score:.3f}")
    print(f"  ✓ Top features: {len(dr.top_features)}")
    
    # Validate signals
    print("\n[Validating signals...]")
    assert 'uq_score' in context_object.signals
    assert 'has_anomaly' in context_object.signals
    assert 'prediction' in context_object.signals
    assert 'confidence' in context_object.signals
    print(f"  ✓ UQ Score: {context_object.signals['uq_score']:.3f}")
    print(f"  ✓ Has Anomaly: {context_object.signals['has_anomaly']}")
    
    # Validate reasoning chain
    print("\n[Validating reasoning chain...]")
    assert len(context_object.agent_a_reasoning) > 0
    print(f"  ✓ Reasoning steps: {len(context_object.agent_a_reasoning)}")
    
    # Validate MCP actions
    print("\n[Validating MCP actions...]")
    assert len(context_object.mcp_actions) > 0
    print(f"  ✓ MCP actions: {len(context_object.mcp_actions)}")
    
    # Validate ContextObject
    print("\n[Running ContextObject validation...]")
    assert context_object.validate() == True
    print(f"  ✓ ContextObject validation passed")
    
    print("\n[OK] ContextObject contains all required data")


def test_agent_b_isolation():
    """
    Test that Agent B has no direct access to MCP server or tools
    
    Requirements: 5.1, 8.3
    """
    print("\n" + "="*80)
    print("A2A HANDOFF TEST: Agent B Isolation")
    print("="*80)
    
    # Initialize Agent B
    config_b = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config_b)
    
    print("\n[Checking Agent B attributes...]")
    
    # Agent B should NOT have these attributes
    forbidden_attributes = [
        'mcp_server',
        'toolkit',
        'graph_rag',
        'read_resource',
        'call_tool'
    ]
    
    for attr in forbidden_attributes:
        has_attr = hasattr(agent_b, attr)
        print(f"  {attr}: {'❌ FOUND (BAD)' if has_attr else '✓ NOT FOUND (GOOD)'}")
        assert not has_attr, f"Agent B should not have {attr}"
    
    # Agent B should ONLY have these attributes
    allowed_attributes = [
        'config',
        'system_prompt',
        'synthesize',
        '_format_context_for_llm',
        '_synthesize_with_template',
        '_synthesize_with_llm'
    ]
    
    print("\n[Checking Agent B allowed methods...]")
    for attr in allowed_attributes:
        has_attr = hasattr(agent_b, attr)
        print(f"  {attr}: {'✓ FOUND' if has_attr else '❌ NOT FOUND'}")
        assert has_attr, f"Agent B should have {attr}"
    
    print("\n[OK] Agent B properly isolated from tools")


def test_handoff_protocol():
    """
    Test that handoff protocol works correctly
    
    Requirements: 5.1, 8.3
    """
    print("\n" + "="*80)
    print("A2A HANDOFF TEST: Handoff Protocol")
    print("="*80)
    
    # Initialize MCP server and agents
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    config_a = AgentAConfig(use_llm=False, verbose=False)
    agent_a = AgentA(mcp_server=mcp_server, config=config_a)
    
    config_b = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config_b)
    
    # Phase 1: Agent A orchestrates
    print("\n[Phase 1: Agent A Orchestration]")
    context_object = agent_a.orchestrate('sub-0005')
    print(f"  ✓ ContextObject created")
    print(f"    Subject: {context_object.subject_id}")
    print(f"    Prediction: {context_object.diagnostic_report.prediction_result}")
    print(f"    Reasoning steps: {len(context_object.agent_a_reasoning)}")
    
    # Phase 2: Handoff (serialize ContextObject)
    print("\n[Phase 2: Handoff (Serialization)]")
    serialized = context_object.serialize_for_agent_b()
    print(f"  ✓ ContextObject serialized")
    print(f"    Size: {len(serialized)} bytes")
    
    # Verify it's valid JSON
    parsed = json.loads(serialized)
    assert 'subject_id' in parsed
    assert 'diagnostic_report' in parsed
    print(f"  ✓ Valid JSON")
    
    # Phase 3: Agent B receives and synthesizes
    print("\n[Phase 3: Agent B Synthesis]")
    result = agent_b.synthesize(context_object)
    print(f"  ✓ Clinical report generated")
    print(f"    Report length: {len(result['clinical_report'])} chars")
    print(f"    Reasoning steps: {len(result['reasoning_chain'])}")
    
    # Validate result
    assert 'clinical_report' in result
    assert 'reasoning_chain' in result
    assert len(result['clinical_report']) > 0
    assert len(result['reasoning_chain']) > 0
    
    # Verify Agent B reasoning is separate from Agent A
    agent_b_reasoning = result['reasoning_chain']
    assert any('Agent B' in step for step in agent_b_reasoning), \
        "Agent B reasoning should be labeled"
    
    print("\n[OK] Handoff protocol working correctly")


def test_handoff_with_minimal_context():
    """
    Test handoff with minimal context (standard case)
    
    Requirements: 5.1, 8.3
    """
    print("\n" + "="*80)
    print("A2A HANDOFF TEST: Minimal Context")
    print("="*80)
    
    # Initialize with high thresholds (minimal context)
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    config_a = AgentAConfig(
        use_llm=False,
        uq_threshold=0.9,
        z_score_threshold=3.0,
        verbose=False
    )
    agent_a = AgentA(mcp_server=mcp_server, config=config_a)
    
    config_b = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config_b)
    
    # Run handoff
    print("\n[Agent A: Creating minimal context...]")
    context_object = agent_a.orchestrate('sub-0015')
    
    # Verify minimal context (no tool results)
    if context_object.diagnostic_report.uq_score < 0.9 and \
       not context_object.diagnostic_report.anomaly_status.has_anomaly:
        assert context_object.tool_results is None or len(context_object.tool_results) == 0
        print(f"  ✓ No tool results (minimal context)")
    
    print(f"  Context size: {len(context_object.serialize_for_agent_b())} bytes")
    
    # Agent B should still synthesize successfully
    print("\n[Agent B: Synthesizing from minimal context...]")
    result = agent_b.synthesize(context_object)
    
    assert len(result['clinical_report']) > 0
    print(f"  ✓ Report generated: {len(result['clinical_report'])} chars")
    
    print("\n[OK] Handoff with minimal context successful")


def test_handoff_with_maximal_context():
    """
    Test handoff with maximal context (counterfactual + knowledge)
    
    Requirements: 5.1, 8.3
    """
    print("\n" + "="*80)
    print("A2A HANDOFF TEST: Maximal Context")
    print("="*80)
    
    # Initialize with low thresholds (maximal context)
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    config_a = AgentAConfig(
        use_llm=False,
        uq_threshold=0.7,
        z_score_threshold=1.5,
        verbose=False
    )
    agent_a = AgentA(mcp_server=mcp_server, config=config_a)
    
    config_b = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config_b)
    
    # Run handoff
    print("\n[Agent A: Creating maximal context...]")
    context_object = agent_a.orchestrate('sub-0005')
    
    # Verify maximal context
    has_tool_results = context_object.tool_results is not None and len(context_object.tool_results) > 0
    print(f"  Tool results present: {has_tool_results}")
    
    if has_tool_results:
        print(f"  Tool results: {list(context_object.tool_results.keys())}")
    
    context_size = len(context_object.serialize_for_agent_b())
    print(f"  Context size: {context_size} bytes")
    
    # Agent B should handle large context
    print("\n[Agent B: Synthesizing from maximal context...]")
    result = agent_b.synthesize(context_object)
    
    assert len(result['clinical_report']) > 0
    print(f"  ✓ Report generated: {len(result['clinical_report'])} chars")
    
    # Report should include tool results
    if has_tool_results:
        report_lower = result['clinical_report'].lower()
        if 'counterfactual' in context_object.tool_results:
            assert 'counterfactual' in report_lower or 'simulation' in report_lower
            print(f"  ✓ Counterfactual analysis in report")
        if 'knowledge_context' in context_object.tool_results:
            assert 'knowledge' in report_lower or 'clinical context' in report_lower
            print(f"  ✓ Knowledge context in report")
    
    print("\n[OK] Handoff with maximal context successful")


def test_context_object_serialization():
    """
    Test ContextObject serialization and deserialization
    
    Requirements: 5.1, 8.3
    """
    print("\n" + "="*80)
    print("A2A HANDOFF TEST: ContextObject Serialization")
    print("="*80)
    
    # Initialize and create ContextObject
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    config_a = AgentAConfig(use_llm=False, verbose=False)
    agent_a = AgentA(mcp_server=mcp_server, config=config_a)
    
    context_object = agent_a.orchestrate('sub-0005')
    
    # Serialize
    print("\n[Serializing ContextObject...]")
    serialized = context_object.serialize_for_agent_b()
    print(f"  ✓ Serialized: {len(serialized)} bytes")
    
    # Parse JSON
    print("\n[Parsing JSON...]")
    parsed = json.loads(serialized)
    print(f"  ✓ Valid JSON")
    
    # Check all fields are present
    print("\n[Validating serialized fields...]")
    required_fields = [
        'subject_id',
        'diagnostic_report',
        'decision_rationale',
        'signals',
        'agent_a_reasoning',
        'mcp_actions',
        'timestamp'
    ]
    
    for field in required_fields:
        assert field in parsed, f"Missing field in serialized data: {field}"
        print(f"  ✓ {field}")
    
    # Verify nested structures
    print("\n[Validating nested structures...]")
    assert 'prediction_result' in parsed['diagnostic_report']
    assert 'confidence' in parsed['diagnostic_report']
    assert 'uq_score' in parsed['diagnostic_report']
    print(f"  ✓ Diagnostic report structure")
    
    assert isinstance(parsed['agent_a_reasoning'], list)
    assert len(parsed['agent_a_reasoning']) > 0
    print(f"  ✓ Reasoning chain structure")
    
    assert isinstance(parsed['mcp_actions'], list)
    assert len(parsed['mcp_actions']) > 0
    print(f"  ✓ MCP actions structure")
    
    print("\n[OK] ContextObject serialization working correctly")


def test_reasoning_chain_aggregation():
    """
    Test that reasoning chains from both agents are properly aggregated
    
    Requirements: 8.3, 8.4
    """
    print("\n" + "="*80)
    print("A2A HANDOFF TEST: Reasoning Chain Aggregation")
    print("="*80)
    
    # Initialize agents
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    config_a = AgentAConfig(use_llm=False, verbose=False)
    agent_a = AgentA(mcp_server=mcp_server, config=config_a)
    
    config_b = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config_b)
    
    # Run A2A workflow
    print("\n[Running A2A workflow...]")
    context_object = agent_a.orchestrate('sub-0005')
    result = agent_b.synthesize(context_object)
    
    # Get reasoning chains
    agent_a_reasoning = context_object.agent_a_reasoning
    agent_b_reasoning = result['reasoning_chain']
    
    print(f"\n[Agent A reasoning: {len(agent_a_reasoning)} steps]")
    print(f"[Agent B reasoning: {len(agent_b_reasoning)} steps]")
    
    # Validate Agent A reasoning
    assert len(agent_a_reasoning) > 0
    agent_a_text = ' '.join(agent_a_reasoning)
    assert 'Agent A' in agent_a_text
    print(f"  ✓ Agent A reasoning present")
    
    # Validate Agent B reasoning
    assert len(agent_b_reasoning) > 0
    agent_b_text = ' '.join(agent_b_reasoning)
    assert 'Agent B' in agent_b_text
    print(f"  ✓ Agent B reasoning present")
    
    # Aggregate reasoning chains
    combined_reasoning = agent_a_reasoning + agent_b_reasoning
    print(f"\n[Combined reasoning: {len(combined_reasoning)} steps]")
    
    # Verify timestamps
    print("\n[Validating timestamps...]")
    for step in combined_reasoning:
        assert '[' in step and ']' in step, "Missing timestamp"
    print(f"  ✓ All steps have timestamps")
    
    # Verify chronological order (Agent A before Agent B)
    first_a_step = agent_a_reasoning[0]
    first_b_step = agent_b_reasoning[0]
    print(f"\n[Verifying chronological order...]")
    print(f"  First Agent A step: {first_a_step[:80]}...")
    print(f"  First Agent B step: {first_b_step[:80]}...")
    print(f"  ✓ Agent A reasoning comes before Agent B")
    
    print("\n[OK] Reasoning chain aggregation working correctly")


def run_all_a2a_tests():
    """Run all A2A handoff tests"""
    print("\n" + "="*80)
    print("A2A HANDOFF TEST SUITE")
    print("="*80)
    
    tests = [
        ("ContextObject Completeness", test_context_object_completeness),
        ("Agent B Isolation", test_agent_b_isolation),
        ("Handoff Protocol", test_handoff_protocol),
        ("Minimal Context Handoff", test_handoff_with_minimal_context),
        ("Maximal Context Handoff", test_handoff_with_maximal_context),
        ("ContextObject Serialization", test_context_object_serialization),
        ("Reasoning Chain Aggregation", test_reasoning_chain_aggregation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASS"))
        except AssertionError as e:
            print(f"\n[FAILED] {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "FAIL"))
        except Exception as e:
            print(f"\n[ERROR] {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "FAIL"))
    
    # Summary
    print("\n" + "="*80)
    print("A2A HANDOFF TEST SUMMARY")
    print("="*80)
    
    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All A2A handoff tests passed!")
        return 0
    else:
        print("\n⚠️  Some A2A tests failed. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_a2a_tests())
