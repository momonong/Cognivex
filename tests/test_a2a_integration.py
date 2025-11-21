"""
Integration Tests for A2A Agent System

This module tests the complete Agent A → Agent B handoff workflow.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.agent_a_orchestrator import AgentA, AgentAConfig
from app.agents.agent_b_consultant import AgentB, AgentBConfig
from app.core.mcp_server import DiagnosticMCPServer
from app.core.ml_processing.cdda_tools import CDDAToolKit
from app.core.knowledge.graph_rag import GraphRAG


def test_a2a_handoff_standard_case():
    """Test A2A handoff for standard case (low UQ, no anomalies)"""
    print("\n" + "="*80)
    print("TEST: A2A Handoff - Standard Case")
    print("="*80)
    
    # Initialize MCP server
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # Initialize Agent A (rule-based for testing)
    config_a = AgentAConfig(use_llm=False, verbose=False)
    agent_a = AgentA(mcp_server=mcp_server, config=config_a)
    
    # Initialize Agent B (template-based for testing)
    config_b = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config_b)
    
    # Agent A: Orchestrate
    print("[Agent A] Orchestrating analysis...")
    context_object = agent_a.orchestrate('sub-0005')
    
    # Validate ContextObject
    assert context_object is not None
    assert context_object.subject_id == 'sub-0005'
    assert context_object.diagnostic_report is not None
    assert context_object.validate()
    
    print(f"[Agent A] ContextObject created:")
    print(f"  - Subject: {context_object.subject_id}")
    print(f"  - Prediction: {context_object.diagnostic_report.prediction_result}")
    print(f"  - Confidence: {context_object.diagnostic_report.confidence:.1%}")
    print(f"  - UQ Score: {context_object.diagnostic_report.uq_score:.3f}")
    print(f"  - Reasoning steps: {len(context_object.agent_a_reasoning)}")
    
    # Agent B: Synthesize
    print("\n[Agent B] Synthesizing clinical report...")
    result = agent_b.synthesize(context_object)
    
    # Validate result
    assert 'clinical_report' in result
    assert 'reasoning_chain' in result
    assert len(result['clinical_report']) > 0
    assert len(result['reasoning_chain']) > 0
    
    print(f"[Agent B] Clinical report generated:")
    print(f"  - Report length: {len(result['clinical_report'])} chars")
    print(f"  - Reasoning steps: {len(result['reasoning_chain'])}")
    
    # Validate report content
    report = result['clinical_report']
    assert 'sub-0005' in report
    assert 'DIAGNOSTIC SUMMARY' in report
    assert 'KEY FINDINGS' in report
    assert 'RECOMMENDATIONS' in report
    
    print("\n[OK] A2A handoff successful for standard case")
    return context_object, result


def test_a2a_handoff_high_uq():
    """Test A2A handoff for high UQ case (triggers counterfactual)"""
    print("\n" + "="*80)
    print("TEST: A2A Handoff - High UQ Case")
    print("="*80)
    
    # Initialize MCP server
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # Initialize Agent A with lower UQ threshold to trigger counterfactual
    config_a = AgentAConfig(use_llm=False, uq_threshold=0.5, verbose=False)
    agent_a = AgentA(mcp_server=mcp_server, config=config_a)
    
    # Initialize Agent B
    config_b = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config_b)
    
    # Agent A: Orchestrate
    print("[Agent A] Orchestrating analysis...")
    context_object = agent_a.orchestrate('sub-0005')
    
    # Check if counterfactual was triggered
    has_counterfactual = (
        context_object.tool_results is not None and
        'counterfactual' in context_object.tool_results
    )
    
    print(f"[Agent A] Counterfactual triggered: {has_counterfactual}")
    
    # Agent B: Synthesize
    print("\n[Agent B] Synthesizing clinical report...")
    result = agent_b.synthesize(context_object)
    
    # Validate result
    assert 'clinical_report' in result
    report = result['clinical_report']
    
    # If counterfactual was triggered, report should mention it
    if has_counterfactual:
        assert 'COUNTERFACTUAL' in report or 'counterfactual' in report.lower()
        print("[OK] Counterfactual analysis included in report")
    
    print("\n[OK] A2A handoff successful for high UQ case")
    return context_object, result


def test_context_object_isolation():
    """Test that Agent B has no direct access to MCP server"""
    print("\n" + "="*80)
    print("TEST: Agent B Isolation (No MCP Access)")
    print("="*80)
    
    # Initialize Agent B
    config_b = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config_b)
    
    # Verify Agent B has no MCP server attribute
    assert not hasattr(agent_b, 'mcp_server')
    assert not hasattr(agent_b, 'toolkit')
    assert not hasattr(agent_b, 'graph_rag')
    
    print("[OK] Agent B has no direct access to MCP server or tools")
    print("[OK] Agent B can only work with ContextObject")


def test_reasoning_chain_aggregation():
    """Test that reasoning chains from both agents are preserved"""
    print("\n" + "="*80)
    print("TEST: Reasoning Chain Aggregation")
    print("="*80)
    
    # Initialize MCP server
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # Initialize agents
    config_a = AgentAConfig(use_llm=False, verbose=False)
    agent_a = AgentA(mcp_server=mcp_server, config=config_a)
    
    config_b = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config_b)
    
    # Run A2A workflow
    context_object = agent_a.orchestrate('sub-0005')
    result = agent_b.synthesize(context_object)
    
    # Validate reasoning chains
    agent_a_reasoning = context_object.agent_a_reasoning
    agent_b_reasoning = result['reasoning_chain']
    
    assert len(agent_a_reasoning) > 0
    assert len(agent_b_reasoning) > 0
    
    # Combine reasoning chains
    combined_reasoning = agent_a_reasoning + agent_b_reasoning
    
    print(f"[OK] Agent A reasoning steps: {len(agent_a_reasoning)}")
    print(f"[OK] Agent B reasoning steps: {len(agent_b_reasoning)}")
    print(f"[OK] Combined reasoning steps: {len(combined_reasoning)}")
    
    # Verify timestamps in reasoning
    for step in combined_reasoning:
        assert '[' in step and ']' in step  # Should have timestamp
    
    print("[OK] Reasoning chains properly aggregated")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("A2A INTEGRATION TEST SUITE")
    print("="*80)
    
    try:
        test_context_object_isolation()
        test_a2a_handoff_standard_case()
        test_a2a_handoff_high_uq()
        test_reasoning_chain_aggregation()
        
        print("\n" + "="*80)
        print("ALL INTEGRATION TESTS PASSED")
        print("="*80)
        print("\n[SUCCESS] Agent A → Agent B handoff working correctly")
        print("[SUCCESS] ContextObject properly isolates Agent B from tools")
        print("[SUCCESS] Reasoning chains preserved and aggregated")
        
    except AssertionError as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
