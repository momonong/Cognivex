"""
Tests for Agent A Orchestrator

This module tests the Agent A orchestrator implementation including:
- Rule-based orchestration
- LLM-based orchestration (when available)
- MCP client functionality
- Reasoning chain logging
- ContextObject compilation

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.1, 8.2
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.agent_a_orchestrator import AgentA, AgentAConfig
from app.core.mcp_server import DiagnosticMCPServer
from app.core.ml_processing.cdda_tools import CDDAToolKit
from app.core.knowledge.graph_rag import GraphRAG
from app.core.models import ContextObject


@pytest.fixture
def mcp_server():
    """Create MCP server for testing"""
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    return DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)


@pytest.fixture
def agent_a_rule_based(mcp_server):
    """Create Agent A with rule-based orchestration"""
    config = AgentAConfig(use_llm=False, verbose=False)
    return AgentA(mcp_server=mcp_server, config=config)


def test_agent_a_initialization(mcp_server):
    """Test Agent A initialization"""
    config = AgentAConfig(use_llm=False, verbose=False)
    agent = AgentA(mcp_server=mcp_server, config=config)
    
    assert agent.mcp_server is not None
    assert agent.config is not None
    assert agent.system_prompt is not None
    assert isinstance(agent.reasoning_chain, list)
    assert isinstance(agent.mcp_actions, list)


def test_orchestrate_rule_based(agent_a_rule_based):
    """
    Test rule-based orchestration
    
    Requirements: 3.1, 3.2, 3.3, 3.4
    """
    # Run orchestration
    context_object = agent_a_rule_based.orchestrate('sub-0005')
    
    # Verify ContextObject structure
    assert isinstance(context_object, ContextObject)
    assert context_object.subject_id == 'sub-0005'
    assert context_object.diagnostic_report is not None
    assert context_object.decision_rationale != ""
    assert len(context_object.agent_a_reasoning) > 0
    assert len(context_object.mcp_actions) > 0
    
    # Verify diagnostic report
    assert context_object.diagnostic_report.prediction_result in ['AD', 'NC', 'MCI']
    assert 0.0 <= context_object.diagnostic_report.confidence <= 1.0
    assert 0.0 <= context_object.diagnostic_report.uq_score <= 1.0
    
    # Verify signals
    assert 'uq_score' in context_object.signals
    assert 'has_anomaly' in context_object.signals
    assert 'prediction' in context_object.signals
    assert 'confidence' in context_object.signals


def test_diagnostic_report_always_first(agent_a_rule_based):
    """
    Test that diagnostic report is always read first
    
    Requirements: 3.1
    """
    context_object = agent_a_rule_based.orchestrate('sub-0005')
    
    # First MCP action should be reading diagnostic report
    assert len(context_object.mcp_actions) > 0
    first_action = context_object.mcp_actions[0]
    assert first_action.type == "read_resource"
    assert "diagnosis://" in first_action.target
    assert "/report" in first_action.target


def test_high_uq_triggers_counterfactual(mcp_server):
    """
    Test that high UQ triggers counterfactual simulation
    
    Requirements: 3.2
    """
    # Use lower threshold to trigger counterfactual
    config = AgentAConfig(use_llm=False, uq_threshold=0.7, verbose=False)
    agent = AgentA(mcp_server=mcp_server, config=config)
    
    context_object = agent.orchestrate('sub-0005')
    
    # Check if counterfactual was triggered
    if context_object.diagnostic_report.uq_score > 0.7:
        assert context_object.tool_results is not None
        assert 'counterfactual' in context_object.tool_results
        
        # Verify counterfactual result structure
        cf_result = context_object.tool_results['counterfactual']
        assert 'original_prediction' in cf_result
        assert 'new_prediction' in cf_result
        assert 'confidence_delta' in cf_result


def test_anomaly_triggers_knowledge_graph(mcp_server):
    """
    Test that anomalies trigger knowledge graph queries
    
    Requirements: 3.3
    """
    # Use lower threshold to trigger anomaly detection
    config = AgentAConfig(use_llm=False, z_score_threshold=1.5, verbose=False)
    agent = AgentA(mcp_server=mcp_server, config=config)
    
    context_object = agent.orchestrate('sub-0005')
    
    # Check if knowledge graph was queried
    if context_object.diagnostic_report.anomaly_status.has_anomaly:
        assert context_object.tool_results is not None
        assert 'knowledge_context' in context_object.tool_results
        
        # Verify knowledge context structure
        kg_result = context_object.tool_results['knowledge_context']
        assert 'query_regions' in kg_result
        assert 'contexts' in kg_result
        assert 'summary' in kg_result
        assert len(kg_result['contexts']) > 0


def test_reasoning_chain_logging(agent_a_rule_based):
    """
    Test that reasoning chain is logged correctly
    
    Requirements: 3.5, 8.1, 8.2
    """
    context_object = agent_a_rule_based.orchestrate('sub-0005')
    
    # Verify reasoning chain exists and has content
    assert len(context_object.agent_a_reasoning) > 0
    
    # Verify each reasoning step has timestamp
    for step in context_object.agent_a_reasoning:
        assert '[' in step  # Contains timestamp
        assert ']' in step
    
    # Verify key reasoning steps are present
    reasoning_text = ' '.join(context_object.agent_a_reasoning)
    assert 'Starting orchestration' in reasoning_text
    assert 'Read diagnostic report' in reasoning_text


def test_mcp_actions_logging(agent_a_rule_based):
    """
    Test that MCP actions are logged correctly
    
    Requirements: 8.1, 8.2
    """
    context_object = agent_a_rule_based.orchestrate('sub-0005')
    
    # Verify MCP actions exist
    assert len(context_object.mcp_actions) > 0
    
    # Verify each action has required fields
    for action in context_object.mcp_actions:
        assert action.type in ['read_resource', 'call_tool']
        assert action.target is not None
        assert action.status in ['pending', 'success', 'error']
        assert action.timestamp is not None
        
        # If successful, should have result
        if action.status == 'success':
            assert action.result is not None


def test_context_object_validation(agent_a_rule_based):
    """
    Test that ContextObject is properly validated
    
    Requirements: 5.1
    """
    context_object = agent_a_rule_based.orchestrate('sub-0005')
    
    # Validate should return True
    assert context_object.validate() == True
    
    # Verify all required fields are present
    assert context_object.subject_id is not None
    assert context_object.diagnostic_report is not None
    assert context_object.signals is not None
    assert context_object.decision_rationale is not None


def test_context_object_serialization(agent_a_rule_based):
    """
    Test that ContextObject can be serialized for Agent B
    
    Requirements: 5.1, 8.3
    """
    context_object = agent_a_rule_based.orchestrate('sub-0005')
    
    # Serialize for Agent B
    serialized = context_object.serialize_for_agent_b()
    
    # Verify it's valid JSON
    import json
    parsed = json.loads(serialized)
    
    # Verify key fields are present
    assert 'subject_id' in parsed
    assert 'diagnostic_report' in parsed
    assert 'decision_rationale' in parsed
    assert 'signals' in parsed
    assert 'agent_a_reasoning' in parsed
    assert 'mcp_actions' in parsed


def test_reasoning_log_save(agent_a_rule_based, tmp_path):
    """
    Test that reasoning log can be saved to file
    
    Requirements: 3.5, 8.1, 8.2
    """
    context_object = agent_a_rule_based.orchestrate('sub-0005')
    
    # Save reasoning log
    log_path = tmp_path / "test_reasoning_log.json"
    agent_a_rule_based.save_reasoning_log(str(log_path))
    
    # Verify file was created
    assert log_path.exists()
    
    # Verify file contains valid JSON
    import json
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # Verify structure
    assert 'agent' in log_data
    assert 'timestamp' in log_data
    assert 'reasoning_chain' in log_data
    assert 'mcp_actions' in log_data
    assert log_data['agent'] == 'Agent A - Orchestrator'


def test_standard_case_decision(mcp_server):
    """
    Test standard case (low UQ, no anomalies)
    
    Requirements: 3.4
    """
    # Use high thresholds to avoid triggering tools
    config = AgentAConfig(
        use_llm=False, 
        uq_threshold=0.9, 
        z_score_threshold=3.0,
        verbose=False
    )
    agent = AgentA(mcp_server=mcp_server, config=config)
    
    context_object = agent.orchestrate('sub-0005')
    
    # For standard case, tool_results should be empty or None
    if context_object.diagnostic_report.uq_score < 0.9 and \
       not context_object.diagnostic_report.anomaly_status.has_anomaly:
        assert context_object.tool_results is None or \
               len(context_object.tool_results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
