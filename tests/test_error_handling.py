"""
Tests for LLM Error Handling

This module tests the error handling functionality including:
- Retry logic with exponential backoff
- JSON parsing with recovery
- Error logging
- Fallback mechanisms

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import pytest
import json
import time
from pathlib import Path

from app.services.llm_providers.error_handling import (
    retry_with_backoff,
    parse_json_with_recovery,
    exponential_backoff,
    log_llm_error,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMParsingError,
    LLMRetryExhausted
)


# ============================================================================
# Test Exponential Backoff
# ============================================================================

def test_exponential_backoff():
    """Test exponential backoff calculation"""
    # Test basic exponential growth
    assert exponential_backoff(0, base_delay=1.0) == 1.0
    assert exponential_backoff(1, base_delay=1.0) == 2.0
    assert exponential_backoff(2, base_delay=1.0) == 4.0
    assert exponential_backoff(3, base_delay=1.0) == 8.0
    
    # Test max delay cap
    assert exponential_backoff(10, base_delay=1.0, max_delay=5.0) == 5.0


# ============================================================================
# Test Retry Logic
# ============================================================================

def test_retry_success_on_first_attempt():
    """Test function succeeds on first attempt"""
    call_count = [0]
    
    @retry_with_backoff(max_retries=3, base_delay=0.1, verbose=False)
    def successful_function():
        call_count[0] += 1
        return "success"
    
    result = successful_function()
    assert result == "success"
    assert call_count[0] == 1


def test_retry_success_after_failures():
    """Test function succeeds after some failures"""
    call_count = [0]
    
    @retry_with_backoff(max_retries=3, base_delay=0.1, verbose=False)
    def flaky_function():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError("Simulated error")
        return "success"
    
    result = flaky_function()
    assert result == "success"
    assert call_count[0] == 3


def test_retry_exhausted():
    """Test all retries exhausted"""
    call_count = [0]
    
    @retry_with_backoff(max_retries=2, base_delay=0.1, verbose=False)
    def always_fails():
        call_count[0] += 1
        raise ConnectionError("Always fails")
    
    with pytest.raises(LLMRetryExhausted):
        always_fails()
    
    assert call_count[0] == 3  # Initial + 2 retries


# ============================================================================
# Test JSON Parsing with Recovery
# ============================================================================

def test_parse_valid_json():
    """Test parsing valid JSON"""
    text = '{"key": "value", "number": 42}'
    result = parse_json_with_recovery(text, verbose=False)
    assert result == {"key": "value", "number": 42}


def test_parse_json_in_markdown():
    """Test parsing JSON in markdown code blocks"""
    # Test ```json block
    text = '```json\n{"key": "value"}\n```'
    result = parse_json_with_recovery(text, verbose=False)
    assert result == {"key": "value"}
    
    # Test ``` block
    text = '```\n{"key": "value"}\n```'
    result = parse_json_with_recovery(text, verbose=False)
    assert result == {"key": "value"}


def test_parse_json_with_extra_text():
    """Test parsing JSON embedded in text"""
    text = 'Here is the result: {"key": "value"} Hope this helps!'
    result = parse_json_with_recovery(text, verbose=False)
    assert result == {"key": "value"}


def test_parse_json_array():
    """Test parsing JSON array"""
    text = 'The data is: [1, 2, 3, 4, 5]'
    result = parse_json_with_recovery(text, verbose=False)
    assert result == [1, 2, 3, 4, 5]


def test_parse_invalid_json():
    """Test parsing invalid JSON raises error"""
    text = 'This is not JSON at all'
    with pytest.raises(LLMParsingError):
        parse_json_with_recovery(text, verbose=False)


def test_parse_empty_string():
    """Test parsing empty string raises error"""
    with pytest.raises(LLMParsingError):
        parse_json_with_recovery("", verbose=False)


# ============================================================================
# Test Error Logging
# ============================================================================

def test_error_logging(tmp_path):
    """Test error logging to file"""
    log_file = tmp_path / "test_errors.log"
    
    # Log an error
    error = ConnectionError("Test error")
    context = {"test": "context", "value": 123}
    log_llm_error(error, context, str(log_file))
    
    # Verify log file exists
    assert log_file.exists()
    
    # Read and verify log content
    with open(log_file, 'r') as f:
        log_line = f.readline()
        log_entry = json.loads(log_line)
    
    assert log_entry['error_type'] == 'ConnectionError'
    assert log_entry['error_message'] == 'Test error'
    assert log_entry['context'] == context
    assert 'timestamp' in log_entry


# ============================================================================
# Test Integration with Agent A
# ============================================================================

def test_agent_a_fallback_on_llm_failure():
    """Test Agent A falls back to rule-based orchestration when LLM fails"""
    from app.agents.agent_a_orchestrator import AgentA, AgentAConfig
    from app.core.mcp_server import DiagnosticMCPServer
    from app.core.ml_processing.cdda_tools import CDDAToolKit
    from app.core.knowledge.graph_rag import GraphRAG
    
    # Initialize components
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # Initialize Agent A with LLM disabled (forces fallback)
    config = AgentAConfig(use_llm=False, verbose=False)
    agent_a = AgentA(mcp_server=mcp_server, config=config)
    
    # Run orchestration (should use rule-based fallback)
    context_object = agent_a.orchestrate('sub-0005')
    
    # Verify context object was created
    assert context_object is not None
    assert context_object.subject_id == 'sub-0005'
    assert context_object.diagnostic_report is not None
    assert context_object.validate()


# ============================================================================
# Test Integration with Agent B
# ============================================================================

def test_agent_b_fallback_on_llm_failure():
    """Test Agent B falls back to template-based synthesis when LLM fails"""
    from app.agents.agent_b_consultant import AgentB, AgentBConfig
    from app.core.models import ContextObject, DiagnosticReport, Feature, AnomalyStatus
    
    # Create mock ContextObject
    diagnostic_report = DiagnosticReport(
        subject_id='sub-0005',
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
        subject_id='sub-0005',
        diagnostic_report=diagnostic_report,
        decision_rationale="Standard case",
        signals={'uq_score': 0.75, 'has_anomaly': False}
    )
    
    # Initialize Agent B with LLM disabled (forces fallback)
    config = AgentBConfig(use_llm=False, verbose=False)
    agent_b = AgentB(config=config)
    
    # Synthesize report (should use template-based fallback)
    result = agent_b.synthesize(context_object)
    
    # Verify report was generated
    assert result is not None
    assert 'clinical_report' in result
    assert len(result['clinical_report']) > 0
    assert 'DIAGNOSTIC SUMMARY' in result['clinical_report']


# ============================================================================
# Test Error Annotations
# ============================================================================

def test_context_object_error_annotations():
    """Test ContextObject error annotation functionality"""
    from app.core.models import ContextObject, DiagnosticReport, Feature, AnomalyStatus
    
    # Create ContextObject
    diagnostic_report = DiagnosticReport(
        subject_id='sub-0005',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.75,
        top_features=[],
        anomaly_status=AnomalyStatus(has_anomaly=False, anomalous_regions=[])
    )
    
    context_object = ContextObject(
        subject_id='sub-0005',
        diagnostic_report=diagnostic_report,
        decision_rationale="Test"
    )
    
    # Initially no errors
    assert not context_object.has_errors()
    assert len(context_object.errors) == 0
    
    # Add error annotation
    context_object.add_error(
        error_type="GraphRAGError",
        error_message="GraphRAG query failed",
        component="MCP Server"
    )
    
    # Verify error was added
    assert context_object.has_errors()
    assert len(context_object.errors) == 1
    assert context_object.errors[0]['type'] == "GraphRAGError"
    assert context_object.errors[0]['message'] == "GraphRAG query failed"
    assert context_object.errors[0]['component'] == "MCP Server"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
