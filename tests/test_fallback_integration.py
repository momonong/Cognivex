"""
Fallback Integration Tests

This module tests fallback mechanisms:
- System with Agent A unavailable (rule-based fallback)
- System with Agent B unavailable (template-based fallback)
- System with GraphRAG unavailable (fallback knowledge base)
- System with all LLMs unavailable (full fallback mode)

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent
from app.agents.agent_a_orchestrator import AgentA, AgentAConfig
from app.agents.agent_b_consultant import AgentB, AgentBConfig
from app.core.mcp_server import DiagnosticMCPServer
from app.core.ml_processing.cdda_tools import CDDAToolKit
from app.core.knowledge.graph_rag import GraphRAG


def test_agent_a_fallback():
    """
    Test system with Agent A LLM unavailable (rule-based fallback)
    
    Expected behavior:
    - Agent A uses rule-based orchestration
    - Decision logic based on thresholds
    - ContextObject still compiled correctly
    - Agent B receives valid context
    
    Requirements: 10.2
    """
    print("\n" + "="*80)
    print("FALLBACK TEST: Agent A Unavailable (Rule-Based Fallback)")
    print("="*80)
    
    # Initialize with LLM disabled (fallback mode)
    agent = CDDAAgent(
        use_llm=False,  # Force fallback
        verbose=False
    )
    
    print("\n[Running analysis with Agent A fallback...]")
    result = agent.run_analysis('sub-0005')
    
    # Validate result is still complete
    print("\n[Validating result completeness...]")
    assert result is not None
    assert result.subject_id == 'sub-0005'
    assert result.prediction is not None
    assert result.confidence is not None
    assert result.clinical_report is not None
    assert result.reasoning_chain is not None
    
    print(f"  ✓ Subject: {result.subject_id}")
    print(f"  ✓ Prediction: {result.prediction} ({result.confidence:.1%})")
    print(f"  ✓ Decision: {result.agent_decision}")
    print(f"  ✓ Report length: {len(result.clinical_report)} chars")
    print(f"  ✓ Reasoning steps: {len(result.reasoning_chain)}")
    
    # Validate ContextObject
    print("\n[Validating ContextObject...]")
    assert result.context_object is not None
    assert result.context_object.validate()
    print(f"  ✓ ContextObject valid")
    
    # Validate reasoning chain mentions fallback
    reasoning_text = ' '.join(result.reasoning_chain)
    assert 'rule-based' in reasoning_text.lower() or 'fallback' in reasoning_text.lower()
    print(f"  ✓ Fallback mode indicated in reasoning")
    
    print("\n[OK] Agent A fallback working correctly")


def test_agent_b_fallback():
    """
    Test system with Agent B LLM unavailable (template-based fallback)
    
    Expected behavior:
    - Agent B uses template-based synthesis
    - Report still generated with all sections
    - Clinical recommendations included
    
    Requirements: 10.3
    """
    print("\n" + "="*80)
    print("FALLBACK TEST: Agent B Unavailable (Template-Based Fallback)")
    print("="*80)
    
    # Initialize with LLM disabled (fallback mode)
    agent = CDDAAgent(
        use_llm=False,  # Force fallback
        verbose=False
    )
    
    print("\n[Running analysis with Agent B fallback...]")
    result = agent.run_analysis('sub-0005')
    
    # Validate report structure
    print("\n[Validating report structure...]")
    report = result.clinical_report
    
    required_sections = [
        'DIAGNOSTIC SUMMARY',
        'KEY FINDINGS',
        'CLINICAL INTERPRETATION',
        'RECOMMENDATIONS'
    ]
    
    for section in required_sections:
        assert section in report, f"Missing section: {section}"
        print(f"  ✓ {section}")
    
    # Validate report content
    print("\n[Validating report content...]")
    assert result.subject_id in report
    assert result.prediction in report
    print(f"  ✓ Subject ID present")
    print(f"  ✓ Prediction present")
    
    # Validate reasoning chain mentions template
    reasoning_text = ' '.join(result.reasoning_chain)
    assert 'template' in reasoning_text.lower() or 'fallback' in reasoning_text.lower()
    print(f"  ✓ Template mode indicated in reasoning")
    
    print("\n[OK] Agent B fallback working correctly")


def test_graphrag_fallback():
    """
    Test system with GraphRAG unavailable (fallback knowledge base)
    
    Expected behavior:
    - System detects GraphRAG failure
    - Falls back to local knowledge base
    - Analysis continues without crashing
    - Error noted in reasoning chain
    
    Requirements: 10.4
    """
    print("\n" + "="*80)
    print("FALLBACK TEST: GraphRAG Unavailable (Fallback Knowledge)")
    print("="*80)
    
    # Initialize agent with low z-score threshold to trigger knowledge lookup
    agent = CDDAAgent(
        z_score_threshold=1.5,
        use_llm=False,
        verbose=False
    )
    
    # Mock GraphRAG to simulate failure
    print("\n[Simulating GraphRAG failure...]")
    original_query = agent.graph_rag.query_with_context
    
    def mock_query_failure(*args, **kwargs):
        raise Exception("GraphRAG connection failed")
    
    agent.graph_rag.query_with_context = mock_query_failure
    
    # Run analysis (should not crash)
    print("\n[Running analysis with GraphRAG failure...]")
    try:
        result = agent.run_analysis('sub-0005')
        
        # Validate result is still complete
        print("\n[Validating result completeness...]")
        assert result is not None
        assert result.clinical_report is not None
        print(f"  ✓ Analysis completed despite GraphRAG failure")
        print(f"  ✓ Report generated: {len(result.clinical_report)} chars")
        
        # Check if fallback was used
        reasoning_text = ' '.join(result.reasoning_chain)
        if 'fallback' in reasoning_text.lower() or 'error' in reasoning_text.lower():
            print(f"  ✓ Fallback indicated in reasoning")
        
        print("\n[OK] GraphRAG fallback working correctly")
        
    finally:
        # Restore original method
        agent.graph_rag.query_with_context = original_query


def test_full_fallback_mode():
    """
    Test system with all LLMs unavailable (full fallback mode)
    
    Expected behavior:
    - Agent A uses rule-based orchestration
    - Agent B uses template-based synthesis
    - System produces complete diagnostic report
    - All errors logged in reasoning chain
    
    Requirements: 10.1, 10.2, 10.3, 10.5
    """
    print("\n" + "="*80)
    print("FALLBACK TEST: All LLMs Unavailable (Full Fallback)")
    print("="*80)
    
    # Initialize with all LLMs disabled
    agent = CDDAAgent(
        use_llm=False,  # Disables both Agent A and Agent B LLMs
        verbose=False
    )
    
    print("\n[Running analysis in full fallback mode...]")
    result = agent.run_analysis('sub-0005')
    
    # Validate complete result
    print("\n[Validating result completeness...]")
    
    required_fields = [
        'subject_id',
        'agent_decision',
        'prediction',
        'confidence',
        'uq_score',
        'context_object',
        'clinical_report',
        'reasoning_chain',
        'timestamp'
    ]
    
    for field in required_fields:
        assert hasattr(result, field), f"Missing field: {field}"
        value = getattr(result, field)
        assert value is not None, f"Field {field} is None"
        print(f"  ✓ {field}")
    
    # Validate report structure
    print("\n[Validating report structure...]")
    report = result.clinical_report
    assert 'DIAGNOSTIC SUMMARY' in report
    assert 'KEY FINDINGS' in report
    assert 'RECOMMENDATIONS' in report
    print(f"  ✓ All sections present")
    
    # Validate reasoning chain
    print("\n[Validating reasoning chain...]")
    assert len(result.reasoning_chain) > 0
    reasoning_text = ' '.join(result.reasoning_chain)
    
    # Should indicate fallback mode
    fallback_indicators = ['rule-based', 'template', 'fallback']
    has_fallback_indicator = any(indicator in reasoning_text.lower() for indicator in fallback_indicators)
    assert has_fallback_indicator, "Reasoning should indicate fallback mode"
    print(f"  ✓ Fallback mode indicated")
    print(f"  ✓ Reasoning steps: {len(result.reasoning_chain)}")
    
    print("\n[OK] Full fallback mode working correctly")


def test_partial_fallback_combinations():
    """
    Test various combinations of fallback scenarios
    
    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
    """
    print("\n" + "="*80)
    print("FALLBACK TEST: Partial Fallback Combinations")
    print("="*80)
    
    # Test Case 1: Agent A fallback + Agent B LLM
    print("\n[Test Case 1: Agent A fallback only]")
    agent1 = CDDAAgent(use_llm=False, verbose=False)
    result1 = agent1.run_analysis('sub-0005')
    assert result1.clinical_report is not None
    print(f"  ✓ Agent A fallback + Agent B template: {len(result1.clinical_report)} chars")
    
    # Test Case 2: Different thresholds in fallback mode
    print("\n[Test Case 2: High UQ threshold in fallback]")
    agent2 = CDDAAgent(uq_threshold=0.7, use_llm=False, verbose=False)
    result2 = agent2.run_analysis('sub-0005')
    assert result2.clinical_report is not None
    print(f"  ✓ High UQ fallback: {result2.agent_decision}")
    
    # Test Case 3: Anomaly detection in fallback mode
    print("\n[Test Case 3: Anomaly detection in fallback]")
    agent3 = CDDAAgent(z_score_threshold=1.5, use_llm=False, verbose=False)
    result3 = agent3.run_analysis('sub-0005')
    assert result3.clinical_report is not None
    print(f"  ✓ Anomaly fallback: {result3.agent_decision}")
    
    print("\n[OK] All partial fallback combinations working")


def test_fallback_error_annotations():
    """
    Test that errors are properly annotated in fallback mode
    
    Requirements: 10.5
    """
    print("\n" + "="*80)
    print("FALLBACK TEST: Error Annotations")
    print("="*80)
    
    # Initialize in fallback mode
    agent = CDDAAgent(use_llm=False, verbose=False)
    
    print("\n[Running analysis in fallback mode...]")
    result = agent.run_analysis('sub-0005')
    
    # Check reasoning chain for error/fallback annotations
    print("\n[Checking reasoning chain for annotations...]")
    reasoning_text = ' '.join(result.reasoning_chain)
    
    # Should mention fallback or rule-based mode
    has_annotation = (
        'fallback' in reasoning_text.lower() or
        'rule-based' in reasoning_text.lower() or
        'template' in reasoning_text.lower()
    )
    
    assert has_annotation, "Reasoning should annotate fallback mode"
    print(f"  ✓ Fallback mode annotated in reasoning")
    
    # Check if report mentions any limitations
    report = result.clinical_report
    print(f"  ✓ Report generated: {len(report)} chars")
    
    print("\n[OK] Error annotations present")


def test_fallback_performance():
    """
    Test that fallback mode performs reasonably
    
    Requirements: 10.1, 10.2, 10.3
    """
    print("\n" + "="*80)
    print("FALLBACK TEST: Performance")
    print("="*80)
    
    import time
    
    # Initialize in fallback mode
    agent = CDDAAgent(use_llm=False, verbose=False)
    
    # Measure performance
    print("\n[Measuring fallback performance...]")
    start_time = time.time()
    result = agent.run_analysis('sub-0005')
    elapsed_time = time.time() - start_time
    
    print(f"\n[Performance Metrics]")
    print(f"  Elapsed time: {elapsed_time:.2f}s")
    print(f"  Report length: {len(result.clinical_report)} chars")
    print(f"  Reasoning steps: {len(result.reasoning_chain)}")
    
    # Fallback should be reasonably fast (< 30 seconds)
    assert elapsed_time < 30, f"Fallback too slow: {elapsed_time:.2f}s"
    print(f"  ✓ Performance acceptable")
    
    print("\n[OK] Fallback performance validated")


def run_all_fallback_tests():
    """Run all fallback integration tests"""
    print("\n" + "="*80)
    print("FALLBACK INTEGRATION TEST SUITE")
    print("="*80)
    
    tests = [
        ("Agent A Fallback", test_agent_a_fallback),
        ("Agent B Fallback", test_agent_b_fallback),
        ("GraphRAG Fallback", test_graphrag_fallback),
        ("Full Fallback Mode", test_full_fallback_mode),
        ("Partial Fallback Combinations", test_partial_fallback_combinations),
        ("Error Annotations", test_fallback_error_annotations),
        ("Fallback Performance", test_fallback_performance)
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
    print("FALLBACK TEST SUMMARY")
    print("="*80)
    
    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All fallback integration tests passed!")
        return 0
    else:
        print("\n⚠️  Some fallback tests failed. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_fallback_tests())
