"""
Unit Tests for CDDA Agent

Tests the autonomous decision logic of the CDDA Agent (Layer 3)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


def test_agent_initialization():
    """Test that agent initializes correctly"""
    print("\n" + "="*80)
    print("TEST: Agent Initialization")
    print("="*80)
    
    agent = CDDAAgent(verbose=False)
    
    # Check toolkit is initialized
    assert agent.toolkit is not None, "Toolkit should be initialized"
    assert agent.uq_threshold == 0.8, "Default UQ threshold should be 0.8"
    assert agent.z_score_threshold == 2.5, "Default Z-score threshold should be 2.5"
    
    print("  ✓ Agent initialized successfully")
    print("  ✓ Toolkit connected")
    print("  ✓ Thresholds set correctly")
    
    print("\n[SUCCESS] Agent initialization test passed")
    return True


def test_standard_case_decision():
    """Test Decision C: Standard case (low uncertainty, no anomalies)"""
    print("\n" + "="*80)
    print("TEST: Standard Case Decision Logic")
    print("="*80)
    
    agent = CDDAAgent(use_llm=False, verbose=False)
    result = agent.run_analysis('sub-0015')
    
    # Check decision
    assert result.agent_decision == 'STANDARD_REPORT', \
        "Should trigger standard report for low uncertainty, no anomalies"
    
    # Check required fields
    assert result.subject_id is not None
    assert result.prediction is not None
    assert result.confidence is not None
    assert result.uq_score is not None
    assert result.clinical_report is not None
    assert result.reasoning_chain is not None
    
    print(f"\n  Subject: {result.subject_id}")
    print(f"  Decision: {result.agent_decision}")
    print(f"  Prediction: {result.prediction} ({result.confidence:.1%})")
    print(f"  UQ Score: {result.uq_score:.3f}")
    
    print("\n  Reasoning Chain (first 5 steps):")
    for step in result.reasoning_chain[:5]:
        print(f"    {step}")
    
    print("\n[SUCCESS] Standard case decision logic working")
    return True


def test_high_uncertainty_decision():
    """Test Decision A: High uncertainty triggers counterfactual"""
    print("\n" + "="*80)
    print("TEST: High Uncertainty Decision Logic")
    print("="*80)
    
    # Lower threshold to trigger simulation
    agent = CDDAAgent(uq_threshold=0.7, use_llm=False, verbose=False)
    result = agent.run_analysis('sub-0005')
    
    # Check decision
    assert result.agent_decision == 'SIMULATION_TRIGGERED', \
        "Should trigger simulation for high uncertainty"
    
    # Check counterfactual data is present
    assert result.context_object.tool_results is not None, "Should include tool results"
    assert 'counterfactual' in result.context_object.tool_results, "Should include counterfactual results"
    assert 'confidence_delta' in result.context_object.tool_results['counterfactual']
    
    print(f"\n  Subject: {result.subject_id}")
    print(f"  Decision: {result.agent_decision}")
    print(f"  UQ Score: {result.uq_score:.3f} (threshold: 0.7)")
    print(f"  Counterfactual Impact: {result.context_object.tool_results['counterfactual']['confidence_delta']:+.1%}")
    
    print("\n  Reasoning Chain (first 5 steps):")
    for step in result.reasoning_chain[:5]:
        print(f"    {step}")
    
    print("\n[SUCCESS] High uncertainty decision logic working")
    return True


def test_anomaly_detection_decision():
    """Test Decision B: Anomaly triggers knowledge lookup"""
    print("\n" + "="*80)
    print("TEST: Anomaly Detection Decision Logic")
    print("="*80)
    
    # Lower z-score threshold to trigger anomaly detection
    agent = CDDAAgent(z_score_threshold=1.5, use_llm=False, verbose=False)
    result = agent.run_analysis('sub-0005')
    
    # Check decision
    assert result.agent_decision == 'ANOMALY_INVESTIGATION', \
        "Should trigger anomaly investigation"
    
    # Check knowledge context is present
    assert result.context_object.tool_results is not None, "Should include tool results"
    assert 'knowledge_context' in result.context_object.tool_results, "Should include knowledge context"
    assert 'contexts' in result.context_object.tool_results['knowledge_context']
    
    print(f"\n  Subject: {result.subject_id}")
    print(f"  Decision: {result.agent_decision}")
    print(f"  Anomalous Regions: {len(result.context_object.diagnostic_report.anomaly_status.anomalous_regions)}")
    print(f"  Knowledge Contexts Retrieved: {len(result.context_object.tool_results['knowledge_context']['contexts'])}")
    
    print("\n  Reasoning Chain (first 5 steps):")
    for step in result.reasoning_chain[:5]:
        print(f"    {step}")
    
    print("\n[SUCCESS] Anomaly detection decision logic working")
    return True


def test_decision_priority():
    """Test that UQ check has priority over anomaly check"""
    print("\n" + "="*80)
    print("TEST: Decision Priority (UQ > Anomaly)")
    print("="*80)
    
    # Set both thresholds low so both conditions could trigger
    agent = CDDAAgent(uq_threshold=0.7, z_score_threshold=1.5, use_llm=False, verbose=False)
    result = agent.run_analysis('sub-0005')
    
    # UQ should take priority
    assert result.agent_decision == 'SIMULATION_TRIGGERED', \
        "UQ check should have priority over anomaly check"
    
    print(f"\n  Subject: {result.subject_id}")
    print(f"  UQ Score: {result.uq_score:.3f} (threshold: 0.7) ✓ TRIGGERED")
    print(f"  Anomalies: {result.context_object.diagnostic_report.anomaly_status.has_anomaly} ✓ PRESENT")
    print(f"  Decision: {result.agent_decision} (UQ takes priority)")
    
    print("\n[SUCCESS] Decision priority working correctly")
    return True


def test_knowledge_graph_lookup():
    """Test knowledge graph lookup functionality"""
    print("\n" + "="*80)
    print("TEST: Knowledge Graph Lookup")
    print("="*80)
    
    agent = CDDAAgent(verbose=False)
    
    # Test with known regions
    test_regions = ['SN_pc', 'Hippocampus', 'ACC']
    knowledge = agent.knowledge_graph_lookup(test_regions)
    
    # Check structure
    assert 'query_regions' in knowledge
    assert 'contexts' in knowledge
    assert 'summary' in knowledge
    
    print(f"\n  Query Regions: {test_regions}")
    print(f"  Contexts Retrieved: {len(knowledge['contexts'])}")
    print(f"\n  Summary:")
    print(f"    {knowledge['summary']}")
    
    # Check each context
    for ctx in knowledge['contexts']:
        print(f"\n  Region: {ctx['region']}")
        print(f"    Full Name: {ctx['context']['full_name']}")
        print(f"    Function: {ctx['context']['function']}")
    
    print("\n[SUCCESS] Knowledge graph lookup working")
    return True


def test_report_output_format():
    """Test that all report types have consistent output format"""
    print("\n" + "="*80)
    print("TEST: Report Output Format Consistency")
    print("="*80)
    
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
    
    # Test all three decision paths
    test_cases = [
        ("Standard", CDDAAgent(use_llm=False, verbose=False), 'sub-0015'),
        ("High UQ", CDDAAgent(uq_threshold=0.7, use_llm=False, verbose=False), 'sub-0005'),
        ("Anomaly", CDDAAgent(z_score_threshold=1.5, use_llm=False, verbose=False), 'sub-0005')
    ]
    
    for case_name, agent, subject_id in test_cases:
        print(f"\n  Testing {case_name} Case...")
        result = agent.run_analysis(subject_id)
        
        for field in required_fields:
            assert hasattr(result, field), f"Missing required field: {field}"
            print(f"    ✓ {field}")
    
    print("\n[SUCCESS] All report formats consistent")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("CDDA AGENT - TEST SUITE")
    print("="*80)
    
    tests = [
        ("Agent Initialization", test_agent_initialization),
        ("Standard Case Decision", test_standard_case_decision),
        ("High Uncertainty Decision", test_high_uncertainty_decision),
        ("Anomaly Detection Decision", test_anomaly_detection_decision),
        ("Decision Priority", test_decision_priority),
        ("Knowledge Graph Lookup", test_knowledge_graph_lookup),
        ("Report Output Format", test_report_output_format)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "FAIL"))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! CDDA Agent is ready for Phase 3.")
    else:
        print("\n⚠️  Some tests failed. Please review.")
    
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
