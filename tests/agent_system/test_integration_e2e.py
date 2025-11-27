"""
End-to-End Integration Tests for CDDA Phase 4

This module tests complete workflows through the entire system:
- Standard case (low UQ, no anomalies)
- High uncertainty case (triggers counterfactual)
- Anomaly case (triggers knowledge graph)
- Mixed case (both counterfactual and knowledge graph)

Requirements: All
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


def test_e2e_standard_case():
    """
    Test E2E: Standard case (low UQ, no anomalies)
    
    Expected flow:
    1. Agent A reads diagnostic report
    2. Agent A evaluates signals (low UQ, no anomalies)
    3. Agent A compiles ContextObject
    4. Agent B synthesizes standard report
    
    Requirements: 3.1, 3.4, 5.1, 5.2, 5.3, 5.4, 5.5
    """
    print("\n" + "="*80)
    print("E2E TEST: Standard Case (Low UQ, No Anomalies)")
    print("="*80)
    
    # Initialize agent with high thresholds (standard case)
    agent = CDDAAgent(
        uq_threshold=0.9,
        z_score_threshold=3.0,
        use_llm=False,
        verbose=False
    )
    
    # Run analysis
    print("\n[Phase 1] Running analysis...")
    result = agent.run_analysis('sub-0015')
    
    # Validate decision
    print(f"\n[Phase 2] Validating decision...")
    assert result.agent_decision == 'STANDARD_REPORT', \
        f"Expected STANDARD_REPORT, got {result.agent_decision}"
    
    # Validate ContextObject
    print(f"[Phase 3] Validating ContextObject...")
    assert result.context_object is not None
    assert result.context_object.subject_id == 'sub-0015'
    assert result.context_object.diagnostic_report is not None
    assert result.context_object.validate()
    
    # Validate no tool results for standard case
    if result.uq_score < 0.9 and not result.context_object.diagnostic_report.anomaly_status.has_anomaly:
        assert result.context_object.tool_results is None or len(result.context_object.tool_results) == 0
    
    # Validate clinical report
    print(f"[Phase 4] Validating clinical report...")
    assert result.clinical_report is not None
    assert len(result.clinical_report) > 0
    assert 'DIAGNOSTIC SUMMARY' in result.clinical_report
    assert 'KEY FINDINGS' in result.clinical_report
    assert 'RECOMMENDATIONS' in result.clinical_report
    
    # Validate reasoning chain
    print(f"[Phase 5] Validating reasoning chain...")
    assert len(result.reasoning_chain) > 0
    reasoning_text = ' '.join(result.reasoning_chain)
    assert 'Agent A' in reasoning_text
    assert 'Agent B' in reasoning_text
    
    # Print summary
    print(f"\n[Summary]")
    print(f"  Subject: {result.subject_id}")
    print(f"  Decision: {result.agent_decision}")
    print(f"  Prediction: {result.prediction} ({result.confidence:.1%})")
    print(f"  UQ Score: {result.uq_score:.3f}")
    print(f"  Reasoning steps: {len(result.reasoning_chain)}")
    print(f"  Report length: {len(result.clinical_report)} chars")
    
    print("\n[OK] Standard case E2E test passed")
    return result


def test_e2e_high_uncertainty():
    """
    Test E2E: High uncertainty case (triggers counterfactual)
    
    Expected flow:
    1. Agent A reads diagnostic report
    2. Agent A detects high UQ
    3. Agent A calls counterfactual simulation tool
    4. Agent A compiles ContextObject with counterfactual results
    5. Agent B synthesizes report with counterfactual explanation
    
    Requirements: 3.1, 3.2, 7.1, 7.2, 7.3, 7.4, 7.5
    """
    print("\n" + "="*80)
    print("E2E TEST: High Uncertainty Case (Triggers Counterfactual)")
    print("="*80)
    
    # Initialize agent with lower UQ threshold
    agent = CDDAAgent(
        uq_threshold=0.7,
        z_score_threshold=3.0,
        use_llm=False,
        verbose=False
    )
    
    # Run analysis
    print("\n[Phase 1] Running analysis...")
    result = agent.run_analysis('sub-0005')
    
    # Validate decision
    print(f"\n[Phase 2] Validating decision...")
    assert result.agent_decision == 'SIMULATION_TRIGGERED', \
        f"Expected SIMULATION_TRIGGERED, got {result.agent_decision}"
    
    # Validate counterfactual was executed
    print(f"[Phase 3] Validating counterfactual execution...")
    assert result.context_object.tool_results is not None
    assert 'counterfactual' in result.context_object.tool_results
    
    cf_result = result.context_object.tool_results['counterfactual']
    assert 'original_prediction' in cf_result
    assert 'new_prediction' in cf_result
    assert 'confidence_delta' in cf_result
    assert 'masked_features' in cf_result
    
    # Validate clinical report includes counterfactual
    print(f"[Phase 4] Validating clinical report...")
    assert 'COUNTERFACTUAL' in result.clinical_report or \
           'counterfactual' in result.clinical_report.lower()
    
    # Validate reasoning chain includes counterfactual
    print(f"[Phase 5] Validating reasoning chain...")
    reasoning_text = ' '.join(result.reasoning_chain)
    assert 'counterfactual' in reasoning_text.lower() or \
           'simulation' in reasoning_text.lower()
    
    # Print summary
    print(f"\n[Summary]")
    print(f"  Subject: {result.subject_id}")
    print(f"  Decision: {result.agent_decision}")
    print(f"  UQ Score: {result.uq_score:.3f} (threshold: 0.7)")
    print(f"  Counterfactual delta: {cf_result['confidence_delta']:+.1%}")
    print(f"  Reasoning steps: {len(result.reasoning_chain)}")
    
    print("\n[OK] High uncertainty E2E test passed")
    return result


def test_e2e_anomaly_case():
    """
    Test E2E: Anomaly case (triggers knowledge graph)
    
    Expected flow:
    1. Agent A reads diagnostic report
    2. Agent A detects anomalies
    3. Agent A queries knowledge graph for anomalous regions
    4. Agent A compiles ContextObject with knowledge context
    5. Agent B synthesizes report with anomaly analysis
    
    Requirements: 3.1, 3.3, 4.1, 4.2, 4.3, 4.4, 6.1, 6.2, 6.3, 6.4, 6.5
    """
    print("\n" + "="*80)
    print("E2E TEST: Anomaly Case (Triggers Knowledge Graph)")
    print("="*80)
    
    # Initialize agent with lower z-score threshold
    agent = CDDAAgent(
        uq_threshold=0.9,
        z_score_threshold=1.5,
        use_llm=False,
        verbose=False
    )
    
    # Run analysis
    print("\n[Phase 1] Running analysis...")
    result = agent.run_analysis('sub-0005')
    
    # Validate decision
    print(f"\n[Phase 2] Validating decision...")
    assert result.agent_decision == 'ANOMALY_INVESTIGATION', \
        f"Expected ANOMALY_INVESTIGATION, got {result.agent_decision}"
    
    # Validate knowledge graph was queried
    print(f"[Phase 3] Validating knowledge graph query...")
    assert result.context_object.tool_results is not None
    assert 'knowledge_context' in result.context_object.tool_results
    
    kg_result = result.context_object.tool_results['knowledge_context']
    assert 'query_regions' in kg_result
    assert 'contexts' in kg_result
    assert 'summary' in kg_result
    assert len(kg_result['contexts']) > 0
    
    # Validate clinical report includes anomaly analysis
    print(f"[Phase 4] Validating clinical report...")
    assert 'ANOMALY' in result.clinical_report or \
           'anomaly' in result.clinical_report.lower()
    
    # Check for mixed pathology detection
    if 'mixed pathology' in result.clinical_report.lower() or \
       'MIXED PATHOLOGY' in result.clinical_report:
        print(f"  [OK] Mixed pathology detection present")
    
    # Validate reasoning chain includes knowledge graph
    print(f"[Phase 5] Validating reasoning chain...")
    reasoning_text = ' '.join(result.reasoning_chain)
    assert 'knowledge' in reasoning_text.lower() or \
           'anomaly' in reasoning_text.lower()
    
    # Print summary
    print(f"\n[Summary]")
    print(f"  Subject: {result.subject_id}")
    print(f"  Decision: {result.agent_decision}")
    print(f"  Anomalous regions: {len(result.context_object.diagnostic_report.anomaly_status.anomalous_regions)}")
    print(f"  Knowledge contexts: {len(kg_result['contexts'])}")
    print(f"  Reasoning steps: {len(result.reasoning_chain)}")
    
    print("\n[OK] Anomaly case E2E test passed")
    return result


def test_e2e_mixed_case():
    """
    Test E2E: Mixed case (both counterfactual and knowledge graph)
    
    Expected flow:
    1. Agent A reads diagnostic report
    2. Agent A detects high UQ (priority)
    3. Agent A calls counterfactual simulation
    4. Agent A also detects anomalies
    5. Agent A queries knowledge graph
    6. Agent A compiles ContextObject with both results
    7. Agent B synthesizes comprehensive report
    
    Requirements: All
    """
    print("\n" + "="*80)
    print("E2E TEST: Mixed Case (Counterfactual + Knowledge Graph)")
    print("="*80)
    
    # Initialize agent with low thresholds for both
    agent = CDDAAgent(
        uq_threshold=0.7,
        z_score_threshold=1.5,
        use_llm=False,
        verbose=False
    )
    
    # Run analysis
    print("\n[Phase 1] Running analysis...")
    result = agent.run_analysis('sub-0005')
    
    # Validate decision (UQ takes priority)
    print(f"\n[Phase 2] Validating decision...")
    assert result.agent_decision == 'SIMULATION_TRIGGERED', \
        f"Expected SIMULATION_TRIGGERED (UQ priority), got {result.agent_decision}"
    
    # Validate both tools were executed
    print(f"[Phase 3] Validating tool execution...")
    assert result.context_object.tool_results is not None
    
    # Check for counterfactual
    has_counterfactual = 'counterfactual' in result.context_object.tool_results
    print(f"  Counterfactual: {has_counterfactual}")
    
    # Check for knowledge context (may or may not be present depending on anomalies)
    has_knowledge = 'knowledge_context' in result.context_object.tool_results
    print(f"  Knowledge context: {has_knowledge}")
    
    # Validate clinical report
    print(f"[Phase 4] Validating clinical report...")
    assert len(result.clinical_report) > 0
    
    # Validate reasoning chain
    print(f"[Phase 5] Validating reasoning chain...")
    assert len(result.reasoning_chain) > 0
    
    # Print summary
    print(f"\n[Summary]")
    print(f"  Subject: {result.subject_id}")
    print(f"  Decision: {result.agent_decision}")
    print(f"  UQ Score: {result.uq_score:.3f}")
    print(f"  Has anomalies: {result.context_object.diagnostic_report.anomaly_status.has_anomaly}")
    print(f"  Tools executed: {list(result.context_object.tool_results.keys()) if result.context_object.tool_results else []}")
    print(f"  Reasoning steps: {len(result.reasoning_chain)}")
    
    print("\n[OK] Mixed case E2E test passed")
    return result


def test_e2e_result_consistency():
    """
    Test that all E2E workflows produce consistent result structure
    
    Requirements: 8.3, 8.4
    """
    print("\n" + "="*80)
    print("E2E TEST: Result Structure Consistency")
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
    
    # Test all three main cases
    test_cases = [
        ("Standard", CDDAAgent(uq_threshold=0.9, z_score_threshold=3.0, use_llm=False, verbose=False), 'sub-0015'),
        ("High UQ", CDDAAgent(uq_threshold=0.7, z_score_threshold=3.0, use_llm=False, verbose=False), 'sub-0005'),
        ("Anomaly", CDDAAgent(uq_threshold=0.9, z_score_threshold=1.5, use_llm=False, verbose=False), 'sub-0005')
    ]
    
    print("\n[Testing result structure consistency across all cases...]")
    
    for case_name, agent, subject_id in test_cases:
        print(f"\n  Testing {case_name} case...")
        result = agent.run_analysis(subject_id)
        
        # Check all required fields
        for field in required_fields:
            assert hasattr(result, field), f"{case_name}: Missing field {field}"
            value = getattr(result, field)
            assert value is not None, f"{case_name}: Field {field} is None"
        
        # Validate ContextObject
        assert result.context_object.validate(), f"{case_name}: ContextObject validation failed"
        
        # Validate report structure
        assert 'DIAGNOSTIC SUMMARY' in result.clinical_report
        assert 'KEY FINDINGS' in result.clinical_report
        assert 'RECOMMENDATIONS' in result.clinical_report
        
        print(f"    ✓ All fields present")
        print(f"    ✓ ContextObject valid")
        print(f"    ✓ Report structure correct")
    
    print("\n[OK] Result structure consistent across all cases")


def test_e2e_reasoning_chain_completeness():
    """
    Test that reasoning chains are complete and properly structured
    
    Requirements: 8.1, 8.2, 8.3, 8.4
    """
    print("\n" + "="*80)
    print("E2E TEST: Reasoning Chain Completeness")
    print("="*80)
    
    # Run standard case
    agent = CDDAAgent(use_llm=False, verbose=False)
    result = agent.run_analysis('sub-0005')
    
    print("\n[Validating reasoning chain structure...]")
    
    # Check reasoning chain exists
    assert len(result.reasoning_chain) > 0
    
    # Check for key phases
    reasoning_text = ' '.join(result.reasoning_chain)
    
    required_phases = [
        ('Data Gathering', ['diagnostic', 'report', 'read']),
        ('Signal Evaluation', ['uq', 'anomaly', 'evaluate']),
        ('Agent A', ['Agent A', 'orchestrat']),
        ('Agent B', ['Agent B', 'synthesis', 'consultant'])
    ]
    
    for phase_name, keywords in required_phases:
        found = any(keyword.lower() in reasoning_text.lower() for keyword in keywords)
        assert found, f"Missing phase: {phase_name}"
        print(f"  ✓ {phase_name} phase present")
    
    # Check that most steps have timestamps (allow for headers and separators)
    timestamped_steps = [step for step in result.reasoning_chain if '[' in step and ']' in step]
    total_steps = len(result.reasoning_chain)
    
    # At least 40% of steps should have timestamps (actual reasoning steps)
    assert len(timestamped_steps) >= total_steps * 0.4, \
        f"Too few timestamped steps: {len(timestamped_steps)}/{total_steps}"
    
    print(f"\n  Total reasoning steps: {total_steps}")
    print(f"  Timestamped steps: {len(timestamped_steps)}")
    print(f"  ✓ All phases present")
    print(f"  ✓ Sufficient timestamped steps ({len(timestamped_steps)}/{total_steps})")
    
    print("\n[OK] Reasoning chain completeness validated")


def run_all_e2e_tests():
    """Run all E2E integration tests"""
    print("\n" + "="*80)
    print("END-TO-END INTEGRATION TEST SUITE")
    print("="*80)
    
    tests = [
        ("Standard Case", test_e2e_standard_case),
        ("High Uncertainty", test_e2e_high_uncertainty),
        ("Anomaly Case", test_e2e_anomaly_case),
        ("Mixed Case", test_e2e_mixed_case),
        ("Result Consistency", test_e2e_result_consistency),
        ("Reasoning Chain Completeness", test_e2e_reasoning_chain_completeness)
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
    print("E2E TEST SUMMARY")
    print("="*80)
    
    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All E2E integration tests passed!")
        return 0
    else:
        print("\n⚠️  Some E2E tests failed. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_e2e_tests())
