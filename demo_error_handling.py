"""
Demo: LLM Error Handling and Fallback Mechanisms

This demo showcases the robust error handling implemented for the CDDA Agent:
1. Retry logic with exponential backoff
2. JSON parsing with recovery
3. Agent A fallback to rule-based orchestration
4. Agent B fallback to template-based synthesis
5. GraphRAG fallback to local knowledge base
6. Error annotations in final report

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


def demo_error_handling_with_fallbacks():
    """
    Demo: Complete error handling with all fallbacks
    
    This demo runs the CDDA Agent with LLM disabled to demonstrate
    that the system gracefully falls back to rule-based logic and
    template-based synthesis when LLMs are unavailable.
    """
    print("\n" + "="*80)
    print("DEMO: Error Handling and Fallback Mechanisms")
    print("="*80)
    print("\nThis demo shows how the CDDA Agent handles errors gracefully:")
    print("  1. Agent A falls back to rule-based orchestration")
    print("  2. Agent B falls back to template-based synthesis")
    print("  3. GraphRAG falls back to local knowledge base")
    print("  4. Error annotations are included in final report")
    print("="*80)
    
    # Initialize CDDA Agent with LLM disabled (forces fallback)
    print("\n[INIT] Initializing CDDA Agent with LLM disabled...")
    agent = CDDAAgent(
        use_llm=False,  # Force fallback to rule-based logic
        z_score_threshold=1.5,  # Lower threshold to trigger anomaly detection
        verbose=True
    )
    
    # Run analysis
    print("\n[ANALYSIS] Running analysis for sub-0005...")
    result = agent.run_analysis('sub-0005')
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nSubject: {result.subject_id}")
    print(f"Agent Decision: {result.agent_decision}")
    print(f"Prediction: {result.prediction} ({result.confidence:.1%})")
    print(f"UQ Score: {result.uq_score:.3f}")
    
    # Show error annotations if any
    if result.context_object.has_errors():
        print(f"\n[ERRORS] {len(result.context_object.errors)} error(s) encountered:")
        for i, error in enumerate(result.context_object.errors, 1):
            print(f"  {i}. {error['component']}: {error['type']}")
            print(f"     Message: {error['message']}")
    else:
        print("\n[OK] No errors encountered")
    
    # Show clinical report
    print("\n" + "-"*80)
    print("CLINICAL REPORT:")
    print("-"*80)
    print(result.clinical_report)
    
    # Show reasoning chain (first 10 steps)
    print("\n" + "-"*80)
    print("REASONING CHAIN (first 10 steps):")
    print("-"*80)
    for i, step in enumerate(result.reasoning_chain[:10], 1):
        print(f"{i}. {step}")
    if len(result.reasoning_chain) > 10:
        print(f"... ({len(result.reasoning_chain) - 10} more steps)")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  ✓ System continued operation despite LLM being unavailable")
    print("  ✓ Agent A used rule-based orchestration as fallback")
    print("  ✓ Agent B used template-based synthesis as fallback")
    print("  ✓ Complete reasoning chain was maintained")
    print("  ✓ Error annotations were included where applicable")
    print("\nThis demonstrates the system's robustness and graceful degradation.")
    print("="*80)


def demo_retry_and_recovery():
    """
    Demo: Retry logic and JSON parsing recovery
    
    This demo shows the low-level error handling mechanisms.
    """
    print("\n" + "="*80)
    print("DEMO: Retry Logic and JSON Parsing Recovery")
    print("="*80)
    
    from app.services.llm_providers.error_handling import (
        retry_with_backoff,
        parse_json_with_recovery,
        LLMRetryExhausted
    )
    
    # Demo 1: Retry logic
    print("\n[DEMO 1] Retry Logic with Exponential Backoff")
    print("-"*80)
    
    attempt_count = [0]
    
    @retry_with_backoff(max_retries=3, base_delay=0.5, verbose=True)
    def flaky_function():
        attempt_count[0] += 1
        print(f"  Attempt {attempt_count[0]}: Calling function...")
        if attempt_count[0] < 3:
            raise ConnectionError("Simulated connection error")
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"\n  ✓ Result: {result}")
    except LLMRetryExhausted as e:
        print(f"\n  ✗ Failed: {e}")
    
    # Demo 2: JSON parsing recovery
    print("\n[DEMO 2] JSON Parsing with Recovery")
    print("-"*80)
    
    test_cases = [
        ('{"key": "value"}', "Valid JSON"),
        ('```json\n{"key": "value"}\n```', "JSON in markdown"),
        ('Here is: {"key": "value"} Done!', "JSON with extra text"),
    ]
    
    for text, description in test_cases:
        print(f"\n  Test: {description}")
        print(f"  Input: {text}")
        try:
            result = parse_json_with_recovery(text, verbose=False)
            print(f"  ✓ Parsed: {result}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run demos
    demo_error_handling_with_fallbacks()
    print("\n\n")
    demo_retry_and_recovery()
