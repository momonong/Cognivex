#!/usr/bin/env python3
"""
Demo Script: Agent-to-Agent (A2A) Handoff Protocol

This script demonstrates the dual-LLM A2A architecture where:
- Agent A (Orchestrator) reads resources, invokes tools, compiles context
- Agent B (Consultant) synthesizes clinical reports from provided context
- Handoff via ContextObject ensures separation of concerns

Usage:
    python scripts/demo_a2a_agents.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


def demo_standard_case():
    """Demo: Standard case (low uncertainty, no anomalies)"""
    print("\n" + "="*80)
    print("DEMO 1: A2A STANDARD CASE")
    print("="*80)
    print("\nScenario: Low uncertainty, no anomalies")
    print("Expected: Agent A reads diagnostic report → Agent B synthesizes standard report")
    
    # Initialize agent with rule-based fallback (no LLM required for demo)
    print("\n[INIT] Initializing CDDA Agent (A2A mode, rule-based)...")
    agent = CDDAAgent(
        use_llm=False,  # Use rule-based orchestration for demo
        verbose=True
    )
    
    # Run analysis
    print("\n" + "-"*80)
    print("EXECUTING A2A WORKFLOW")
    print("-"*80)
    subject_id = "sub-0015"  # Known standard case
    result = agent.run_analysis(subject_id)
    
    # Print results
    print("\n" + "-"*80)
    print("A2A WORKFLOW RESULTS")
    print("-"*80)
    print(f"\nAgent Decision: {result.agent_decision}")
    print(f"Prediction: {result.prediction} ({result.confidence:.1%})")
    print(f"UQ Score: {result.uq_score:.3f}")
    
    print("\n" + "-"*80)
    print("CLINICAL REPORT (from Agent B)")
    print("-"*80)
    print(result.clinical_report)
    
    print("\n" + "-"*80)
    print("REASONING CHAIN (Agent A + Agent B)")
    print("-"*80)
    for step in result.reasoning_chain[:10]:  # Show first 10 steps
        print(step)
    if len(result.reasoning_chain) > 10:
        print(f"... ({len(result.reasoning_chain) - 10} more steps)")
    
    print("\n" + "="*80)
    print("DEMO 1 COMPLETE: Standard Case")
    print("="*80)
    
    return result


def demo_high_uncertainty_case():
    """Demo: High uncertainty case (triggers counterfactual)"""
    print("\n" + "="*80)
    print("DEMO 2: A2A HIGH UNCERTAINTY CASE")
    print("="*80)
    print("\nScenario: High uncertainty (UQ > 0.8)")
    print("Expected: Agent A triggers counterfactual simulation → Agent B interprets results")
    
    # Initialize agent
    print("\n[INIT] Initializing CDDA Agent (A2A mode, rule-based)...")
    agent = CDDAAgent(
        use_llm=False,
        uq_threshold=0.8,
        verbose=True
    )
    
    # Run analysis
    print("\n" + "-"*80)
    print("EXECUTING A2A WORKFLOW")
    print("-"*80)
    subject_id = "sub-0005"  # Known high UQ case
    result = agent.run_analysis(subject_id)
    
    # Print results
    print("\n" + "-"*80)
    print("A2A WORKFLOW RESULTS")
    print("-"*80)
    print(f"\nAgent Decision: {result.agent_decision}")
    print(f"Prediction: {result.prediction} ({result.confidence:.1%})")
    print(f"UQ Score: {result.uq_score:.3f}")
    
    # Check if counterfactual was triggered
    if result.context_object.tool_results and 'counterfactual' in result.context_object.tool_results:
        cf = result.context_object.tool_results['counterfactual']
        print(f"\n✓ Counterfactual Simulation Triggered:")
        print(f"  Original: {cf['original_prediction']} ({cf['original_confidence']:.1%})")
        print(f"  Counterfactual: {cf['new_prediction']} ({cf['new_confidence']:.1%})")
        print(f"  Delta: {cf['confidence_delta']:+.1%}")
    
    print("\n" + "-"*80)
    print("CLINICAL REPORT (from Agent B)")
    print("-"*80)
    print(result.clinical_report)
    
    print("\n" + "-"*80)
    print("REASONING CHAIN (Agent A + Agent B)")
    print("-"*80)
    for step in result.reasoning_chain[:15]:  # Show first 15 steps
        print(step)
    if len(result.reasoning_chain) > 15:
        print(f"... ({len(result.reasoning_chain) - 15} more steps)")
    
    print("\n" + "="*80)
    print("DEMO 2 COMPLETE: High Uncertainty Case")
    print("="*80)
    
    return result


def demo_anomaly_case():
    """Demo: Anomaly case (triggers knowledge graph lookup)"""
    print("\n" + "="*80)
    print("DEMO 3: A2A ANOMALY CASE")
    print("="*80)
    print("\nScenario: Anomalous regions detected")
    print("Expected: Agent A queries knowledge graph → Agent B flags mixed pathology")
    
    # Initialize agent
    print("\n[INIT] Initializing CDDA Agent (A2A mode, rule-based)...")
    agent = CDDAAgent(
        use_llm=False,
        z_score_threshold=2.5,
        verbose=True
    )
    
    # Run analysis
    print("\n" + "-"*80)
    print("EXECUTING A2A WORKFLOW")
    print("-"*80)
    subject_id = "sub-0011"  # Known anomaly case
    result = agent.run_analysis(subject_id)
    
    # Print results
    print("\n" + "-"*80)
    print("A2A WORKFLOW RESULTS")
    print("-"*80)
    print(f"\nAgent Decision: {result.agent_decision}")
    print(f"Prediction: {result.prediction} ({result.confidence:.1%})")
    print(f"UQ Score: {result.uq_score:.3f}")
    print(f"Anomalies: {result.context_object.diagnostic_report.anomaly_status.has_anomaly}")
    
    # Check if knowledge graph was queried
    if result.context_object.tool_results and 'knowledge_context' in result.context_object.tool_results:
        kc = result.context_object.tool_results['knowledge_context']
        print(f"\n✓ Knowledge Graph Queried:")
        print(f"  Regions: {', '.join(kc['query_regions'][:3])}")
        print(f"  Summary: {kc['summary'][:150]}...")
    
    print("\n" + "-"*80)
    print("CLINICAL REPORT (from Agent B)")
    print("-"*80)
    print(result.clinical_report)
    
    print("\n" + "-"*80)
    print("REASONING CHAIN (Agent A + Agent B)")
    print("-"*80)
    for step in result.reasoning_chain[:15]:  # Show first 15 steps
        print(step)
    if len(result.reasoning_chain) > 15:
        print(f"... ({len(result.reasoning_chain) - 15} more steps)")
    
    print("\n" + "="*80)
    print("DEMO 3 COMPLETE: Anomaly Case")
    print("="*80)
    
    return result


def demo_context_object_structure():
    """Demo: ContextObject structure and validation"""
    print("\n" + "="*80)
    print("DEMO 4: CONTEXT OBJECT STRUCTURE")
    print("="*80)
    print("\nDemonstrating the ContextObject used for A2A handoff")
    
    # Run a quick analysis
    agent = CDDAAgent(use_llm=False, verbose=False)
    result = agent.run_analysis("sub-0005")
    
    context_obj = result.context_object
    
    print("\n" + "-"*80)
    print("CONTEXT OBJECT CONTENTS")
    print("-"*80)
    print(f"\nSubject ID: {context_obj.subject_id}")
    print(f"Timestamp: {context_obj.timestamp}")
    print(f"Decision Rationale: {context_obj.decision_rationale}")
    
    print(f"\nDiagnostic Report:")
    print(f"  Prediction: {context_obj.diagnostic_report.prediction_result}")
    print(f"  Confidence: {context_obj.diagnostic_report.confidence:.1%}")
    print(f"  UQ Score: {context_obj.diagnostic_report.uq_score:.3f}")
    print(f"  Top Features: {len(context_obj.diagnostic_report.top_features)}")
    
    print(f"\nSignals:")
    for key, value in context_obj.signals.items():
        print(f"  {key}: {value}")
    
    print(f"\nAgent A Reasoning Steps: {len(context_obj.agent_a_reasoning)}")
    print(f"MCP Actions: {len(context_obj.mcp_actions)}")
    
    if context_obj.tool_results:
        print(f"\nTool Results:")
        for tool_name in context_obj.tool_results.keys():
            print(f"  - {tool_name}")
    
    print(f"\nValidation: {context_obj.validate()}")
    
    if hasattr(context_obj, 'errors') and context_obj.errors:
        print(f"\nError Annotations: {len(context_obj.errors)}")
        for error in context_obj.errors:
            print(f"  - {error['type']}: {error['message']}")
    
    print("\n" + "-"*80)
    print("KEY PROPERTIES OF CONTEXT OBJECT")
    print("-"*80)
    print("✓ Contains ALL context needed for Agent B")
    print("✓ Agent B has NO direct access to MCP server or tools")
    print("✓ Ensures clear separation between orchestration and synthesis")
    print("✓ Includes complete reasoning chain from Agent A")
    print("✓ Tracks all MCP actions with timestamps")
    
    print("\n" + "="*80)
    print("DEMO 4 COMPLETE: ContextObject Structure")
    print("="*80)


def demo_reasoning_chain_aggregation():
    """Demo: Reasoning chain aggregation from both agents"""
    print("\n" + "="*80)
    print("DEMO 5: REASONING CHAIN AGGREGATION")
    print("="*80)
    print("\nDemonstrating how reasoning chains from Agent A and Agent B are combined")
    
    # Run analysis
    agent = CDDAAgent(use_llm=False, verbose=False)
    result = agent.run_analysis("sub-0005")
    
    print("\n" + "-"*80)
    print("COMPLETE REASONING CHAIN")
    print("-"*80)
    print(f"\nTotal Steps: {len(result.reasoning_chain)}")
    
    # Count steps by agent
    agent_a_steps = sum(1 for step in result.reasoning_chain if 'AGENT A' in step or 'ORCHESTRATION' in step)
    agent_b_steps = sum(1 for step in result.reasoning_chain if 'AGENT B' in step or 'SYNTHESIS' in step)
    mcp_steps = sum(1 for step in result.reasoning_chain if 'MCP' in step)
    
    print(f"  Agent A Steps: {agent_a_steps}")
    print(f"  Agent B Steps: {agent_b_steps}")
    print(f"  MCP Actions: {mcp_steps}")
    
    print("\n" + "-"*80)
    print("REASONING CHAIN STRUCTURE")
    print("-"*80)
    for step in result.reasoning_chain:
        print(step)
    
    print("\n" + "-"*80)
    print("REASONING CHAIN BENEFITS")
    print("-"*80)
    print("✓ Complete transparency: Every decision is logged")
    print("✓ Timestamps: Track when each action occurred")
    print("✓ MCP actions: Record all resource reads and tool calls")
    print("✓ Paper evidence: Can be saved to JSON for analysis")
    print("✓ Debugging: Easy to trace issues through the workflow")
    
    # Save reasoning log
    output_path = "output/demo_reasoning_chain.json"
    agent.save_reasoning_log(result, output_path)
    print(f"\n✓ Reasoning log saved to: {output_path}")
    
    print("\n" + "="*80)
    print("DEMO 5 COMPLETE: Reasoning Chain Aggregation")
    print("="*80)


def main():
    """Run all A2A agent demos"""
    print("\n" + "="*80)
    print("AGENT-TO-AGENT (A2A) DEMONSTRATION")
    print("Dual-LLM Architecture with Handoff Protocol")
    print("="*80)
    print("\nThis demo shows:")
    print("  1. STANDARD CASE: Low uncertainty, straightforward diagnosis")
    print("  2. HIGH UNCERTAINTY: Triggers counterfactual simulation")
    print("  3. ANOMALY CASE: Triggers knowledge graph lookup")
    print("  4. CONTEXT OBJECT: Structure and validation")
    print("  5. REASONING CHAIN: Aggregation from both agents")
    
    try:
        # Run demos
        demo_standard_case()
        print("\n\n")
        demo_high_uncertainty_case()
        print("\n\n")
        demo_anomaly_case()
        print("\n\n")
        demo_context_object_structure()
        print("\n\n")
        demo_reasoning_chain_aggregation()
        
        print("\n" + "="*80)
        print("ALL DEMOS COMPLETE")
        print("="*80)
        print("\nKey Takeaways:")
        print("  ✓ Agent A orchestrates: reads resources, invokes tools, compiles context")
        print("  ✓ Agent B synthesizes: generates clinical reports from context only")
        print("  ✓ ContextObject ensures Agent B has no direct tool access")
        print("  ✓ Reasoning chains provide complete transparency")
        print("  ✓ System handles standard, high-UQ, and anomaly cases")
        print("\nNext: See demo_phase4_complete.py for full system integration")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
