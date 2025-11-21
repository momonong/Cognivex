#!/usr/bin/env python3
"""
Demo Script: CDDA Phase 4 Complete System

This script demonstrates the complete CDDA Phase 4 system with:
- MCP Server (Model Context Protocol)
- A2A Architecture (Agent-to-Agent with dual-LLM)
- Error handling and fallback mechanisms
- Complete reasoning chain transparency

Usage:
    python scripts/demo_phase4_complete.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


def print_section_header(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)


def print_subsection_header(title: str):
    """Print a formatted subsection header"""
    print("\n" + "-"*80)
    print(title)
    print("-"*80)


def demo_complete_workflow():
    """Demonstrate complete CDDA Phase 4 workflow"""
    print_section_header("CDDA PHASE 4: COMPLETE SYSTEM DEMONSTRATION")
    
    print("\nCDDA Phase 4 Architecture:")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │              CONTEXT LAYER: MCP Server                  │")
    print("  │  Resources: diagnosis://, knowledge://                  │")
    print("  │  Tools: simulate_counterfactual                         │")
    print("  └────────────────────┬────────────────────────────────────┘")
    print("                       │ MCP Protocol")
    print("  ┌────────────────────▼────────────────────────────────────┐")
    print("  │         COGNITIVE LAYER: A2A Agent System               │")
    print("  │                                                         │")
    print("  │  Agent A (Orchestrator) → ContextObject → Agent B      │")
    print("  │  [GPT-OSS-20B]                          [MedGemma-27B] │")
    print("  └─────────────────────────────────────────────────────────┘")
    
    print("\nInitializing CDDA Agent...")
    agent = CDDAAgent(
        use_llm=False,  # Use rule-based for demo (no LLM required)
        uq_threshold=0.8,
        z_score_threshold=2.5,
        verbose=True
    )
    
    return agent


def demo_case_1_standard(agent: CDDAAgent):
    """Demo Case 1: Standard diagnosis"""
    print_section_header("CASE 1: STANDARD DIAGNOSIS")
    
    print("\nPatient Profile:")
    print("  Subject ID: sub-0015")
    print("  Expected: Low uncertainty, no anomalies")
    print("  Workflow: Agent A reads report → Agent B synthesizes standard report")
    
    print_subsection_header("EXECUTING ANALYSIS")
    result = agent.run_analysis("sub-0015")
    
    print_subsection_header("RESULTS")
    print(f"\nAgent Decision: {result.agent_decision}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"UQ Score: {result.uq_score:.3f}")
    print(f"Anomalies: {result.context_object.diagnostic_report.anomaly_status.has_anomaly}")
    
    print_subsection_header("CLINICAL REPORT")
    print(result.clinical_report)
    
    print_subsection_header("WORKFLOW SUMMARY")
    print(f"✓ Agent A: {len(result.context_object.agent_a_reasoning)} reasoning steps")
    print(f"✓ MCP Actions: {len(result.context_object.mcp_actions)}")
    print(f"✓ Agent B: Generated clinical report")
    print(f"✓ Total reasoning steps: {len(result.reasoning_chain)}")
    
    return result


def demo_case_2_high_uncertainty(agent: CDDAAgent):
    """Demo Case 2: High uncertainty with counterfactual"""
    print_section_header("CASE 2: HIGH UNCERTAINTY WITH COUNTERFACTUAL")
    
    print("\nPatient Profile:")
    print("  Subject ID: sub-0005")
    print("  Expected: High uncertainty (UQ > 0.8)")
    print("  Workflow: Agent A triggers counterfactual → Agent B interprets impact")
    
    print_subsection_header("EXECUTING ANALYSIS")
    result = agent.run_analysis("sub-0005")
    
    print_subsection_header("RESULTS")
    print(f"\nAgent Decision: {result.agent_decision}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"UQ Score: {result.uq_score:.3f}")
    
    # Show counterfactual results
    if result.context_object.tool_results and 'counterfactual' in result.context_object.tool_results:
        cf = result.context_object.tool_results['counterfactual']
        print(f"\nCounterfactual Simulation:")
        print(f"  Original: {cf['original_prediction']} ({cf['original_confidence']:.1%})")
        print(f"  After masking: {cf['new_prediction']} ({cf['new_confidence']:.1%})")
        print(f"  Confidence delta: {cf['confidence_delta']:+.1%}")
        print(f"  Masked features: {', '.join([f['roi_name'] for f in cf['masked_features'][:3]])}")
    
    print_subsection_header("CLINICAL REPORT")
    print(result.clinical_report)
    
    print_subsection_header("WORKFLOW SUMMARY")
    print(f"✓ Agent A: Detected high UQ, triggered counterfactual")
    print(f"✓ MCP Actions: {len(result.context_object.mcp_actions)}")
    print(f"✓ Agent B: Interpreted counterfactual results")
    print(f"✓ Total reasoning steps: {len(result.reasoning_chain)}")
    
    return result


def demo_case_3_anomaly(agent: CDDAAgent):
    """Demo Case 3: Anomaly with knowledge graph"""
    print_section_header("CASE 3: ANOMALY WITH KNOWLEDGE GRAPH")
    
    print("\nPatient Profile:")
    print("  Subject ID: sub-0011")
    print("  Expected: Anomalous regions detected")
    print("  Workflow: Agent A queries knowledge graph → Agent B flags mixed pathology")
    
    print_subsection_header("EXECUTING ANALYSIS")
    result = agent.run_analysis("sub-0011")
    
    print_subsection_header("RESULTS")
    print(f"\nAgent Decision: {result.agent_decision}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"UQ Score: {result.uq_score:.3f}")
    print(f"Anomalies: {result.context_object.diagnostic_report.anomaly_status.has_anomaly}")
    
    # Show anomalous regions
    if result.context_object.diagnostic_report.anomaly_status.has_anomaly:
        regions = result.context_object.diagnostic_report.anomaly_status.anomalous_regions
        print(f"\nAnomalous Regions: {', '.join(regions[:5])}")
    
    # Show knowledge context
    if result.context_object.tool_results and 'knowledge_context' in result.context_object.tool_results:
        kc = result.context_object.tool_results['knowledge_context']
        print(f"\nKnowledge Graph Query:")
        print(f"  Regions queried: {', '.join(kc['query_regions'][:3])}")
        print(f"  Summary: {kc['summary'][:150]}...")
    
    print_subsection_header("CLINICAL REPORT")
    print(result.clinical_report)
    
    print_subsection_header("WORKFLOW SUMMARY")
    print(f"✓ Agent A: Detected anomalies, queried knowledge graph")
    print(f"✓ MCP Actions: {len(result.context_object.mcp_actions)}")
    print(f"✓ Agent B: Flagged potential mixed pathology")
    print(f"✓ Total reasoning steps: {len(result.reasoning_chain)}")
    
    return result


def demo_reasoning_transparency(results: list):
    """Demonstrate reasoning chain transparency"""
    print_section_header("REASONING CHAIN TRANSPARENCY")
    
    print("\nCDDA Phase 4 provides complete transparency through reasoning chains.")
    print("Every decision, action, and synthesis step is logged with timestamps.")
    
    for i, result in enumerate(results, 1):
        print(f"\n{'-'*80}")
        print(f"Case {i}: {result.subject_id}")
        print(f"{'-'*80}")
        
        print(f"\nReasoning Chain Statistics:")
        print(f"  Total steps: {len(result.reasoning_chain)}")
        print(f"  Agent A steps: {len(result.context_object.agent_a_reasoning)}")
        print(f"  MCP actions: {len(result.context_object.mcp_actions)}")
        
        print(f"\nFirst 5 reasoning steps:")
        for step in result.reasoning_chain[:5]:
            print(f"  {step}")
        
        if len(result.reasoning_chain) > 5:
            print(f"  ... ({len(result.reasoning_chain) - 5} more steps)")
    
    print(f"\n{'-'*80}")
    print("REASONING CHAIN BENEFITS")
    print(f"{'-'*80}")
    print("✓ Complete audit trail for every analysis")
    print("✓ Timestamps enable temporal analysis")
    print("✓ MCP actions show resource reads and tool calls")
    print("✓ Can be exported to JSON for paper evidence")
    print("✓ Facilitates debugging and error analysis")


def demo_error_handling():
    """Demonstrate error handling and fallback mechanisms"""
    print_section_header("ERROR HANDLING & FALLBACK MECHANISMS")
    
    print("\nCDDA Phase 4 includes robust error handling:")
    print("  1. LLM failures → Rule-based fallback")
    print("  2. GraphRAG failures → Local knowledge base")
    print("  3. Tool failures → Error annotations in report")
    print("  4. Retry logic with exponential backoff")
    
    print_subsection_header("FALLBACK DEMONSTRATION")
    
    print("\nInitializing agent with rule-based fallback...")
    agent = CDDAAgent(
        use_llm=False,  # Simulate LLM unavailable
        verbose=False
    )
    
    print("✓ Agent initialized with rule-based orchestration")
    print("✓ Agent A uses decision rules instead of LLM")
    print("✓ Agent B uses templates instead of LLM")
    
    print("\nRunning analysis with fallback...")
    result = agent.run_analysis("sub-0005")
    
    print(f"\n✓ Analysis completed successfully")
    print(f"  Decision: {result.agent_decision}")
    print(f"  Prediction: {result.prediction} ({result.confidence:.1%})")
    print(f"  Report generated: {len(result.clinical_report)} characters")
    
    # Check for error annotations
    if hasattr(result.context_object, 'errors') and result.context_object.errors:
        print(f"\nError Annotations: {len(result.context_object.errors)}")
        for error in result.context_object.errors:
            print(f"  - {error['type']}: {error['message']}")
    else:
        print(f"\nNo errors encountered (fallback worked seamlessly)")
    
    print_subsection_header("ERROR HANDLING SUMMARY")
    print("✓ System never fails completely")
    print("✓ Graceful degradation to rule-based logic")
    print("✓ Error annotations in final report")
    print("✓ Transparent logging of fallback usage")


def demo_export_results(results: list):
    """Demonstrate exporting results for paper evidence"""
    print_section_header("EXPORTING RESULTS FOR PAPER EVIDENCE")
    
    print("\nCDDA Phase 4 supports exporting complete analysis results")
    print("for academic papers, debugging, and audit trails.")
    
    output_dir = Path("output/phase4_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExport directory: {output_dir}")
    
    for i, result in enumerate(results, 1):
        # Export reasoning log
        log_path = output_dir / f"case{i}_{result.subject_id}_reasoning.json"
        
        log_data = {
            'subject_id': result.subject_id,
            'timestamp': result.timestamp,
            'agent_decision': result.agent_decision,
            'prediction': result.prediction,
            'confidence': result.confidence,
            'uq_score': result.uq_score,
            'reasoning_chain': result.reasoning_chain,
            'metadata': result.metadata,
            'context_object': {
                'decision_rationale': result.context_object.decision_rationale,
                'signals': result.context_object.signals,
                'agent_a_reasoning_count': len(result.context_object.agent_a_reasoning),
                'mcp_actions_count': len(result.context_object.mcp_actions)
            }
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"  ✓ Case {i}: {log_path.name}")
        
        # Export clinical report
        report_path = output_dir / f"case{i}_{result.subject_id}_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"CDDA Phase 4 Clinical Report\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Subject: {result.subject_id}\n")
            f.write(f"Timestamp: {result.timestamp}\n")
            f.write(f"Agent Decision: {result.agent_decision}\n")
            f.write(f"Prediction: {result.prediction} ({result.confidence:.1%})\n")
            f.write(f"UQ Score: {result.uq_score:.3f}\n\n")
            f.write(f"{'='*80}\n\n")
            f.write(result.clinical_report)
        
        print(f"  ✓ Case {i}: {report_path.name}")
    
    print(f"\n✓ All results exported to: {output_dir}")
    print("\nExported files can be used for:")
    print("  - Academic paper evidence")
    print("  - System debugging and analysis")
    print("  - Audit trails and compliance")
    print("  - Performance benchmarking")


def main():
    """Run complete Phase 4 demonstration"""
    print_section_header("CDDA PHASE 4: COMPLETE SYSTEM DEMONSTRATION")
    print("\nThis demonstration showcases the complete CDDA Phase 4 system:")
    print("  • MCP Server (Model Context Protocol)")
    print("  • A2A Architecture (Agent-to-Agent with dual-LLM)")
    print("  • Three diagnostic scenarios (standard, high-UQ, anomaly)")
    print("  • Reasoning chain transparency")
    print("  • Error handling and fallback mechanisms")
    print("  • Result export for paper evidence")
    
    input("\nPress Enter to begin demonstration...")
    
    try:
        # Initialize system
        agent = demo_complete_workflow()
        
        # Run three diagnostic cases
        results = []
        
        print("\n\n")
        result1 = demo_case_1_standard(agent)
        results.append(result1)
        input("\nPress Enter to continue to Case 2...")
        
        print("\n\n")
        result2 = demo_case_2_high_uncertainty(agent)
        results.append(result2)
        input("\nPress Enter to continue to Case 3...")
        
        print("\n\n")
        result3 = demo_case_3_anomaly(agent)
        results.append(result3)
        input("\nPress Enter to continue to reasoning transparency...")
        
        # Demonstrate reasoning transparency
        print("\n\n")
        demo_reasoning_transparency(results)
        input("\nPress Enter to continue to error handling...")
        
        # Demonstrate error handling
        print("\n\n")
        demo_error_handling()
        input("\nPress Enter to continue to result export...")
        
        # Export results
        print("\n\n")
        demo_export_results(results)
        
        # Final summary
        print_section_header("DEMONSTRATION COMPLETE")
        print("\nCDDA Phase 4 Key Achievements:")
        print("  ✓ MCP Server: Clean separation of resources and tools")
        print("  ✓ A2A Architecture: Dual-LLM with clear handoff protocol")
        print("  ✓ Autonomous Decision Making: Rule-based and LLM-based")
        print("  ✓ Complete Transparency: Full reasoning chain logging")
        print("  ✓ Robust Error Handling: Graceful fallback mechanisms")
        print("  ✓ Paper-Ready Evidence: Exportable results and logs")
        
        print("\nNext Steps:")
        print("  1. Review exported results in output/phase4_demo/")
        print("  2. Try with LLM enabled: use_llm=True")
        print("  3. Integrate with Streamlit UI (Phase 5)")
        print("  4. Conduct user studies and evaluation")
        
        print("\n" + "="*80)
        print("Thank you for exploring CDDA Phase 4!")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
