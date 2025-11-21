#!/usr/bin/env python3
"""
Demo Script: MCP Server Interface

This script demonstrates the Model Context Protocol (MCP) server interface,
showing how resources and tools are accessed through the MCP protocol.

Usage:
    python scripts/demo_mcp_server.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.mcp_server import DiagnosticMCPServer
from app.core.ml_processing.cdda_tools import CDDAToolKit
from app.core.knowledge.graph_rag import GraphRAG


def demo_mcp_resources():
    """Demonstrate MCP resource access"""
    print("\n" + "="*80)
    print("DEMO 1: MCP RESOURCES - Read-Only Data Access")
    print("="*80)
    
    # Initialize MCP server
    print("\n[INIT] Initializing DiagnosticMCPServer...")
    server = DiagnosticMCPServer(verbose=False)
    print("[OK] MCP Server ready")
    
    # List available resources
    print("\n" + "-"*80)
    print("STEP 1: List Available Resources")
    print("-"*80)
    resources = server.list_resources()
    for i, resource in enumerate(resources, 1):
        print(f"\n{i}. {resource.name}")
        print(f"   URI: {resource.uri}")
        print(f"   Description: {resource.description}")
        print(f"   MIME Type: {resource.mime_type}")
    
    # Read diagnostic report
    print("\n" + "-"*80)
    print("STEP 2: Read Diagnostic Report Resource")
    print("-"*80)
    subject_id = "sub-0005"
    uri = f"diagnosis://{subject_id}/report"
    print(f"Reading: {uri}")
    
    report_response = server.read_resource(uri)
    report = report_response['data']
    
    print(f"\n✓ Resource retrieved successfully")
    print(f"  Subject ID: {report['subject_id']}")
    print(f"  Prediction: {report['prediction_result']}")
    print(f"  Confidence: {report['confidence']:.1%}")
    print(f"  UQ Score: {report['uq_score']:.3f}")
    print(f"  Anomaly Status: {report['anomaly_status']['has_anomaly']}")
    
    if report['top_features']:
        print(f"\n  Top 3 Contributing Features:")
        for feat in report['top_features'][:3]:
            print(f"    - {feat['roi_name']}: Z={feat['z_score']:.2f}, SHAP={feat['shap_value']:.3f}")
    
    # Read knowledge context
    print("\n" + "-"*80)
    print("STEP 3: Read Knowledge Context Resource")
    print("-"*80)
    region_name = "Hippocampus_L"
    uri = f"knowledge://{region_name}/context"
    print(f"Reading: {uri}")
    
    knowledge_response = server.read_resource(uri)
    context = knowledge_response['data']['context']
    
    print(f"\n✓ Resource retrieved successfully")
    print(f"  Region: {context.get('full_name', region_name)}")
    print(f"  Function: {context.get('function', context.get('summary', 'N/A'))[:100]}...")
    print(f"  Clinical Significance: {context.get('clinical_significance', 'N/A')[:100]}...")
    print(f"  AD Hotspot: {context.get('is_ad_hotspot', False)}")
    
    if context.get('related_conditions'):
        print(f"  Related Conditions: {', '.join(context['related_conditions'][:3])}")
    
    # Check for fallback
    if knowledge_response['data'].get('fallback'):
        print(f"\n  ⚠️  Note: Using fallback knowledge base (GraphRAG unavailable)")
    
    print("\n" + "="*80)
    print("DEMO 1 COMPLETE: MCP Resources")
    print("="*80)


def demo_mcp_tools():
    """Demonstrate MCP tool execution"""
    print("\n" + "="*80)
    print("DEMO 2: MCP TOOLS - Executable Actions")
    print("="*80)
    
    # Initialize MCP server
    print("\n[INIT] Initializing DiagnosticMCPServer...")
    server = DiagnosticMCPServer(verbose=False)
    print("[OK] MCP Server ready")
    
    # List available tools
    print("\n" + "-"*80)
    print("STEP 1: List Available Tools")
    print("-"*80)
    tools = server.list_tools()
    for i, tool in enumerate(tools, 1):
        print(f"\n{i}. {tool.name}")
        print(f"   Description: {tool.description}")
        print(f"   Input Schema:")
        print(f"     Required: {tool.input_schema.get('required', [])}")
        for prop_name, prop_schema in tool.input_schema.get('properties', {}).items():
            print(f"     - {prop_name}: {prop_schema.get('type')} - {prop_schema.get('description')}")
    
    # Execute counterfactual simulation
    print("\n" + "-"*80)
    print("STEP 2: Execute Counterfactual Simulation Tool")
    print("-"*80)
    subject_id = "sub-0005"
    features_to_mask = ["Hippocampus_L", "Hippocampus_R", "Amygdala_L"]
    
    print(f"Tool: simulate_counterfactual")
    print(f"Arguments:")
    print(f"  subject_id: {subject_id}")
    print(f"  features_to_mask: {features_to_mask}")
    
    result = server.call_tool(
        "simulate_counterfactual",
        {
            "subject_id": subject_id,
            "features_to_mask": features_to_mask
        }
    )
    
    if result['status'] == 'success':
        data = result['data']
        print(f"\n✓ Tool executed successfully")
        print(f"\n  ORIGINAL PREDICTION:")
        print(f"    Diagnosis: {data['original_prediction']}")
        print(f"    Confidence: {data['original_confidence']:.1%}")
        
        print(f"\n  COUNTERFACTUAL PREDICTION (after masking):")
        print(f"    Diagnosis: {data['new_prediction']}")
        print(f"    Confidence: {data['new_confidence']:.1%}")
        
        print(f"\n  IMPACT ANALYSIS:")
        print(f"    Confidence Delta: {data['confidence_delta']:+.1%}")
        
        if abs(data['confidence_delta']) > 0.1:
            print(f"    → SIGNIFICANT IMPACT: These features are key diagnostic drivers")
        else:
            print(f"    → MINIMAL IMPACT: These features are not primary drivers")
        
        print(f"\n  MASKED FEATURES:")
        for feat in data['masked_features']:
            print(f"    - {feat['roi_name']}: {feat['original_value']:.1f} → {feat['masked_value']:.1f}")
        
        print(f"\n  INTERPRETATION:")
        print(f"    {data['interpretation']}")
    else:
        print(f"\n✗ Tool execution failed")
        print(f"  Error: {result['error']['message']}")
    
    print("\n" + "="*80)
    print("DEMO 2 COMPLETE: MCP Tools")
    print("="*80)


def demo_mcp_error_handling():
    """Demonstrate MCP error handling"""
    print("\n" + "="*80)
    print("DEMO 3: MCP ERROR HANDLING")
    print("="*80)
    
    server = DiagnosticMCPServer(verbose=False)
    
    # Test invalid URI
    print("\n" + "-"*80)
    print("STEP 1: Invalid Resource URI")
    print("-"*80)
    try:
        server.read_resource("invalid://test/resource")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test invalid tool name
    print("\n" + "-"*80)
    print("STEP 2: Invalid Tool Name")
    print("-"*80)
    try:
        server.call_tool("invalid_tool", {})
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test missing tool arguments
    print("\n" + "-"*80)
    print("STEP 3: Missing Tool Arguments")
    print("-"*80)
    try:
        server.call_tool("simulate_counterfactual", {"subject_id": "sub-0005"})
        print("✗ Should have raised KeyError")
    except KeyError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\n" + "="*80)
    print("DEMO 3 COMPLETE: Error Handling")
    print("="*80)


def main():
    """Run all MCP server demos"""
    print("\n" + "="*80)
    print("MCP SERVER DEMONSTRATION")
    print("Model Context Protocol - Diagnostic System")
    print("="*80)
    print("\nThis demo shows how the MCP server provides:")
    print("  1. RESOURCES: Read-only data (diagnostic reports, knowledge)")
    print("  2. TOOLS: Executable actions (counterfactual simulation)")
    print("  3. ERROR HANDLING: Graceful error management")
    
    try:
        # Run demos
        demo_mcp_resources()
        print("\n\n")
        demo_mcp_tools()
        print("\n\n")
        demo_mcp_error_handling()
        
        print("\n" + "="*80)
        print("ALL DEMOS COMPLETE")
        print("="*80)
        print("\nKey Takeaways:")
        print("  ✓ MCP separates read-only data (Resources) from actions (Tools)")
        print("  ✓ Resources use URI-based access (diagnosis://, knowledge://)")
        print("  ✓ Tools use name-based invocation with validated arguments")
        print("  ✓ Error handling ensures graceful degradation")
        print("\nNext: See demo_a2a_agents.py for Agent-to-Agent handoff")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
