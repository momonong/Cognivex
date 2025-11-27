#!/usr/bin/env python3
"""
Debug script for Agent A orchestration issue
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*80)
print("AGENT A DEBUG TEST")
print("="*80)

# Test 1: Import modules
print("\n[TEST 1] Importing modules...")
try:
    from app.agents.cdda_agent import CDDAAgent
    print("✓ CDDAAgent imported")
    
    from app.core.mcp_server import DiagnosticMCPServer
    print("✓ DiagnosticMCPServer imported")
    
    from app.core.ml_processing.cdda_tools import CDDAToolKit
    print("✓ CDDAToolKit imported")
    
    from app.core.models.context_models import DiagnosticReport
    print("✓ DiagnosticReport imported")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize toolkit
print("\n[TEST 2] Initializing CDDAToolKit...")
try:
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_MCI_AD.joblib",
        data_root="data/MRI_processed"
    )
    print("✓ CDDAToolKit initialized")
except Exception as e:
    print(f"✗ Toolkit initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Get diagnostic report directly
print("\n[TEST 3] Getting diagnostic report for sub-0003...")
try:
    report = toolkit.get_diagnostic_report("sub-0003", verbose=True)
    print(f"✓ Report retrieved")
    print(f"  - subject_id: {report.get('subject_id')}")
    print(f"  - prediction: {report.get('prediction_result')}")
    print(f"  - confidence: {report.get('confidence'):.3f}")
    print(f"  - uq_score: {report.get('uq_score'):.3f}")
except Exception as e:
    print(f"✗ Failed to get report: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Convert to DiagnosticReport object
print("\n[TEST 4] Converting to DiagnosticReport object...")
try:
    diagnostic_report = DiagnosticReport.from_toolkit_report(report)
    print(f"✓ DiagnosticReport created")
    print(f"  - subject_id: {diagnostic_report.subject_id}")
    print(f"  - prediction: {diagnostic_report.prediction_result}")
    print(f"  - confidence: {diagnostic_report.confidence:.3f}")
    print(f"  - top_features count: {len(diagnostic_report.top_features)}")
except Exception as e:
    print(f"✗ Failed to create DiagnosticReport: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Initialize MCP Server
print("\n[TEST 5] Initializing MCP Server...")
try:
    from app.core.knowledge.graph_rag import GraphRAG
    
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(
        toolkit=toolkit,
        graph_rag=graph_rag,
        verbose=True
    )
    print("✓ MCP Server initialized")
except Exception as e:
    print(f"✗ MCP Server initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Read resource via MCP
print("\n[TEST 6] Reading resource via MCP...")
try:
    uri = "diagnosis://sub-0003/report"
    result = mcp_server.read_resource(uri)
    
    if 'error' in result:
        print(f"✗ MCP returned error: {result['error']}")
    else:
        print(f"✓ Resource read successfully")
        print(f"  - Keys: {list(result.keys())}")
        print(f"  - subject_id: {result.get('subject_id')}")
        print(f"  - prediction: {result.get('prediction_result')}")
except Exception as e:
    print(f"✗ Failed to read resource: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Initialize Agent A
print("\n[TEST 7] Initializing Agent A...")
try:
    from app.agents.agent_a_orchestrator import AgentA, AgentAConfig
    
    config = AgentAConfig(
        use_llm=False,  # Use rule-based for testing
        verbose=True
    )
    
    agent_a = AgentA(
        mcp_server=mcp_server,
        config=config
    )
    print("✓ Agent A initialized")
except Exception as e:
    print(f"✗ Agent A initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Orchestrate
print("\n[TEST 8] Running orchestration...")
try:
    context_object = agent_a.orchestrate("sub-0003")
    print(f"✓ Orchestration completed")
    print(f"  - subject_id: {context_object.subject_id}")
    print(f"  - decision: {context_object.decision_rationale}")
    print(f"  - prediction: {context_object.diagnostic_report.prediction_result}")
except Exception as e:
    print(f"✗ Orchestration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
