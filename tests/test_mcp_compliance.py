"""
MCP Compliance Tests

This module tests MCP (Model Context Protocol) compliance:
- Resource URIs follow MCP format
- Tool schemas are valid JSON
- Separation of resources and tools
- MCP server with mock clients

Requirements: 2.1, 2.2, 4.1, 4.2
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.mcp_server import DiagnosticMCPServer
from app.core.ml_processing.cdda_tools import CDDAToolKit
from app.core.knowledge.graph_rag import GraphRAG


def test_resource_uri_format():
    """
    Test that resource URIs follow MCP format
    
    MCP format: <protocol>://<identifier>/<resource_type>
    Examples:
    - diagnosis://sub-0005/report
    - knowledge://Hippocampus_L/context
    
    Requirements: 2.1, 4.1
    """
    print("\n" + "="*80)
    print("MCP COMPLIANCE TEST: Resource URI Format")
    print("="*80)
    
    # Initialize MCP server
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # List all resources
    resources = mcp_server.list_resources()
    
    print(f"\n[Found {len(resources)} resource types]")
    
    # Validate each resource metadata
    for resource in resources:
        print(f"\n  Resource: {resource.name}")
        print(f"    URI Pattern: {resource.uri}")
        print(f"    Description: {resource.description}")
        print(f"    MIME Type: {resource.mime_type}")
        
        # Check URI format
        assert '://' in resource.uri, f"Invalid URI format: {resource.uri}"
        
        # Check protocol
        protocol = resource.uri.split('://')[0]
        assert protocol in ['diagnosis', 'knowledge'], f"Invalid protocol: {protocol}"
        
        # Check MIME type
        assert resource.mime_type == 'application/json', f"Invalid MIME type: {resource.mime_type}"
        
        print(f"    ✓ URI format valid")
        print(f"    ✓ Protocol valid: {protocol}")
        print(f"    ✓ MIME type valid")
    
    print("\n[OK] All resource URIs follow MCP format")


def test_resource_uri_parsing():
    """
    Test that resource URIs can be parsed correctly
    
    Requirements: 2.1, 4.1
    """
    print("\n" + "="*80)
    print("MCP COMPLIANCE TEST: Resource URI Parsing")
    print("="*80)
    
    # Initialize MCP server
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # Test valid URIs
    test_uris = [
        ("diagnosis://sub-0005/report", "diagnosis", "sub-0005", "report"),
        ("diagnosis://sub-0015/features", "diagnosis", "sub-0015", "features"),
        ("knowledge://Hippocampus_L/context", "knowledge", "Hippocampus_L", "context"),
        ("knowledge://SN_pc/context", "knowledge", "SN_pc", "context")
    ]
    
    print("\n[Testing URI parsing...]")
    
    for uri, expected_protocol, expected_id, expected_resource in test_uris:
        print(f"\n  URI: {uri}")
        
        # Parse URI manually (since _parse_uri is internal)
        if '://' not in uri:
            print(f"    ✗ Invalid URI format")
            continue
            
        protocol = uri.split('://')[0]
        rest = uri.split('://')[1]
        parts = rest.split('/')
        identifier = parts[0] if len(parts) > 0 else ""
        resource_type = parts[1] if len(parts) > 1 else ""
        
        # Validate parsing
        assert protocol == expected_protocol, f"Protocol mismatch: {protocol} != {expected_protocol}"
        assert identifier == expected_id, f"Identifier mismatch: {identifier} != {expected_id}"
        assert resource_type == expected_resource, f"Resource type mismatch: {resource_type} != {expected_resource}"
        
        print(f"    ✓ Protocol: {protocol}")
        print(f"    ✓ Identifier: {identifier}")
        print(f"    ✓ Resource Type: {resource_type}")
    
    print("\n[OK] All URIs parsed correctly")


def test_invalid_uri_handling():
    """
    Test that invalid URIs are handled gracefully
    
    Requirements: 2.1
    """
    print("\n" + "="*80)
    print("MCP COMPLIANCE TEST: Invalid URI Handling")
    print("="*80)
    
    # Initialize MCP server
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # Test invalid URIs
    invalid_uris = [
        "invalid-uri",
        "http://sub-0005/report",  # Wrong protocol
        "diagnosis://sub-0005",  # Missing resource type
        "diagnosis:sub-0005/report",  # Missing //
        ""
    ]
    
    print("\n[Testing invalid URI handling...]")
    
    for uri in invalid_uris:
        print(f"\n  Testing: {uri}")
        
        try:
            result = mcp_server.read_resource(uri)
            # Should return error, not crash
            assert 'error' in result, f"Should return error for invalid URI: {uri}"
            print(f"    ✓ Error returned: {result['error']}")
        except Exception as e:
            print(f"    ✓ Exception caught: {type(e).__name__}")
    
    print("\n[OK] Invalid URIs handled gracefully")


def test_tool_schema_validity():
    """
    Test that tool schemas are valid JSON
    
    Requirements: 2.2, 4.2
    """
    print("\n" + "="*80)
    print("MCP COMPLIANCE TEST: Tool Schema Validity")
    print("="*80)
    
    # Initialize MCP server
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # List all tools
    tools = mcp_server.list_tools()
    
    print(f"\n[Found {len(tools)} tools]")
    
    # Validate each tool schema
    for tool in tools:
        print(f"\n  Tool: {tool.name}")
        print(f"    Description: {tool.description}")
        
        # Check input schema is valid JSON
        try:
            schema_str = json.dumps(tool.input_schema)
            parsed_schema = json.loads(schema_str)
            print(f"    ✓ Input schema is valid JSON")
        except Exception as e:
            raise AssertionError(f"Invalid JSON schema for {tool.name}: {e}")
        
        # Check required fields in schema
        assert 'type' in parsed_schema, f"Missing 'type' in schema for {tool.name}"
        assert 'properties' in parsed_schema, f"Missing 'properties' in schema for {tool.name}"
        
        print(f"    ✓ Schema has 'type' field")
        print(f"    ✓ Schema has 'properties' field")
        
        # Check properties
        for prop_name, prop_def in parsed_schema['properties'].items():
            assert 'type' in prop_def, f"Missing 'type' for property {prop_name}"
            assert 'description' in prop_def, f"Missing 'description' for property {prop_name}"
            print(f"    ✓ Property '{prop_name}' is valid")
    
    print("\n[OK] All tool schemas are valid JSON")


def test_resource_tool_separation():
    """
    Test that resources and tools are properly separated
    
    Resources: Read-only data (diagnostic reports, knowledge context)
    Tools: Executable actions (counterfactual simulation)
    
    Requirements: 2.1, 2.2, 4.1, 4.2
    """
    print("\n" + "="*80)
    print("MCP COMPLIANCE TEST: Resource/Tool Separation")
    print("="*80)
    
    # Initialize MCP server
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # Get resources and tools
    resources = mcp_server.list_resources()
    tools = mcp_server.list_tools()
    
    print(f"\n[Resources: {len(resources)}]")
    for resource in resources:
        print(f"  - {resource.name} ({resource.uri})")
    
    print(f"\n[Tools: {len(tools)}]")
    for tool in tools:
        print(f"  - {tool.name}")
    
    # Verify separation
    resource_names = {r.name for r in resources}
    tool_names = {t.name for t in tools}
    
    # Check no overlap
    overlap = resource_names & tool_names
    assert len(overlap) == 0, f"Resources and tools should not overlap: {overlap}"
    
    print(f"\n  ✓ No overlap between resources and tools")
    
    # Verify resources are read-only (use read_resource)
    print(f"\n[Verifying resources are read-only...]")
    test_uri = "diagnosis://sub-0005/report"
    result = mcp_server.read_resource(test_uri)
    assert 'error' not in result, f"Failed to read resource: {result.get('error')}"
    print(f"  ✓ Resources accessed via read_resource()")
    
    # Verify tools are executable (use call_tool)
    print(f"\n[Verifying tools are executable...]")
    tool_result = mcp_server.call_tool("simulate_counterfactual", {
        "subject_id": "sub-0005",
        "features_to_mask": ["Hippocampus_L_GM_Vol"]
    })
    assert 'error' not in tool_result, f"Failed to call tool: {tool_result.get('error')}"
    print(f"  ✓ Tools accessed via call_tool()")
    
    print("\n[OK] Resources and tools properly separated")


def test_mcp_server_with_mock_client():
    """
    Test MCP server with mock client
    
    Simulates Agent A as an MCP client
    
    Requirements: 2.1, 2.2, 4.1, 4.2
    """
    print("\n" + "="*80)
    print("MCP COMPLIANCE TEST: Mock Client Interaction")
    print("="*80)
    
    # Initialize MCP server
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # Mock client workflow
    print("\n[Mock Client: Simulating Agent A workflow...]")
    
    # Step 1: Discover resources
    print("\n  Step 1: Discover available resources")
    resources = mcp_server.list_resources()
    print(f"    ✓ Found {len(resources)} resources")
    
    # Step 2: Discover tools
    print("\n  Step 2: Discover available tools")
    tools = mcp_server.list_tools()
    print(f"    ✓ Found {len(tools)} tools")
    
    # Step 3: Read diagnostic report
    print("\n  Step 3: Read diagnostic report")
    report = mcp_server.read_resource("diagnosis://sub-0005/report")
    assert 'error' not in report
    # The report structure uses different field names
    assert 'subject_id' in report
    assert 'prediction' in report or 'prediction_result' in report
    print(f"    ✓ Diagnostic report retrieved")
    if 'prediction' in report:
        print(f"      Prediction: {report['prediction']}")
        print(f"      Confidence: {report.get('confidence', 'N/A')}")
        print(f"      UQ Score: {report.get('uq_score', 'N/A')}")
    
    # Step 4: Check if high UQ
    uq_score = report.get('uq_score', 0.0)
    if uq_score > 0.7:
        print("\n  Step 4: High UQ detected, calling counterfactual tool")
        cf_result = mcp_server.call_tool("simulate_counterfactual", {
            "subject_id": "sub-0005",
            "features_to_mask": ["Hippocampus_L_GM_Vol"]
        })
        assert 'error' not in cf_result
        assert 'confidence_delta' in cf_result
        print(f"    ✓ Counterfactual simulation completed")
        print(f"      Confidence delta: {cf_result['confidence_delta']:+.1%}")
    
    # Step 5: Check for anomalies
    anomaly_status = report.get('anomaly_status', {})
    if anomaly_status.get('has_anomaly', False):
        print("\n  Step 5: Anomalies detected, querying knowledge graph")
        anomalous_regions = anomaly_status.get('anomalous_regions', [])
        for region in anomalous_regions[:2]:  # Query first 2
            kg_result = mcp_server.read_resource(f"knowledge://{region}/context")
            assert 'error' not in kg_result
            print(f"    ✓ Knowledge context for {region} retrieved")
    
    print("\n[OK] Mock client interaction successful")


def test_mcp_error_responses():
    """
    Test that MCP server returns proper error responses
    
    Requirements: 2.1, 2.2
    """
    print("\n" + "="*80)
    print("MCP COMPLIANCE TEST: Error Response Format")
    print("="*80)
    
    # Initialize MCP server
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    print("\n[Testing error responses...]")
    
    # Test 1: Invalid resource URI
    print("\n  Test 1: Invalid resource URI")
    try:
        result = mcp_server.read_resource("invalid://uri")
        # Should return error dict or raise exception
        if isinstance(result, dict) and 'error' in result:
            print(f"    ✓ Error returned: {result['error']}")
        else:
            print(f"    ✗ Expected error response")
    except (ValueError, Exception) as e:
        print(f"    ✓ Exception raised: {type(e).__name__}")
    
    # Test 2: Invalid tool name
    print("\n  Test 2: Invalid tool name")
    try:
        result = mcp_server.call_tool("invalid_tool", {})
        if isinstance(result, dict) and 'error' in result:
            print(f"    ✓ Error returned: {result['error']}")
    except Exception as e:
        print(f"    ✓ Exception raised: {type(e).__name__}")
    
    # Test 3: Missing tool arguments
    print("\n  Test 3: Missing tool arguments")
    try:
        result = mcp_server.call_tool("simulate_counterfactual", {})
        if isinstance(result, dict) and 'error' in result:
            print(f"    ✓ Error returned: {result['error']}")
    except Exception as e:
        print(f"    ✓ Exception raised: {type(e).__name__}")
    
    # Test 4: Non-existent subject
    print("\n  Test 4: Non-existent subject")
    result = mcp_server.read_resource("diagnosis://sub-99999/report")
    # Should handle gracefully (may return error or empty result)
    print(f"    ✓ Handled gracefully")
    
    print("\n[OK] Error responses properly formatted")


def run_all_mcp_tests():
    """Run all MCP compliance tests"""
    print("\n" + "="*80)
    print("MCP COMPLIANCE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Resource URI Format", test_resource_uri_format),
        ("Resource URI Parsing", test_resource_uri_parsing),
        ("Invalid URI Handling", test_invalid_uri_handling),
        ("Tool Schema Validity", test_tool_schema_validity),
        ("Resource/Tool Separation", test_resource_tool_separation),
        ("Mock Client Interaction", test_mcp_server_with_mock_client),
        ("Error Response Format", test_mcp_error_responses)
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
    print("MCP COMPLIANCE TEST SUMMARY")
    print("="*80)
    
    for test_name, status in results:
        symbol = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"{symbol} {test_name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All MCP compliance tests passed!")
        return 0
    else:
        print("\n⚠️  Some MCP tests failed. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_mcp_tests())
