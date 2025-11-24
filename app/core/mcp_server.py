"""
CDDA Framework - MCP Server (Model Context Protocol)

This module implements the DiagnosticMCPServer following MCP principles:
- Resources: Read-only data (diagnostic reports, knowledge context)
- Tools: Executable actions (counterfactual simulation)

The MCP server provides a clean separation between context and action,
enabling Agent A (Orchestrator) to fetch data and invoke tools through
a standardized protocol.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.ml_processing.cdda_tools import CDDAToolKit
from app.core.knowledge.graph_rag import GraphRAG
from app.core.models import ResourceMetadata, ToolMetadata


# ============================================================================
# DiagnosticMCPServer Class
# ============================================================================

class DiagnosticMCPServer:
    """
    MCP-compliant server for diagnostic resources and tools
    
    This server wraps the existing CDDAToolKit and GraphRAG with
    MCP protocol, providing:
    
    RESOURCES (Read-Only Data):
    - diagnosis://{subject_id}/report - Full diagnostic report
    - diagnosis://{subject_id}/features - Raw feature values
    - knowledge://{region_name}/context - Clinical knowledge
    
    TOOLS (Executable Actions):
    - simulate_counterfactual - What-if analysis
    """
    
    def __init__(
        self,
        toolkit: Optional[CDDAToolKit] = None,
        graph_rag: Optional[GraphRAG] = None,
        verbose: bool = False
    ):
        """
        Initialize MCP Server
        
        Args:
            toolkit: CDDAToolKit instance (Layer 1+2)
            graph_rag: GraphRAG instance (Layer 4)
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        
        # Initialize toolkit if not provided
        if toolkit is None:
            if self.verbose:
                print("[MCP] Initializing CDDAToolKit...")
            self.toolkit = CDDAToolKit(
                model_path="model/cnn_rf/rf_model_NC_MCI_AD.joblib",
                data_root="data/MRI_processed"
            )
        else:
            self.toolkit = toolkit
        
        # Initialize GraphRAG if not provided
        if graph_rag is None:
            if self.verbose:
                print("[MCP] Initializing GraphRAG...")
            self.graph_rag = GraphRAG()
        else:
            self.graph_rag = graph_rag
        
        if self.verbose:
            print("[OK] DiagnosticMCPServer initialized")
    
    # ========================================================================
    # MCP Protocol Methods
    # ========================================================================
    
    def list_resources(self) -> List[ResourceMetadata]:
        """
        List all available MCP resources
        
        Returns:
            List of ResourceMetadata objects
        """
        resources = [
            ResourceMetadata(
                uri="diagnosis://{subject_id}/report",
                name="Diagnostic Report",
                description="Complete diagnostic data including prediction, SHAP values, UQ score, and anomaly status",
                mime_type="application/json"
            ),
            ResourceMetadata(
                uri="diagnosis://{subject_id}/features",
                name="Raw Features",
                description="Raw feature values for the subject",
                mime_type="application/json"
            ),
            ResourceMetadata(
                uri="knowledge://{region_name}/context",
                name="Clinical Knowledge Context",
                description="Clinical information about a brain region from knowledge graph",
                mime_type="application/json"
            )
        ]
        
        return resources
    
    def read_resource(self, uri: str) -> Dict:
        """
        Read a resource by URI
        
        Supported URI patterns:
        - diagnosis://{subject_id}/report
        - diagnosis://{subject_id}/features
        - knowledge://{region_name}/context
        
        Args:
            uri: Resource URI
        
        Returns:
            Resource data as dictionary or error dict
        """
        if self.verbose:
            print(f"[MCP] read_resource: {uri}")
        
        try:
            # Parse URI and route to appropriate handler
            if uri.startswith("diagnosis://"):
                return self._read_diagnostic_resource(uri)
            elif uri.startswith("knowledge://"):
                return self._read_knowledge_resource(uri)
            else:
                return {
                    "error": f"Invalid resource URI: {uri}. Supported schemes: diagnosis://, knowledge://",
                    "uri": uri,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "error": str(e),
                "uri": uri,
                "timestamp": datetime.now().isoformat()
            }
    
    def list_tools(self) -> List[ToolMetadata]:
        """
        List all available MCP tools
        
        Returns:
            List of ToolMetadata objects
        """
        tools = [
            ToolMetadata(
                name="simulate_counterfactual",
                description="Execute what-if analysis by masking specific features to assess their diagnostic impact",
                input_schema={
                    "type": "object",
                    "properties": {
                        "subject_id": {
                            "type": "string",
                            "description": "Patient identifier (e.g., 'sub-0005')"
                        },
                        "features_to_mask": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of ROI names or feature names to neutralize"
                        }
                    },
                    "required": ["subject_id", "features_to_mask"]
                }
            )
        ]
        
        return tools
    
    def call_tool(self, name: str, arguments: Dict) -> Dict:
        """
        Execute a tool by name
        
        Supported tools:
        - simulate_counterfactual
        
        Args:
            name: Tool name
            arguments: Tool arguments as dictionary
        
        Returns:
            Tool execution results or error dict
        """
        if self.verbose:
            print(f"[MCP] call_tool: {name}")
            print(f"[MCP] arguments: {arguments}")
        
        try:
            if name == "simulate_counterfactual":
                return self._execute_counterfactual(arguments)
            else:
                return {
                    "error": f"Unknown tool: {name}. Available tools: simulate_counterfactual",
                    "tool": name,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "error": str(e),
                "tool": name,
                "timestamp": datetime.now().isoformat()
            }
    
    # ========================================================================
    # Resource Handlers (Task 1.2: URI Routing)
    # ========================================================================
    
    def _read_diagnostic_resource(self, uri: str) -> Dict:
        """
        Handle diagnosis:// URIs
        
        Patterns:
        - diagnosis://{subject_id}/report
        - diagnosis://{subject_id}/features
        
        Args:
            uri: Diagnostic resource URI
        
        Returns:
            Diagnostic data
        
        Raises:
            ValueError: If URI pattern is invalid
        """
        # Parse URI: diagnosis://{subject_id}/{resource_type}
        pattern = r"^diagnosis://([^/]+)/(.+)$"
        match = re.match(pattern, uri)
        
        if not match:
            raise ValueError(
                f"Invalid diagnosis URI: {uri}. "
                f"Expected format: diagnosis://{{subject_id}}/{{resource_type}}"
            )
        
        subject_id = match.group(1)
        resource_type = match.group(2)
        
        if resource_type == "report":
            # Get full diagnostic report from toolkit
            try:
                report = self.toolkit.get_diagnostic_report(
                    subject_id,
                    verbose=self.verbose
                )
                # Flatten structure for MCP compliance - include subject_id at top level
                return {
                    "uri": uri,
                    "subject_id": subject_id,
                    "prediction_result": report.get("prediction_result"),
                    "prediction": report.get("prediction_result"),  # Alias for compatibility
                    "confidence": report.get("confidence"),
                    "uq_score": report.get("uq_score"),
                    "top_features": report.get("top_features"),
                    "anomaly_status": report.get("anomaly_status"),
                    "metadata": report.get("metadata"),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise ValueError(
                    f"Failed to retrieve diagnostic report for {subject_id}: {e}"
                )
        
        elif resource_type == "features":
            # Get raw features only
            try:
                # Run prediction to get features
                prediction_results = self.toolkit.predictor.predict_subject(
                    subject_id,
                    verbose=False
                )
                return {
                    "uri": uri,
                    "data": {
                        "subject_id": subject_id,
                        "features": prediction_results['features']
                    },
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise ValueError(
                    f"Failed to retrieve features for {subject_id}: {e}"
                )
        
        else:
            raise ValueError(
                f"Unknown diagnostic resource type: {resource_type}. "
                f"Supported types: report, features"
            )
    
    def _read_knowledge_resource(self, uri: str) -> Dict:
        """
        Handle knowledge:// URIs
        
        Pattern:
        - knowledge://{region_name}/context
        
        Args:
            uri: Knowledge resource URI
        
        Returns:
            Clinical knowledge data
        
        Raises:
            ValueError: If URI pattern is invalid
        """
        # Parse URI: knowledge://{region_name}/context
        pattern = r"^knowledge://([^/]+)/context$"
        match = re.match(pattern, uri)
        
        if not match:
            raise ValueError(
                f"Invalid knowledge URI: {uri}. "
                f"Expected format: knowledge://{{region_name}}/context"
            )
        
        region_name = match.group(1)
        
        try:
            # Query GraphRAG for region context
            region_info = self.graph_rag.query_region(region_name)
            
            if region_info is None:
                raise ValueError(f"Region not found: {region_name}")
            
            return {
                "uri": uri,
                "data": {
                    "region_name": region_name,
                    "context": region_info
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            # Use fallback if GraphRAG fails (Requirement 10.4)
            if self.verbose:
                print(f"[WARN] GraphRAG query failed, using fallback: {e}")
            
            # Log the error
            from app.services.llm_providers.error_handling import log_llm_error
            log_llm_error(
                e,
                {
                    'component': 'MCP Server',
                    'resource': 'knowledge',
                    'region': region_name,
                    'fallback': 'local knowledge base'
                }
            )
            
            fallback_info = self.graph_rag._query_region_fallback(region_name)
            return {
                "uri": uri,
                "data": {
                    "region_name": region_name,
                    "context": fallback_info,
                    "fallback": True,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e)
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
    
    # ========================================================================
    # Tool Handlers (Task 1.4: Tool Invocation)
    # ========================================================================
    
    def _execute_counterfactual(self, arguments: Dict) -> Dict:
        """
        Execute counterfactual simulation tool
        
        Args:
            arguments: Tool arguments with keys:
                - subject_id: str
                - features_to_mask: List[str]
        
        Returns:
            Counterfactual simulation results
        
        Raises:
            KeyError: If required arguments are missing
            ValueError: If arguments are invalid
        """
        # Validate required arguments
        if "subject_id" not in arguments:
            raise KeyError("Missing required argument: subject_id")
        if "features_to_mask" not in arguments:
            raise KeyError("Missing required argument: features_to_mask")
        
        subject_id = arguments["subject_id"]
        features_to_mask = arguments["features_to_mask"]
        
        # Validate argument types
        if not isinstance(subject_id, str):
            raise ValueError(f"subject_id must be string, got {type(subject_id)}")
        if not isinstance(features_to_mask, list):
            raise ValueError(f"features_to_mask must be list, got {type(features_to_mask)}")
        if not all(isinstance(f, str) for f in features_to_mask):
            raise ValueError("All features_to_mask items must be strings")
        
        try:
            # Execute counterfactual simulation via toolkit
            results = self.toolkit.simulate_counterfactual(
                subject_id=subject_id,
                features_to_mask=features_to_mask,
                verbose=self.verbose
            )
            
            # Flatten structure for MCP compliance
            return {
                "tool": "simulate_counterfactual",
                "status": "success",
                "subject_id": results.get("subject_id"),
                "original_prediction": results.get("original_prediction"),
                "original_confidence": results.get("original_confidence"),
                "new_prediction": results.get("new_prediction"),
                "new_confidence": results.get("new_confidence"),
                "confidence_delta": results.get("confidence_delta"),
                "masked_features": results.get("masked_features"),
                "interpretation": results.get("interpretation"),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            # Handle tool execution errors gracefully
            error_msg = f"Counterfactual simulation failed: {str(e)}"
            if self.verbose:
                print(f"[ERROR] {error_msg}")
            
            return {
                "tool": "simulate_counterfactual",
                "status": "error",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e)
                },
                "timestamp": datetime.now().isoformat()
            }
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_resource_metadata_by_uri(self, uri: str) -> Optional[ResourceMetadata]:
        """
        Get metadata for a specific resource URI
        
        Args:
            uri: Resource URI (may contain placeholders)
        
        Returns:
            ResourceMetadata or None if not found
        """
        resources = self.list_resources()
        for resource in resources:
            if resource.uri == uri:
                return resource
        return None
    
    def get_tool_metadata_by_name(self, name: str) -> Optional[ToolMetadata]:
        """
        Get metadata for a specific tool
        
        Args:
            name: Tool name
        
        Returns:
            ToolMetadata or None if not found
        """
        tools = self.list_tools()
        for tool in tools:
            if tool.name == name:
                return tool
        return None
    
    def validate_tool_arguments(self, tool_name: str, arguments: Dict) -> bool:
        """
        Validate tool arguments against schema
        
        Args:
            tool_name: Tool name
            arguments: Arguments to validate
        
        Returns:
            True if valid, False otherwise
        """
        tool_metadata = self.get_tool_metadata_by_name(tool_name)
        if not tool_metadata:
            return False
        
        schema = tool_metadata.input_schema
        required_fields = schema.get("required", [])
        
        # Check required fields
        for field in required_fields:
            if field not in arguments:
                return False
        
        return True


# ============================================================================
# Demo Functions
# ============================================================================

def demo_mcp_resources():
    """Demo: MCP resource access"""
    print("\n" + "="*80)
    print("DEMO: MCP Server - Resource Access")
    print("="*80)
    
    # Initialize server
    server = DiagnosticMCPServer(verbose=True)
    
    # List available resources
    print("\n[1] List Resources")
    resources = server.list_resources()
    for resource in resources:
        print(f"  - {resource.uri}")
        print(f"    {resource.description}")
    
    # Read diagnostic report
    print("\n[2] Read Resource: diagnosis://sub-0005/report")
    report = server.read_resource("diagnosis://sub-0005/report")
    print(f"  Subject: {report['data']['subject_id']}")
    print(f"  Prediction: {report['data']['prediction_result']}")
    print(f"  Confidence: {report['data']['confidence']:.1%}")
    print(f"  UQ Score: {report['data']['uq_score']:.3f}")
    
    # Read knowledge context
    print("\n[3] Read Resource: knowledge://Hippocampus_L/context")
    knowledge = server.read_resource("knowledge://Hippocampus_L/context")
    context = knowledge['data']['context']
    print(f"  Region: {context['full_name']}")
    print(f"  Function: {context.get('function', context.get('summary', 'N/A'))}")
    print(f"  AD Hotspot: {context.get('is_ad_hotspot', False)}")
    
    print("\n" + "="*80)


def demo_mcp_tools():
    """Demo: MCP tool execution"""
    print("\n" + "="*80)
    print("DEMO: MCP Server - Tool Execution")
    print("="*80)
    
    # Initialize server
    server = DiagnosticMCPServer(verbose=True)
    
    # List available tools
    print("\n[1] List Tools")
    tools = server.list_tools()
    for tool in tools:
        print(f"  - {tool.name}")
        print(f"    {tool.description}")
    
    # Execute counterfactual simulation
    print("\n[2] Call Tool: simulate_counterfactual")
    result = server.call_tool(
        "simulate_counterfactual",
        {
            "subject_id": "sub-0005",
            "features_to_mask": ["Hippocampus_L", "Hippocampus_R"]
        }
    )
    
    if result['status'] == 'success':
        data = result['data']
        print(f"  Original: {data['original_prediction']} ({data['original_confidence']:.1%})")
        print(f"  Counterfactual: {data['new_prediction']} ({data['new_confidence']:.1%})")
        print(f"  Delta: {data['confidence_delta']:+.1%}")
        print(f"  Interpretation: {data['interpretation']}")
    else:
        print(f"  Error: {result['error']['message']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run demos
    demo_mcp_resources()
    print("\n\n")
    demo_mcp_tools()
