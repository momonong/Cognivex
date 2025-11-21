"""
MCP Protocol Data Models

This module defines data models for the Model Context Protocol (MCP):
- ResourceMetadata: Metadata for read-only resources
- ToolMetadata: Metadata for executable tools
- MCPAction: Record of MCP operations (read_resource or call_tool)

These models enable structured communication between Agent A (Orchestrator)
and the DiagnosticMCPServer.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class ResourceMetadata:
    """
    Metadata for an MCP resource (read-only data)
    
    Resources represent data that can be read but not modified.
    Examples: diagnostic reports, knowledge graph context
    """
    uri: str  # e.g., "diagnosis://{subject_id}/report"
    name: str
    description: str
    mime_type: str = "application/json"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()


@dataclass
class ToolMetadata:
    """
    Metadata for an MCP tool (executable action)
    
    Tools represent actions that can be executed with side effects.
    Examples: counterfactual simulation, model retraining
    """
    name: str  # e.g., "simulate_counterfactual"
    description: str
    input_schema: Dict[str, Any]  # JSON schema for validation
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()


@dataclass
class MCPAction:
    """
    Record of an MCP operation performed by Agent A
    
    This tracks what resources were read and what tools were called
    during the orchestration phase, providing transparency and
    auditability for the agent's decision-making process.
    """
    type: str  # "read_resource" or "call_tool"
    target: str  # URI (for resources) or tool name (for tools)
    arguments: Optional[Dict[str, Any]] = None  # Tool arguments (if type="call_tool")
    result: Optional[Dict[str, Any]] = None  # Operation result
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"  # "pending", "success", "error"
    error: Optional[str] = None  # Error message if status="error"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()
    
    def mark_success(self, result: Dict[str, Any]):
        """Mark action as successful with result"""
        self.status = "success"
        self.result = result
    
    def mark_error(self, error_message: str):
        """Mark action as failed with error message"""
        self.status = "error"
        self.error = error_message


@dataclass
class MCPActionList:
    """
    Collection of MCP actions with utility methods
    
    This provides a convenient way to track and query all MCP operations
    performed during an agent's execution.
    """
    actions: List[MCPAction] = field(default_factory=list)
    
    def add_action(self, action: MCPAction):
        """Add an action to the list"""
        self.actions.append(action)
    
    def get_resource_reads(self) -> List[MCPAction]:
        """Get all resource read actions"""
        return [a for a in self.actions if a.type == "read_resource"]
    
    def get_tool_calls(self) -> List[MCPAction]:
        """Get all tool call actions"""
        return [a for a in self.actions if a.type == "call_tool"]
    
    def get_successful_actions(self) -> List[MCPAction]:
        """Get all successful actions"""
        return [a for a in self.actions if a.status == "success"]
    
    def get_failed_actions(self) -> List[MCPAction]:
        """Get all failed actions"""
        return [a for a in self.actions if a.status == "error"]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "actions": [action.to_dict() for action in self.actions],
            "total_count": len(self.actions),
            "success_count": len(self.get_successful_actions()),
            "error_count": len(self.get_failed_actions())
        }
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()
