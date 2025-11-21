"""
CDDA Framework - Data Models

This module contains all data models for the CDDA system:
- MCP Protocol models (ResourceMetadata, ToolMetadata, MCPAction)
- Context models (ContextObject, DiagnosticReport, etc.)
- Agent result models
- Context builder utilities
"""

from .mcp_models import ResourceMetadata, ToolMetadata, MCPAction, MCPActionList
from .context_models import (
    ContextObject,
    DiagnosticReport,
    Feature,
    AnomalyStatus,
    CounterfactualResult,
    MaskedFeature,
    KnowledgeContext,
    RegionContext,
    AgentResult
)
from .context_builder import (
    ContextObjectBuilder,
    build_context_from_diagnostic_report,
    build_context_with_counterfactual,
    build_context_with_knowledge
)

__all__ = [
    # MCP Models
    'ResourceMetadata',
    'ToolMetadata',
    'MCPAction',
    'MCPActionList',
    
    # Context Models
    'ContextObject',
    'DiagnosticReport',
    'Feature',
    'AnomalyStatus',
    'CounterfactualResult',
    'MaskedFeature',
    'KnowledgeContext',
    'RegionContext',
    'AgentResult',
    
    # Context Builder
    'ContextObjectBuilder',
    'build_context_from_diagnostic_report',
    'build_context_with_counterfactual',
    'build_context_with_knowledge'
]
