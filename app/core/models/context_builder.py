"""
ContextObject Builder

This module provides a helper class to compile ContextObject from diagnostic data.
It ensures all required fields are present and provides validation before handoff
from Agent A to Agent B.

Requirements: 5.1, 8.3
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from .context_models import (
    ContextObject,
    DiagnosticReport,
    CounterfactualResult,
    KnowledgeContext,
    RegionContext
)
from .mcp_models import MCPAction


class ContextObjectBuilder:
    """
    Helper class to compile ContextObject from diagnostic data
    
    This builder ensures that all required fields are present and
    properly formatted before Agent A hands off to Agent B.
    
    Requirements: 5.1, 8.3
    """
    
    def __init__(self):
        """Initialize builder with empty state"""
        self._subject_id: Optional[str] = None
        self._diagnostic_report: Optional[DiagnosticReport] = None
        self._tool_results: Dict[str, Any] = {}
        self._decision_rationale: str = ""
        self._signals: Dict[str, Any] = {}
        self._agent_a_reasoning: List[str] = []
        self._mcp_actions: List[MCPAction] = []
    
    def set_subject_id(self, subject_id: str) -> 'ContextObjectBuilder':
        """
        Set subject identifier
        
        Args:
            subject_id: Patient identifier
        
        Returns:
            Self for method chaining
        """
        self._subject_id = subject_id
        return self
    
    def set_diagnostic_report(
        self, 
        report: DiagnosticReport
    ) -> 'ContextObjectBuilder':
        """
        Set diagnostic report
        
        Args:
            report: DiagnosticReport instance
        
        Returns:
            Self for method chaining
        """
        self._diagnostic_report = report
        return self
    
    def set_diagnostic_report_from_dict(
        self, 
        report_dict: Dict
    ) -> 'ContextObjectBuilder':
        """
        Set diagnostic report from dictionary
        
        Args:
            report_dict: Dictionary from toolkit.get_diagnostic_report()
        
        Returns:
            Self for method chaining
        """
        self._diagnostic_report = DiagnosticReport.from_toolkit_report(report_dict)
        return self
    
    def add_counterfactual_result(
        self, 
        result: CounterfactualResult
    ) -> 'ContextObjectBuilder':
        """
        Add counterfactual simulation result
        
        Args:
            result: CounterfactualResult instance
        
        Returns:
            Self for method chaining
        """
        self._tool_results['counterfactual'] = result
        return self
    
    def add_counterfactual_result_from_dict(
        self, 
        result_dict: Dict
    ) -> 'ContextObjectBuilder':
        """
        Add counterfactual result from dictionary
        
        Args:
            result_dict: Dictionary from toolkit.simulate_counterfactual()
        
        Returns:
            Self for method chaining
        """
        result = CounterfactualResult.from_toolkit_result(result_dict)
        self._tool_results['counterfactual'] = result
        return self
    
    def add_knowledge_context(
        self, 
        context: KnowledgeContext
    ) -> 'ContextObjectBuilder':
        """
        Add knowledge graph context
        
        Args:
            context: KnowledgeContext instance
        
        Returns:
            Self for method chaining
        """
        self._tool_results['knowledge_context'] = context
        return self
    
    def add_knowledge_context_from_dict(
        self, 
        context_dict: Dict
    ) -> 'ContextObjectBuilder':
        """
        Add knowledge context from dictionary
        
        Args:
            context_dict: Dictionary from agent.knowledge_graph_lookup()
        
        Returns:
            Self for method chaining
        """
        # Convert contexts to RegionContext objects
        region_contexts = []
        for ctx in context_dict.get('contexts', []):
            if isinstance(ctx, dict):
                context_data = ctx.get('context', {})
                region_contexts.append(RegionContext(
                    region_name=ctx['region'],
                    full_name=context_data.get('full_name', ctx['region']),
                    function=context_data.get('function', 'Unknown'),
                    clinical_significance=context_data.get('clinical_significance', 'Unknown'),
                    related_conditions=context_data.get('related_conditions', []),
                    is_ad_hotspot=context_data.get('is_ad_hotspot', False)
                ))
        
        knowledge_context = KnowledgeContext(
            query_regions=context_dict.get('query_regions', []),
            contexts=region_contexts,
            summary=context_dict.get('summary', '')
        )
        
        self._tool_results['knowledge_context'] = knowledge_context
        return self
    
    def set_decision_rationale(self, rationale: str) -> 'ContextObjectBuilder':
        """
        Set decision rationale explaining Agent A's actions
        
        Args:
            rationale: Natural language explanation
        
        Returns:
            Self for method chaining
        """
        self._decision_rationale = rationale
        return self
    
    def add_signal(self, key: str, value: Any) -> 'ContextObjectBuilder':
        """
        Add a signal (e.g., uq_score, has_anomaly)
        
        Args:
            key: Signal name
            value: Signal value
        
        Returns:
            Self for method chaining
        """
        self._signals[key] = value
        return self
    
    def set_signals(self, signals: Dict[str, Any]) -> 'ContextObjectBuilder':
        """
        Set all signals at once
        
        Args:
            signals: Dictionary of signals
        
        Returns:
            Self for method chaining
        """
        self._signals = signals
        return self
    
    def add_reasoning_step(self, step: str) -> 'ContextObjectBuilder':
        """
        Add a reasoning step to Agent A's reasoning chain
        
        Args:
            step: Description of reasoning step
        
        Returns:
            Self for method chaining
        """
        self._agent_a_reasoning.append(step)
        return self
    
    def set_reasoning_chain(self, chain: List[str]) -> 'ContextObjectBuilder':
        """
        Set complete reasoning chain
        
        Args:
            chain: List of reasoning steps
        
        Returns:
            Self for method chaining
        """
        self._agent_a_reasoning = chain
        return self
    
    def add_mcp_action(self, action: MCPAction) -> 'ContextObjectBuilder':
        """
        Add an MCP action to the action log
        
        Args:
            action: MCPAction instance
        
        Returns:
            Self for method chaining
        """
        self._mcp_actions.append(action)
        return self
    
    def set_mcp_actions(self, actions: List[MCPAction]) -> 'ContextObjectBuilder':
        """
        Set complete MCP action list
        
        Args:
            actions: List of MCPAction instances
        
        Returns:
            Self for method chaining
        """
        self._mcp_actions = actions
        return self
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate that all required fields are present
        
        Requirements: 5.1 - Ensure all required fields are present
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._subject_id:
            return False, "Missing required field: subject_id"
        
        if not self._diagnostic_report:
            return False, "Missing required field: diagnostic_report"
        
        if not self._signals:
            return False, "Missing required field: signals"
        
        # Validate diagnostic report has required fields
        if not self._diagnostic_report.prediction_result:
            return False, "Diagnostic report missing prediction_result"
        
        if self._diagnostic_report.confidence is None:
            return False, "Diagnostic report missing confidence"
        
        if self._diagnostic_report.uq_score is None:
            return False, "Diagnostic report missing uq_score"
        
        return True, None
    
    def build(self) -> ContextObject:
        """
        Build and return ContextObject
        
        Requirements: 5.1, 8.3
        
        Returns:
            ContextObject instance
        
        Raises:
            ValueError: If validation fails
        """
        # Validate before building
        is_valid, error_msg = self.validate()
        if not is_valid:
            raise ValueError(f"Cannot build ContextObject: {error_msg}")
        
        # Auto-populate signals from diagnostic report if not set
        if 'uq_score' not in self._signals and self._diagnostic_report:
            self._signals['uq_score'] = self._diagnostic_report.uq_score
        
        if 'has_anomaly' not in self._signals and self._diagnostic_report:
            self._signals['has_anomaly'] = self._diagnostic_report.anomaly_status.has_anomaly
        
        if 'prediction' not in self._signals and self._diagnostic_report:
            self._signals['prediction'] = self._diagnostic_report.prediction_result
        
        if 'confidence' not in self._signals and self._diagnostic_report:
            self._signals['confidence'] = self._diagnostic_report.confidence
        
        # Build ContextObject
        context_object = ContextObject(
            subject_id=self._subject_id,
            diagnostic_report=self._diagnostic_report,
            tool_results=self._tool_results if self._tool_results else None,
            decision_rationale=self._decision_rationale,
            signals=self._signals,
            agent_a_reasoning=self._agent_a_reasoning,
            mcp_actions=self._mcp_actions,
            timestamp=datetime.now().isoformat()
        )
        
        return context_object
    
    def reset(self) -> 'ContextObjectBuilder':
        """
        Reset builder to empty state
        
        Returns:
            Self for method chaining
        """
        self._subject_id = None
        self._diagnostic_report = None
        self._tool_results = {}
        self._decision_rationale = ""
        self._signals = {}
        self._agent_a_reasoning = []
        self._mcp_actions = []
        return self


# ============================================================================
# Convenience Functions
# ============================================================================

def build_context_from_diagnostic_report(
    subject_id: str,
    report_dict: Dict,
    decision_rationale: str = "",
    reasoning_chain: Optional[List[str]] = None
) -> ContextObject:
    """
    Quick builder for standard case (no tool results)
    
    Args:
        subject_id: Patient identifier
        report_dict: Dictionary from toolkit.get_diagnostic_report()
        decision_rationale: Why Agent A made this decision
        reasoning_chain: Agent A's reasoning steps
    
    Returns:
        ContextObject instance
    """
    builder = ContextObjectBuilder()
    builder.set_subject_id(subject_id)
    builder.set_diagnostic_report_from_dict(report_dict)
    builder.set_decision_rationale(decision_rationale)
    
    # Auto-populate signals from report
    builder.add_signal("uq_score", report_dict.get('uq_score', 0.0))
    builder.add_signal("has_anomaly", report_dict.get('anomaly_status', {}).get('has_anomaly', False))
    builder.add_signal("prediction", report_dict.get('prediction_result', ''))
    builder.add_signal("confidence", report_dict.get('confidence', 0.0))
    
    if reasoning_chain:
        builder.set_reasoning_chain(reasoning_chain)
    
    return builder.build()


def build_context_with_counterfactual(
    subject_id: str,
    report_dict: Dict,
    counterfactual_dict: Dict,
    decision_rationale: str = "",
    reasoning_chain: Optional[List[str]] = None
) -> ContextObject:
    """
    Quick builder for high uncertainty case (with counterfactual)
    
    Args:
        subject_id: Patient identifier
        report_dict: Dictionary from toolkit.get_diagnostic_report()
        counterfactual_dict: Dictionary from toolkit.simulate_counterfactual()
        decision_rationale: Why Agent A triggered counterfactual
        reasoning_chain: Agent A's reasoning steps
    
    Returns:
        ContextObject instance
    """
    builder = ContextObjectBuilder()
    builder.set_subject_id(subject_id)
    builder.set_diagnostic_report_from_dict(report_dict)
    builder.add_counterfactual_result_from_dict(counterfactual_dict)
    builder.set_decision_rationale(decision_rationale)
    
    # Auto-populate signals from report
    builder.add_signal("uq_score", report_dict.get('uq_score', 0.0))
    builder.add_signal("has_anomaly", report_dict.get('anomaly_status', {}).get('has_anomaly', False))
    builder.add_signal("prediction", report_dict.get('prediction_result', ''))
    builder.add_signal("confidence", report_dict.get('confidence', 0.0))
    
    if reasoning_chain:
        builder.set_reasoning_chain(reasoning_chain)
    
    return builder.build()


def build_context_with_knowledge(
    subject_id: str,
    report_dict: Dict,
    knowledge_dict: Dict,
    decision_rationale: str = "",
    reasoning_chain: Optional[List[str]] = None
) -> ContextObject:
    """
    Quick builder for anomaly case (with knowledge context)
    
    Args:
        subject_id: Patient identifier
        report_dict: Dictionary from toolkit.get_diagnostic_report()
        knowledge_dict: Dictionary from agent.knowledge_graph_lookup()
        decision_rationale: Why Agent A queried knowledge graph
        reasoning_chain: Agent A's reasoning steps
    
    Returns:
        ContextObject instance
    """
    builder = ContextObjectBuilder()
    builder.set_subject_id(subject_id)
    builder.set_diagnostic_report_from_dict(report_dict)
    builder.add_knowledge_context_from_dict(knowledge_dict)
    builder.set_decision_rationale(decision_rationale)
    
    # Auto-populate signals from report
    builder.add_signal("uq_score", report_dict.get('uq_score', 0.0))
    builder.add_signal("has_anomaly", report_dict.get('anomaly_status', {}).get('has_anomaly', False))
    builder.add_signal("prediction", report_dict.get('prediction_result', ''))
    builder.add_signal("confidence", report_dict.get('confidence', 0.0))
    
    if reasoning_chain:
        builder.set_reasoning_chain(reasoning_chain)
    
    return builder.build()
