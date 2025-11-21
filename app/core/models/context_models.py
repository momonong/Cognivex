"""
Context and Agent Data Models

This module defines data models for agent context and results:
- DiagnosticReport: ML model predictions and analysis
- Feature: Individual brain region feature with SHAP and Z-score
- AnomalyStatus: Statistical anomaly detection results
- CounterfactualResult: What-if simulation results
- KnowledgeContext: Clinical knowledge from knowledge graph
- ContextObject: Complete context for Agent A → Agent B handoff
- AgentResult: Final output from the A2A system

These models enable structured data flow through the CDDA pipeline.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
from datetime import datetime


# ============================================================================
# Diagnostic Data Models
# ============================================================================

@dataclass
class Feature:
    """
    Individual brain region feature with analysis metrics
    
    Represents a single ROI measurement with its statistical
    and explainability metrics.
    """
    roi_name: str  # Brain region name (e.g., "Hippocampus_L")
    feature_name: str  # Full feature name (e.g., "Hippocampus_L_GM_Vol")
    feature_value: float  # Raw measurement value
    z_score: float  # Standardized score vs. population
    shap_value: float  # SHAP importance score
    rank: int  # Importance ranking (1 = most important)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()
    
    def is_anomalous(self, threshold: float = 2.5) -> bool:
        """Check if feature is statistically anomalous"""
        return abs(self.z_score) > threshold


@dataclass
class AnomalyStatus:
    """
    Statistical anomaly detection results
    
    Tracks which brain regions exhibit unusual patterns
    that may indicate mixed pathology or data quality issues.
    """
    has_anomaly: bool
    anomalous_regions: List[str]  # List of ROI names
    anomaly_type: Optional[str] = None  # e.g., "statistical_outlier"
    threshold_used: float = 2.5  # Z-score threshold
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()


@dataclass
class DiagnosticReport:
    """
    Complete diagnostic report from ML model
    
    Contains prediction, confidence, uncertainty, feature importance,
    and anomaly detection results.
    """
    subject_id: str
    prediction_result: str  # "AD", "NC", or "MCI"
    confidence: float  # 0.0 to 1.0
    uq_score: float  # Uncertainty quantification (0.0 to 1.0)
    top_features: List[Feature]  # Ranked by SHAP importance
    anomaly_status: AnomalyStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert nested Feature objects
        data['top_features'] = [f.to_dict() if hasattr(f, 'to_dict') else f 
                                for f in self.top_features]
        # Convert AnomalyStatus
        if hasattr(self.anomaly_status, 'to_dict'):
            data['anomaly_status'] = self.anomaly_status.to_dict()
        return data
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()
    
    @classmethod
    def from_toolkit_report(cls, report: Dict) -> 'DiagnosticReport':
        """
        Create DiagnosticReport from CDDAToolKit output
        
        Args:
            report: Dictionary from toolkit.get_diagnostic_report()
        
        Returns:
            DiagnosticReport instance
        """
        # Convert top_features to Feature objects
        features = []
        for i, feat in enumerate(report.get('top_features', []), 1):
            if isinstance(feat, dict):
                features.append(Feature(
                    roi_name=feat['roi_name'],
                    feature_name=feat['feature_name'],
                    feature_value=feat['feature_value'],
                    z_score=feat['z_score'],
                    shap_value=feat['shap_value'],
                    rank=i
                ))
            else:
                features.append(feat)
        
        # Convert anomaly_status to AnomalyStatus object
        anomaly_data = report.get('anomaly_status', {})
        if isinstance(anomaly_data, dict):
            anomaly_status = AnomalyStatus(
                has_anomaly=anomaly_data.get('has_anomaly', False),
                anomalous_regions=anomaly_data.get('anomalous_regions', []),
                anomaly_type=anomaly_data.get('anomaly_type')
            )
        else:
            anomaly_status = anomaly_data
        
        return cls(
            subject_id=report['subject_id'],
            prediction_result=report['prediction_result'],
            confidence=report['confidence'],
            uq_score=report['uq_score'],
            top_features=features,
            anomaly_status=anomaly_status,
            metadata=report.get('metadata', {})
        )


# ============================================================================
# Tool Result Models
# ============================================================================

@dataclass
class MaskedFeature:
    """
    Feature that was masked in counterfactual simulation
    
    Tracks which features were neutralized and their original values.
    """
    roi_name: str
    feature_name: str
    original_value: float
    masked_value: float  # Typically population mean
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()


@dataclass
class CounterfactualResult:
    """
    Results from counterfactual simulation (Tool 2)
    
    Shows how prediction changes when specific features are masked,
    helping identify key diagnostic drivers.
    """
    subject_id: str
    original_prediction: str
    original_confidence: float
    new_prediction: str
    new_confidence: float
    confidence_delta: float  # new - original
    masked_features: List[MaskedFeature]
    interpretation: str  # Natural language explanation
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert nested MaskedFeature objects
        data['masked_features'] = [f.to_dict() if hasattr(f, 'to_dict') else f 
                                   for f in self.masked_features]
        return data
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()
    
    @classmethod
    def from_toolkit_result(cls, result: Dict) -> 'CounterfactualResult':
        """
        Create CounterfactualResult from CDDAToolKit output
        
        Args:
            result: Dictionary from toolkit.simulate_counterfactual()
        
        Returns:
            CounterfactualResult instance
        """
        # Convert masked_features to MaskedFeature objects
        masked_features = []
        for feat in result.get('masked_features', []):
            if isinstance(feat, dict):
                masked_features.append(MaskedFeature(
                    roi_name=feat['roi_name'],
                    feature_name=feat['feature_name'],
                    original_value=feat['original_value'],
                    masked_value=feat['masked_value']
                ))
            else:
                masked_features.append(feat)
        
        return cls(
            subject_id=result['subject_id'],
            original_prediction=result['original_prediction'],
            original_confidence=result['original_confidence'],
            new_prediction=result['new_prediction'],
            new_confidence=result['new_confidence'],
            confidence_delta=result['confidence_delta'],
            masked_features=masked_features,
            interpretation=result['interpretation']
        )


@dataclass
class RegionContext:
    """
    Clinical knowledge about a single brain region
    
    Retrieved from Neo4j knowledge graph via GraphRAG.
    """
    region_name: str
    full_name: str
    function: str
    clinical_significance: str
    related_conditions: List[str] = field(default_factory=list)
    is_ad_hotspot: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()


@dataclass
class KnowledgeContext:
    """
    Clinical knowledge from knowledge graph (Tool 4)
    
    Contains context about anomalous brain regions to help
    interpret unusual patterns.
    """
    query_regions: List[str]  # Regions that were queried
    contexts: List[RegionContext]  # Context for each region
    summary: str  # Natural language summary
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert nested RegionContext objects
        data['contexts'] = [c.to_dict() if hasattr(c, 'to_dict') else c 
                           for c in self.contexts]
        return data
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()


# ============================================================================
# A2A Handoff Models
# ============================================================================

@dataclass
class ContextObject:
    """
    Complete context for Agent A → Agent B handoff
    
    This is the structured data package that Agent A compiles and
    hands off to Agent B. It contains all information needed for
    clinical synthesis, ensuring Agent B has no direct tool access.
    
    Requirements: 5.1, 8.3, 10.4, 10.5
    """
    subject_id: str
    diagnostic_report: DiagnosticReport
    tool_results: Optional[Dict[str, Any]] = None  # counterfactual or knowledge_context
    decision_rationale: str = ""  # Why Agent A took certain actions
    signals: Dict[str, Any] = field(default_factory=dict)  # uq_score, has_anomaly, etc.
    agent_a_reasoning: List[str] = field(default_factory=list)  # Step-by-step reasoning
    mcp_actions: List[Any] = field(default_factory=list)  # List of MCPAction objects
    errors: List[Dict[str, Any]] = field(default_factory=list)  # Error annotations (Requirement 10.5)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = {
            'subject_id': self.subject_id,
            'diagnostic_report': (self.diagnostic_report.to_dict() 
                                 if hasattr(self.diagnostic_report, 'to_dict') 
                                 else self.diagnostic_report),
            'tool_results': self.tool_results,
            'decision_rationale': self.decision_rationale,
            'signals': self.signals,
            'agent_a_reasoning': self.agent_a_reasoning,
            'mcp_actions': [a.to_dict() if hasattr(a, 'to_dict') else a 
                           for a in self.mcp_actions],
            'errors': self.errors,
            'timestamp': self.timestamp
        }
        return data
    
    def add_error(self, error_type: str, error_message: str, component: str):
        """
        Add error annotation to ContextObject
        
        Requirements: 10.5
        
        Args:
            error_type: Type of error (e.g., "GraphRAGError", "LLMError")
            error_message: Error message
            component: Component where error occurred (e.g., "Agent A", "MCP Server")
        """
        self.errors.append({
            'type': error_type,
            'message': error_message,
            'component': component,
            'timestamp': datetime.now().isoformat()
        })
    
    def has_errors(self) -> bool:
        """Check if any errors were recorded"""
        return len(self.errors) > 0
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()
    
    def validate(self) -> bool:
        """
        Validate that all required fields are present
        
        Requirements: 5.1 - Ensure all required fields are present
        
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not self.subject_id:
            return False
        if not self.diagnostic_report:
            return False
        if not self.signals:
            return False
        
        # Check diagnostic report has required fields
        if hasattr(self.diagnostic_report, 'prediction_result'):
            if not self.diagnostic_report.prediction_result:
                return False
        
        return True
    
    def serialize_for_agent_b(self) -> str:
        """
        Serialize ContextObject for Agent B consumption
        
        Requirements: 5.1, 8.3
        
        Returns:
            JSON string representation suitable for LLM prompt
        """
        import json
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# Agent Result Models
# ============================================================================

@dataclass
class AgentResult:
    """
    Final output from the A2A system
    
    Contains the complete diagnostic analysis including:
    - Agent A's orchestration decisions
    - Agent B's clinical synthesis
    - Complete reasoning chain from both agents
    - ContextObject for transparency
    
    Requirements: 3.5, 8.1, 8.2, 8.3, 8.4
    """
    subject_id: str
    agent_decision: str  # SIMULATION_TRIGGERED, ANOMALY_INVESTIGATION, STANDARD_REPORT
    prediction: str  # AD, NC, or MCI
    confidence: float  # 0.0 to 1.0
    uq_score: float  # Uncertainty quantification
    context_object: ContextObject  # Full context from Agent A
    clinical_report: str  # Natural language report from Agent B
    reasoning_chain: List[str]  # Combined reasoning from both agents
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Legacy fields for backward compatibility
    report: Optional[Dict] = None
    counterfactual: Optional[Dict] = None
    knowledge_context: Optional[Dict] = None
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = {
            'subject_id': self.subject_id,
            'agent_decision': self.agent_decision,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'uq_score': self.uq_score,
            'context_object': (self.context_object.to_dict() 
                              if hasattr(self.context_object, 'to_dict') 
                              else self.context_object),
            'clinical_report': self.clinical_report,
            'reasoning_chain': self.reasoning_chain,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
        
        # Include legacy fields if present
        if self.report is not None:
            data['report'] = self.report
        if self.counterfactual is not None:
            data['counterfactual'] = self.counterfactual
        if self.knowledge_context is not None:
            data['knowledge_context'] = self.knowledge_context
        if self.explanation is not None:
            data['explanation'] = self.explanation
        
        return data
    
    def to_json(self) -> Dict:
        """Alias for to_dict() for consistency"""
        return self.to_dict()
    
    @classmethod
    def from_legacy_result(cls, legacy_result: Dict) -> 'AgentResult':
        """
        Create AgentResult from legacy CDDA agent output
        
        This enables backward compatibility with existing code.
        
        Args:
            legacy_result: Dictionary from old CDDAAgent.run_analysis()
        
        Returns:
            AgentResult instance
        """
        # Extract diagnostic report
        report_data = legacy_result.get('report', {})
        diagnostic_report = DiagnosticReport.from_toolkit_report(report_data)
        
        # Create ContextObject
        context_object = ContextObject(
            subject_id=legacy_result['subject_id'],
            diagnostic_report=diagnostic_report,
            tool_results={
                'counterfactual': legacy_result.get('counterfactual'),
                'knowledge_context': legacy_result.get('knowledge_context')
            },
            decision_rationale=f"Agent decision: {legacy_result['agent_decision']}",
            signals={
                'uq_score': legacy_result['uq_score'],
                'has_anomaly': report_data.get('anomaly_status', {}).get('has_anomaly', False)
            },
            agent_a_reasoning=legacy_result.get('reasoning_chain', [])
        )
        
        return cls(
            subject_id=legacy_result['subject_id'],
            agent_decision=legacy_result['agent_decision'],
            prediction=legacy_result['prediction'],
            confidence=legacy_result['confidence'],
            uq_score=legacy_result['uq_score'],
            context_object=context_object,
            clinical_report=legacy_result.get('explanation', ''),
            reasoning_chain=legacy_result.get('reasoning_chain', []),
            timestamp=legacy_result.get('timestamp', datetime.now().isoformat()),
            # Legacy fields
            report=report_data,
            counterfactual=legacy_result.get('counterfactual'),
            knowledge_context=legacy_result.get('knowledge_context'),
            explanation=legacy_result.get('explanation')
        )
