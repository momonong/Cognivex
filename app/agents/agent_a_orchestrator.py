"""
Agent A - Orchestrator (MCP Client)

This module implements Agent A, the orchestrator in the dual-LLM A2A system.
Agent A is responsible for:
1. Reading diagnostic resources from MCP server
2. Evaluating signals (UQ score, anomaly status)
3. Deciding which tools to invoke
4. Compiling ContextObject for handoff to Agent B

Agent A uses Phi-4-mini for function calling and decision logic.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.1, 8.2
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.mcp_server import DiagnosticMCPServer
from app.core.prompt_loader import PromptLoader
from app.core.models import (
    MCPAction, 
    ContextObject, 
    DiagnosticReport,
    CounterfactualResult,
    KnowledgeContext,
    RegionContext
)
from app.services.llm_providers import ollama, huggingface
from app.services.llm_providers.error_handling import (
    parse_json_with_recovery,
    log_llm_error,
    LLMRetryExhausted,
    LLMParsingError
)


# ============================================================================
# Agent A Configuration
# ============================================================================

@dataclass
class AgentAConfig:
    """Configuration for Agent A"""
    model: str = "phi-4-mini"  # Phi-4-mini for function calling
    model_path: Optional[str] = "D:/hf_models/Phi-4-mini-instruct"  # Path for HuggingFace models
    provider: str = "huggingface"  # "ollama" or "huggingface"
    temperature: float = 0.1
    uq_threshold: float = 0.8
    z_score_threshold: float = 2.5
    use_llm: bool = True  # If False, use rule-based logic
    load_in_8bit: bool = True  # Use 8-bit quantization to save memory
    verbose: bool = True
    prompt_path: str = "config/prompts/agent_a_orchestrator.txt"


# ============================================================================
# Agent A - Orchestrator
# ============================================================================

class AgentA:
    """
    Agent A - Orchestrator (MCP Client)
    
    Responsibilities:
    - Read resources from MCP server
    - Evaluate diagnostic signals
    - Invoke tools when needed
    - Compile ContextObject for Agent B
    - Log all decisions with reasoning
    
    Requirements: 3.1, 3.2, 3.3, 3.4
    """
    
    def __init__(
        self,
        mcp_server: DiagnosticMCPServer,
        config: Optional[AgentAConfig] = None
    ):
        """
        Initialize Agent A
        
        Args:
            mcp_server: DiagnosticMCPServer instance
            config: Agent A configuration
        """
        self.mcp_server = mcp_server
        self.config = config or AgentAConfig()
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        
        # Initialize reasoning chain for logging
        self.reasoning_chain: List[str] = []
        self.mcp_actions: List[MCPAction] = []
        
        if self.config.verbose:
            print("\n" + "="*80)
            print("AGENT A - ORCHESTRATOR (Phi-4-mini)")
            print("="*80)
            print(f"Model: {self.config.model}")
            print(f"UQ Threshold: {self.config.uq_threshold}")
            print(f"Z-Score Threshold: {self.config.z_score_threshold}")
            print(f"LLM Mode: {'Enabled' if self.config.use_llm else 'Rule-based fallback'}")
            print("="*80)
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file using PromptLoader"""
        try:
            loader = PromptLoader()
            return loader.load_agent_a_prompt()
        except Exception as e:
            if self.config.verbose:
                print(f"[WARN] Failed to load prompt via PromptLoader: {e}")
                print("[WARN] Falling back to direct file read")
            
            # Fallback to direct file read
            prompt_path = Path(self.config.prompt_path)
            if prompt_path.exists():
                return prompt_path.read_text(encoding='utf-8')
            else:
                # Final fallback to embedded prompt
                return """You are Agent A, the Orchestrator in a diagnostic system.
Your role is to read diagnostic resources, evaluate signals, and decide which tools to invoke.
Respond with JSON containing 'actions' and 'decision_rationale'."""
    
    def orchestrate(self, subject_id: str) -> ContextObject:
        """
        Main orchestration method
        
        This is the entry point for Agent A. It:
        1. Reads diagnostic report from MCP server
        2. Evaluates signals (UQ, anomalies)
        3. Decides which tools to invoke
        4. Compiles ContextObject for Agent B
        
        Args:
            subject_id: Subject identifier
        
        Returns:
            ContextObject for handoff to Agent B
        
        Requirements: 3.1, 3.2, 3.3, 3.4
        """
        if self.config.verbose:
            print(f"\n[AGENT A] Orchestrating analysis for {subject_id}")
        
        # Reset reasoning chain and actions
        self.reasoning_chain = []
        self.mcp_actions = []
        
        # Log start
        self._log_reasoning(f"Starting orchestration for {subject_id}")
        
        # Choose orchestration strategy
        if self.config.use_llm:
            return self._orchestrate_with_llm(subject_id)
        else:
            return self._orchestrate_with_rules(subject_id)
    
    def _orchestrate_with_llm(self, subject_id: str) -> ContextObject:
        """
        Orchestrate using LLM for decision making
        
        Includes automatic fallback to rule-based orchestration if LLM fails.
        
        Requirements: 1.1, 1.2, 3.2, 10.2
        """
        if self.config.verbose:
            print("[AGENT A] Using LLM-based orchestration")
        
        try:
            # Step 1: Read diagnostic report (always first)
            diagnostic_report = self._read_diagnostic_report(subject_id)
            
            # Step 2: Get LLM decision (with retry and error handling)
            llm_decision = self._get_llm_decision(subject_id, diagnostic_report)
            
            # Step 3: Execute LLM-decided actions
            tool_results = self._execute_llm_actions(llm_decision['actions'])
            
            # Step 4: Compile ContextObject
            context_object = self._compile_context_object(
                subject_id=subject_id,
                diagnostic_report=diagnostic_report,
                tool_results=tool_results,
                decision_rationale=llm_decision['decision_rationale']
            )
            
            return context_object
            
        except (LLMRetryExhausted, LLMParsingError, Exception) as e:
            # Log the error
            if self.config.verbose:
                print(f"[AGENT A] LLM orchestration failed: {type(e).__name__}: {e}")
                print("[AGENT A] Falling back to rule-based orchestration")
            
            # Log error with context
            log_llm_error(
                e,
                {
                    'agent': 'Agent A',
                    'subject_id': subject_id,
                    'fallback': 'rule-based orchestration'
                }
            )
            
            self._log_reasoning(f"LLM orchestration failed: {type(e).__name__}. Using rule-based fallback.")
            
            # Fallback to rule-based orchestration (Requirement 10.2)
            return self._orchestrate_with_rules(subject_id)
    
    def _orchestrate_with_rules(self, subject_id: str) -> ContextObject:
        """
        Orchestrate using rule-based logic (fallback)
        
        Requirements: 10.2
        """
        if self.config.verbose:
            print("[AGENT A] Using rule-based orchestration")
        
        # Step 1: Read diagnostic report
        diagnostic_report = self._read_diagnostic_report(subject_id)
        
        # Step 2: Evaluate signals
        uq_score = diagnostic_report.uq_score
        has_anomaly = diagnostic_report.anomaly_status.has_anomaly
        
        self._log_reasoning(f"Evaluated signals: UQ={uq_score:.3f}, Anomaly={has_anomaly}")
        
        # Step 3: Apply decision rules
        tool_results = {}
        decision_rationale = ""
        
        # Rule A: High UQ → Counterfactual
        if uq_score > self.config.uq_threshold:
            self._log_reasoning(
                f"High UQ detected ({uq_score:.3f} > {self.config.uq_threshold}). "
                "Triggering counterfactual simulation."
            )
            
            # Get top features for simulation
            top_features = [f.roi_name for f in diagnostic_report.top_features[:3]]
            
            # Call counterfactual tool
            cf_result = self._call_counterfactual_tool(subject_id, top_features)
            tool_results['counterfactual'] = cf_result
            
            decision_rationale += f"High uncertainty (UQ={uq_score:.3f}). Simulated counterfactual. "
        
        # Rule B: Anomaly → Knowledge Graph
        if has_anomaly:
            anomalous_regions = diagnostic_report.anomaly_status.anomalous_regions
            
            self._log_reasoning(
                f"Anomalies detected in {len(anomalous_regions)} regions. "
                "Querying knowledge graph."
            )
            
            # Query knowledge for each anomalous region
            knowledge_contexts = []
            for region in anomalous_regions[:5]:  # Limit to top 5
                context = self._read_knowledge_context(region)
                if context:
                    knowledge_contexts.append(context)
            
            if knowledge_contexts:
                tool_results['knowledge_context'] = {
                    'query_regions': anomalous_regions,
                    'contexts': knowledge_contexts,
                    'summary': self._summarize_knowledge(knowledge_contexts)
                }
            
            decision_rationale += f"Anomalies in {len(anomalous_regions)} regions. Retrieved clinical context. "
        
        # Rule C: Standard case
        if not decision_rationale:
            decision_rationale = "Standard case: low uncertainty, no anomalies. Proceeding to synthesis."
            self._log_reasoning(decision_rationale)
        
        # Step 4: Compile ContextObject
        context_object = self._compile_context_object(
            subject_id=subject_id,
            diagnostic_report=diagnostic_report,
            tool_results=tool_results,
            decision_rationale=decision_rationale.strip()
        )
        
        return context_object
    
    # ========================================================================
    # LLM Integration (Subtask 3.2)
    # ========================================================================
    
    def _get_llm_decision(
        self, 
        subject_id: str, 
        diagnostic_report: DiagnosticReport
    ) -> Dict[str, Any]:
        """
        Get decision from LLM about which MCP actions to take
        
        Requirements: 1.1, 1.2, 9.1
        """
        if self.config.verbose:
            print(f"[AGENT A] Consulting LLM ({self.config.model}) for decision...")
        
        # Check if model is available based on provider
        if self.config.provider == "huggingface":
            model_info = huggingface.get_model_info(self.config.model_path)
            if not model_info['exists']:
                if self.config.verbose:
                    print(f"[WARNING] Model not found at: {self.config.model_path}")
                    print(f"[INFO] Please ensure the model is downloaded")
        else:  # ollama
            available_models = ollama.list_models()
            if self.config.model not in available_models:
                if self.config.verbose:
                    print(f"[WARNING] Model '{self.config.model}' not found in Ollama")
                    print(f"[INFO] Available models: {', '.join(available_models) if available_models else 'None'}")
                    print(f"[INFO] To install: ollama pull {self.config.model}")
                    print(f"[INFO] Or use alternative: ollama pull llama3.1:8b")
        
        # Format diagnostic data for LLM
        diagnostic_summary = {
            'subject_id': subject_id,
            'prediction': diagnostic_report.prediction_result,
            'confidence': diagnostic_report.confidence,
            'uq_score': diagnostic_report.uq_score,
            'has_anomaly': diagnostic_report.anomaly_status.has_anomaly,
            'anomalous_regions': diagnostic_report.anomaly_status.anomalous_regions,
            'top_features': [
                {
                    'roi_name': f.roi_name,
                    'z_score': f.z_score,
                    'shap_value': f.shap_value
                }
                for f in diagnostic_report.top_features[:5]
            ]
        }
        
        # Create user prompt
        user_prompt = f"""
Based on the diagnostic data below, decide which MCP actions to take.

DIAGNOSTIC DATA:
{json.dumps(diagnostic_summary, indent=2)}

DECISION THRESHOLDS:
- UQ Threshold: {self.config.uq_threshold}
- Z-Score Threshold: {self.config.z_score_threshold}

Respond with JSON containing:
1. "actions": List of MCP actions to execute
2. "decision_rationale": Explanation of your decisions
"""
        
        # Call LLM based on provider
        try:
            if self.config.provider == "huggingface":
                response_text = huggingface.handle_text(
                    prompt=user_prompt,
                    model_path=self.config.model_path,
                    system_instruction=self.system_prompt,
                    temperature=self.config.temperature,
                    max_new_tokens=512,
                    load_in_8bit=self.config.load_in_8bit
                )
            else:  # ollama
                response_text = ollama.handle_text(
                    prompt=user_prompt,
                    model=self.config.model,
                    system_instruction=self.system_prompt,
                    temperature=self.config.temperature
                )
            
            # Parse JSON response
            response_json = self._parse_llm_response(response_text)
            
            self._log_reasoning(f"LLM decision: {response_json['decision_rationale']}")
            
            return response_json
            
        except Exception as e:
            if self.config.verbose:
                print(f"[AGENT A] LLM call failed: {e}")
            raise e
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured format with error recovery
        
        Uses robust JSON parsing with multiple recovery strategies.
        
        Requirements: 2.3, 10.1
        """
        try:
            # Use robust JSON parsing with recovery
            parsed = parse_json_with_recovery(response_text, verbose=self.config.verbose)
            
            # Validate structure
            if 'actions' not in parsed:
                if self.config.verbose:
                    print("[AGENT A] Response missing 'actions' field, adding empty list")
                parsed['actions'] = []
            
            if 'decision_rationale' not in parsed:
                if self.config.verbose:
                    print("[AGENT A] Response missing 'decision_rationale' field, adding default")
                parsed['decision_rationale'] = "No rationale provided by LLM"
            
            return parsed
            
        except LLMParsingError as e:
            if self.config.verbose:
                print(f"[AGENT A] Failed to parse LLM response: {e}")
                print(f"[AGENT A] Raw response preview: {response_text[:200]}...")
            
            # Log the parsing error
            log_llm_error(
                e,
                {
                    'agent': 'Agent A',
                    'function': '_parse_llm_response',
                    'response_preview': response_text[:200]
                }
            )
            
            raise LLMParsingError(f"Failed to parse LLM response as JSON: {e}") from e
    
    def _execute_llm_actions(self, actions: List[Dict]) -> Dict[str, Any]:
        """
        Execute MCP actions decided by LLM
        
        Requirements: 2.4, 2.5
        """
        tool_results = {}
        
        for action_dict in actions:
            action_type = action_dict.get('type')
            
            if action_type == 'read_resource':
                uri = action_dict.get('uri')
                if uri and uri.startswith('knowledge://'):
                    # Extract region name from URI
                    region_name = uri.split('//')[1].split('/')[0]
                    context = self._read_knowledge_context(region_name)
                    if context:
                        if 'knowledge_context' not in tool_results:
                            tool_results['knowledge_context'] = {
                                'query_regions': [],
                                'contexts': [],
                                'summary': ''
                            }
                        tool_results['knowledge_context']['query_regions'].append(region_name)
                        tool_results['knowledge_context']['contexts'].append(context)
            
            elif action_type == 'call_tool':
                tool_name = action_dict.get('name')
                tool_args = action_dict.get('args', {})
                
                if tool_name == 'simulate_counterfactual':
                    subject_id = tool_args.get('subject_id')
                    features_to_mask = tool_args.get('features_to_mask', [])
                    
                    cf_result = self._call_counterfactual_tool(subject_id, features_to_mask)
                    tool_results['counterfactual'] = cf_result
        
        # Generate summary for knowledge context if present
        if 'knowledge_context' in tool_results and tool_results['knowledge_context']['contexts']:
            tool_results['knowledge_context']['summary'] = self._summarize_knowledge(
                tool_results['knowledge_context']['contexts']
            )
        
        return tool_results
    
    # ========================================================================
    # MCP Client Methods
    # ========================================================================
    
    def _read_diagnostic_report(self, subject_id: str) -> DiagnosticReport:
        """
        Read diagnostic report from MCP server
        
        Requirements: 3.1, 4.1
        """
        uri = f"diagnosis://{subject_id}/report"
        
        if self.config.verbose:
            print(f"[AGENT A] Reading resource: {uri}")
        
        # Create MCP action
        action = MCPAction(
            type="read_resource",
            target=uri
        )
        
        try:
            # Call MCP server
            result = self.mcp_server.read_resource(uri)
            
            if self.config.verbose:
                print(f"[DEBUG] MCP server returned: {type(result)}")
                if isinstance(result, dict):
                    print(f"[DEBUG] Keys: {list(result.keys())}")
            
            # Mark action as successful
            action.mark_success(result)
            
            # Log action
            self.mcp_actions.append(action)
            self._log_reasoning(f"Read diagnostic report for {subject_id}")
            
            # Check for errors in result
            if 'error' in result:
                raise ValueError(f"MCP server returned error: {result['error']}")
            
            # Ensure subject_id is present
            if 'subject_id' not in result:
                if self.config.verbose:
                    print(f"[WARNING] subject_id not in result, adding it")
                result['subject_id'] = subject_id
            
            # Convert to DiagnosticReport object
            # MCP server now returns flattened structure
            if 'data' in result:
                # Old nested structure
                if self.config.verbose:
                    print(f"[DEBUG] Using nested structure (result['data'])")
                data = result['data']
                if 'subject_id' not in data:
                    data['subject_id'] = subject_id
                diagnostic_report = DiagnosticReport.from_toolkit_report(data)
            else:
                # New flattened structure
                if self.config.verbose:
                    print(f"[DEBUG] Using flattened structure")
                diagnostic_report = DiagnosticReport.from_toolkit_report(result)
            
            if self.config.verbose:
                print(f"[DEBUG] DiagnosticReport created: subject_id={diagnostic_report.subject_id}")
            
            return diagnostic_report
            
        except Exception as e:
            # Mark action as failed
            action.mark_error(str(e))
            self.mcp_actions.append(action)
            self._log_reasoning(f"Failed to read diagnostic report: {e}")
            
            # Print detailed error info
            if self.config.verbose:
                print(f"[ERROR] Failed to read diagnostic report: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            
            raise e
    
    def _read_knowledge_context(self, region_name: str) -> Optional[Dict]:
        """
        Read knowledge context from MCP server
        
        Includes error tracking for GraphRAG fallback.
        
        Requirements: 4.1, 4.2, 10.4
        """
        uri = f"knowledge://{region_name}/context"
        
        if self.config.verbose:
            print(f"[AGENT A] Reading resource: {uri}")
        
        # Create MCP action
        action = MCPAction(
            type="read_resource",
            target=uri
        )
        
        try:
            # Call MCP server
            result = self.mcp_server.read_resource(uri)
            
            # Mark action as successful
            action.mark_success(result)
            
            # Log action
            self.mcp_actions.append(action)
            
            # Handle both nested and flattened structures
            if 'data' in result:
                # Old nested structure
                data = result['data']
            else:
                # New flattened structure
                data = result
            
            # Check if fallback was used (Requirement 10.4)
            if data.get('fallback', False):
                self._log_reasoning(
                    f"Retrieved knowledge context for {region_name} (using fallback knowledge base)"
                )
                
                # Note: Error will be added to ContextObject during compilation
            else:
                self._log_reasoning(f"Retrieved knowledge context for {region_name}")
            
            # Extract context data
            context_data = data.get('context', data)
            
            return {
                'region': region_name,
                'context': context_data,
                'fallback': data.get('fallback', False),
                'error': data.get('error')
            }
            
        except Exception as e:
            # Mark action as failed
            action.mark_error(str(e))
            self.mcp_actions.append(action)
            self._log_reasoning(f"Failed to retrieve knowledge for {region_name}: {e}")
            return None
    
    def _call_counterfactual_tool(
        self, 
        subject_id: str, 
        features_to_mask: List[str]
    ) -> Dict:
        """
        Call counterfactual simulation tool
        
        Requirements: 2.4, 2.5, 7.1
        """
        if self.config.verbose:
            print(f"[AGENT A] Calling tool: simulate_counterfactual")
            print(f"[AGENT A] Masking features: {features_to_mask}")
        
        # Create MCP action
        action = MCPAction(
            type="call_tool",
            target="simulate_counterfactual",
            arguments={
                'subject_id': subject_id,
                'features_to_mask': features_to_mask
            }
        )
        
        try:
            # Call MCP server
            result = self.mcp_server.call_tool(
                "simulate_counterfactual",
                {
                    'subject_id': subject_id,
                    'features_to_mask': features_to_mask
                }
            )
            
            # Mark action as successful
            action.mark_success(result)
            
            # Log action
            self.mcp_actions.append(action)
            self._log_reasoning(
                f"Simulated counterfactual: masked {len(features_to_mask)} features"
            )
            
            # MCP server now returns flattened structure
            if 'data' in result:
                # Old nested structure
                return result['data']
            else:
                # New flattened structure - return the whole result minus metadata
                return {k: v for k, v in result.items() if k not in ['tool', 'status', 'timestamp']}
            
        except Exception as e:
            # Mark action as failed
            action.mark_error(str(e))
            self.mcp_actions.append(action)
            self._log_reasoning(f"Counterfactual simulation failed: {e}")
            raise e
    
    # ========================================================================
    # Context Compilation
    # ========================================================================
    
    def _compile_context_object(
        self,
        subject_id: str,
        diagnostic_report: DiagnosticReport,
        tool_results: Dict[str, Any],
        decision_rationale: str
    ) -> ContextObject:
        """
        Compile ContextObject for handoff to Agent B
        
        Includes error annotations for any fallbacks or failures.
        
        Requirements: 5.1, 8.3, 10.4, 10.5
        """
        if self.config.verbose:
            print("[AGENT A] Compiling ContextObject for Agent B")
        
        # Extract signals
        signals = {
            'uq_score': diagnostic_report.uq_score,
            'has_anomaly': diagnostic_report.anomaly_status.has_anomaly,
            'anomalous_regions': diagnostic_report.anomaly_status.anomalous_regions,
            'prediction': diagnostic_report.prediction_result,
            'confidence': diagnostic_report.confidence
        }
        
        # Create ContextObject
        context_object = ContextObject(
            subject_id=subject_id,
            diagnostic_report=diagnostic_report,
            tool_results=tool_results if tool_results else None,
            decision_rationale=decision_rationale,
            signals=signals,
            agent_a_reasoning=self.reasoning_chain.copy(),
            mcp_actions=self.mcp_actions.copy()
        )
        
        # Add error annotations for GraphRAG fallback (Requirement 10.4, 10.5)
        if tool_results and 'knowledge_context' in tool_results:
            kc = tool_results['knowledge_context']
            for ctx in kc.get('contexts', []):
                if ctx.get('fallback', False) and ctx.get('error'):
                    error_info = ctx['error']
                    context_object.add_error(
                        error_type=error_info.get('type', 'GraphRAGError'),
                        error_message=error_info.get('message', 'GraphRAG query failed'),
                        component='MCP Server - Knowledge Resource'
                    )
                    self._log_reasoning(
                        f"Added error annotation: GraphRAG fallback for {ctx['region']}"
                    )
        
        # Add error annotations for failed MCP actions
        for action in self.mcp_actions:
            if hasattr(action, 'status') and action.status == 'error':
                context_object.add_error(
                    error_type='MCPActionError',
                    error_message=action.error.get('message', 'MCP action failed') if hasattr(action, 'error') else 'Unknown error',
                    component=f'Agent A - {action.type}'
                )
        
        # Validate
        if not context_object.validate():
            raise ValueError("ContextObject validation failed")
        
        self._log_reasoning("ContextObject compiled and validated")
        
        if context_object.has_errors():
            self._log_reasoning(f"ContextObject contains {len(context_object.errors)} error annotation(s)")
        
        return context_object
    
    # ========================================================================
    # Reasoning Chain Logging (Subtask 3.4)
    # ========================================================================
    
    def _log_reasoning(self, message: str):
        """
        Log reasoning step with timestamp
        
        Requirements: 3.5, 8.1, 8.2
        """
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [Agent A] {message}"
        
        self.reasoning_chain.append(log_entry)
        
        if self.config.verbose:
            print(f"[REASONING] {message}")
    
    def save_reasoning_log(self, output_path: str):
        """
        Save reasoning chain to file for paper evidence
        
        Requirements: 3.5, 8.1, 8.2
        """
        log_data = {
            'agent': 'Agent A - Orchestrator',
            'timestamp': datetime.now().isoformat(),
            'reasoning_chain': self.reasoning_chain,
            'mcp_actions': [action.to_dict() for action in self.mcp_actions]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        if self.config.verbose:
            print(f"[AGENT A] Reasoning log saved to: {output_path}")
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _summarize_knowledge(self, contexts: List[Dict]) -> str:
        """Generate natural language summary of knowledge contexts"""
        if not contexts:
            return "No clinical context available."
        
        summaries = []
        for ctx in contexts:
            region = ctx['region']
            info = ctx['context']
            
            if info.get('related_conditions'):
                conditions = ', '.join(info['related_conditions'][:2])
                summary = f"{region}: {info.get('clinical_significance', 'Unknown')}. Related to {conditions}."
            else:
                summary = f"{region}: {info.get('clinical_significance', 'Unknown')}"
            
            summaries.append(summary)
        
        return ' '.join(summaries)


# ============================================================================
# Demo Functions
# ============================================================================

def demo_agent_a_rule_based():
    """Demo: Agent A with rule-based orchestration"""
    print("\n" + "="*80)
    print("DEMO: Agent A - Rule-Based Orchestration")
    print("="*80)
    
    # Initialize MCP server
    from app.core.ml_processing.cdda_tools import CDDAToolKit
    from app.core.knowledge.graph_rag import GraphRAG
    
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # Initialize Agent A (rule-based)
    config = AgentAConfig(use_llm=False, verbose=True)
    agent_a = AgentA(mcp_server=mcp_server, config=config)
    
    # Run orchestration
    context_object = agent_a.orchestrate('sub-0005')
    
    # Print results
    print("\n" + "-"*80)
    print("CONTEXT OBJECT FOR AGENT B:")
    print("-"*80)
    print(f"Subject: {context_object.subject_id}")
    print(f"Prediction: {context_object.diagnostic_report.prediction_result}")
    print(f"Confidence: {context_object.diagnostic_report.confidence:.1%}")
    print(f"UQ Score: {context_object.diagnostic_report.uq_score:.3f}")
    print(f"Decision Rationale: {context_object.decision_rationale}")
    print(f"\nReasoning Chain ({len(context_object.agent_a_reasoning)} steps):")
    for step in context_object.agent_a_reasoning:
        print(f"  {step}")
    print("-"*80)


def demo_agent_a_with_llm():
    """Demo: Agent A with LLM-based orchestration"""
    print("\n" + "="*80)
    print("DEMO: Agent A - LLM-Based Orchestration (Phi-4-mini)")
    print("="*80)
    
    # Check if Ollama is available
    if not ollama.check_availability():
        print("[WARNING] Ollama not available. Skipping LLM demo.")
        print("To run this demo:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Pull model: ollama pull gpt-oss-20b")
        print("     (Alternative: ollama pull llama3.1:8b)")
        print("  3. Start server: ollama serve")
        return
    
    # Initialize MCP server
    from app.core.ml_processing.cdda_tools import CDDAToolKit
    from app.core.knowledge.graph_rag import GraphRAG
    
    toolkit = CDDAToolKit(
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root="data/MRI_processed"
    )
    graph_rag = GraphRAG()
    mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag, verbose=False)
    
    # Initialize Agent A (LLM-based)
    config = AgentAConfig(use_llm=True, verbose=True)
    agent_a = AgentA(mcp_server=mcp_server, config=config)
    
    # Run orchestration
    context_object = agent_a.orchestrate('sub-0005')
    
    # Print results
    print("\n" + "-"*80)
    print("CONTEXT OBJECT FOR AGENT B:")
    print("-"*80)
    print(f"Subject: {context_object.subject_id}")
    print(f"Prediction: {context_object.diagnostic_report.prediction_result}")
    print(f"Confidence: {context_object.diagnostic_report.confidence:.1%}")
    print(f"UQ Score: {context_object.diagnostic_report.uq_score:.3f}")
    print(f"Decision Rationale: {context_object.decision_rationale}")
    print(f"\nReasoning Chain ({len(context_object.agent_a_reasoning)} steps):")
    for step in context_object.agent_a_reasoning:
        print(f"  {step}")
    print("-"*80)
    
    # Save reasoning log
    agent_a.save_reasoning_log("output/agent_a_reasoning_log.json")


if __name__ == "__main__":
    # Run demos
    demo_agent_a_rule_based()
    print("\n\n")
    demo_agent_a_with_llm()
