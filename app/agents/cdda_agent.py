"""
CDDA Agent - Layer 3: Cognitive/Orchestration (A2A Pattern)

This module implements the autonomous CDDA Agent using the Agent-to-Agent (A2A)
pattern with dual-LLM architecture:

- Agent A (Orchestrator): Reads resources, invokes tools, compiles context
- Agent B (Consultant): Synthesizes clinical reports from provided context

The A2A pattern ensures clear separation of concerns:
- Agent A handles tool orchestration and data gathering
- Agent B focuses purely on clinical reasoning and synthesis
- Handoff via ContextObject ensures Agent B has no direct tool access

Requirements: 1.1, 1.2, 1.3, 3.1, 8.3, 8.4
Reference: docs/CDDA_Architecture_Spec.md
"""

import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Helper Functions
# ============================================================================

def _safe_get_attr(obj: Any, attr_name: str, default: Any = None) -> Any:
    """
    安全地從對象或字典中獲取屬性
    
    支持 Feature dataclass 對象和字典兩種格式
    
    Args:
        obj: Feature 對象或字典
        attr_name: 屬性名稱
        default: 預設值
    
    Returns:
        屬性值或預設值
    """
    if isinstance(obj, dict):
        return obj.get(attr_name, default)
    else:
        return getattr(obj, attr_name, default)

from app.core.ml_processing.cdda_tools import CDDAToolKit
from app.core.knowledge.graph_rag import GraphRAG
from app.core.mcp_server import DiagnosticMCPServer
from app.core.models import AgentResult, ContextObject
from app.agents.agent_a_orchestrator import AgentA, AgentAConfig
from app.agents.agent_b_consultant import AgentB, AgentBConfig


class CDDAAgent:
    """
    Cognitive Discrepancy-Driven Agent (A2A Pattern)
    
    Implements dual-LLM architecture with Agent-to-Agent handoff:
    - Agent A (Orchestrator): MCP client, reads resources, invokes tools
    - Agent B (Consultant): Medical specialist, synthesizes clinical reports
    
    The A2A pattern ensures:
    - Clear separation between orchestration and clinical reasoning
    - Agent B has no direct tool access (receives ContextObject only)
    - Complete reasoning chain from both agents for transparency
    
    Requirements: 1.1, 1.2, 1.3, 3.1
    """
    
    def __init__(
        self,
        orchestrator_model: str = "phi-4-mini",
        orchestrator_model_path: Optional[str] = "D:/hf_models/Phi-4-mini-instruct",
        consultant_model: str = "llama3.1-aloe-beta-8b",
        consultant_model_path: Optional[str] = "D:/hf_models/Llama3.1-Aloe-Beta-8B",
        model_path: str = "model/cnn_rf/rf_model_NC_MCI_AD.joblib",
        data_root: str = "data/MRI_processed",
        uq_threshold: float = 0.8,
        z_score_threshold: float = 2.5,
        use_llm: bool = True,
        use_4bit: bool = True,
        verbose: bool = True
    ):
        """
        Initialize CDDA Agent with A2A architecture
        
        Args:
            orchestrator_model: Model for Agent A (default: "phi-4-mini")
            orchestrator_model_path: Path for Phi-4-mini model (Agent A)
            consultant_model: Model for Agent B (default: "llama3.1-aloe-beta-8b")
            consultant_model_path: Path for Llama3.1-Aloe-Beta-8B model (Agent B)
            model_path: Path to trained CNN-RF model (default: 3-class model)
            data_root: Root directory for MRI data
            uq_threshold: Threshold for high uncertainty trigger
            z_score_threshold: Threshold for anomaly detection
            use_llm: Enable LLM-based agents (if False, use rule-based fallback)
            use_4bit: Use 4-bit quantization to save VRAM (recommended)
            verbose: Print detailed information
        """
        self.verbose = verbose
        
        if self.verbose:
            print("\n" + "="*80)
            print("CDDA AGENT - A2A Dual-LLM Architecture")
            print("="*80)
            print("Initializing Agent-to-Agent system...")
        
        # Initialize toolkit (Layer 1 + Layer 2)
        if self.verbose:
            print("\n[1/4] Initializing CDDAToolKit (Layer 1+2)...")
        
        self.toolkit = CDDAToolKit(
            model_path=model_path,
            data_root=data_root,
            uq_threshold=uq_threshold,
            z_score_threshold=z_score_threshold
        )
        
        # Initialize GraphRAG (Layer 4)
        if self.verbose:
            print("\n[2/4] Initializing GraphRAG (Layer 4)...")
        
        self.graph_rag = GraphRAG()
        
        # Initialize MCP Server (Context Layer)
        if self.verbose:
            print("\n[3/4] Initializing DiagnosticMCPServer...")
        
        self.mcp_server = DiagnosticMCPServer(
            toolkit=self.toolkit,
            graph_rag=self.graph_rag,
            verbose=False  # Reduce noise
        )
        
        # Initialize Agent A (Orchestrator)
        if self.verbose:
            print("\n[4/4] Initializing A2A Agents...")
            print(f"   Agent A (Orchestrator): {orchestrator_model}")
            if orchestrator_model_path:
                print(f"      Path: {orchestrator_model_path}")
                print(f"      Provider: HuggingFace")
                print(f"      Quantization: {'4-bit' if use_4bit else '8-bit'}")
            else:
                print(f"      Provider: Ollama (fallback)")
        
        agent_a_config = AgentAConfig(
            model=orchestrator_model,
            model_path=orchestrator_model_path,
            provider="huggingface" if orchestrator_model_path else "ollama",
            temperature=0.1,
            uq_threshold=uq_threshold,
            z_score_threshold=z_score_threshold,
            use_llm=use_llm,
            load_in_8bit=not use_4bit,  # Use 8-bit only if not using 4-bit
            verbose=verbose  # Use same verbose setting as CDDAAgent
        )
        
        self.agent_a = AgentA(
            mcp_server=self.mcp_server,
            config=agent_a_config
        )
        
        # Initialize Agent B (Consultant)
        if self.verbose:
            print(f"   Agent B (Consultant): {consultant_model}")
            if consultant_model_path:
                print(f"      Path: {consultant_model_path}")
                print(f"      Provider: HuggingFace")
                print(f"      Quantization: {'4-bit' if use_4bit else '8-bit'}")
            else:
                print(f"      Provider: Ollama (fallback)")
        
        agent_b_config = AgentBConfig(
            model=consultant_model,
            model_path=consultant_model_path,
            provider="huggingface" if consultant_model_path else "ollama",
            temperature=0.3,
            use_llm=use_llm,
            load_in_8bit=not use_4bit,  # Use 8-bit only if not using 4-bit
            verbose=verbose  # Use same verbose setting as CDDAAgent
        )
        
        self.agent_b = AgentB(config=agent_b_config)
        
        # Store configuration
        self.uq_threshold = uq_threshold
        self.z_score_threshold = z_score_threshold
        self.use_llm = use_llm
        
        if self.verbose:
            print(f"\n[INFO] Orchestrator: Phi-4-mini | Consultant: Llama3.1-Aloe-Beta-8B")
            print(f"[OK] CDDA Agent ready (A2A mode)")
            print(f"   Decision Thresholds:")
            print(f"      UQ > {uq_threshold} → Trigger Counterfactual Simulation")
            print(f"      |Z| > {z_score_threshold} → Trigger Knowledge Lookup")
            print(f"   LLM Mode: {'Enabled' if use_llm else 'Rule-based fallback'}")
            print(f"   Quantization: {'4-bit' if use_4bit else '8-bit'}")
            print("="*80)
    
    # ========================================================================
    # Reasoning Chain Aggregation (Subtask 5.2)
    # ========================================================================
    
    def _aggregate_reasoning_chains(
        self,
        context_object: ContextObject,
        agent_b_reasoning: List[str]
    ) -> List[str]:
        """
        Combine Agent A's reasoning with Agent B's reasoning
        
        This method aggregates the complete reasoning chain from both agents,
        including MCP actions with timestamps, to provide full transparency
        for paper evidence and debugging.
        
        Requirements: 8.3, 8.4
        
        Args:
            context_object: ContextObject from Agent A
            agent_b_reasoning: Reasoning chain from Agent B
        
        Returns:
            Combined reasoning chain with timestamps
        """
        combined_reasoning = []
        
        # Section 1: Agent A Orchestration
        combined_reasoning.append("="*80)
        combined_reasoning.append("AGENT A - ORCHESTRATION")
        combined_reasoning.append("="*80)
        
        # Add Agent A's reasoning steps
        for step in context_object.agent_a_reasoning:
            combined_reasoning.append(step)
        
        # Section 2: MCP Actions (with timestamps)
        if context_object.mcp_actions:
            combined_reasoning.append("")
            combined_reasoning.append("-"*80)
            combined_reasoning.append("MCP ACTIONS")
            combined_reasoning.append("-"*80)
            
            for action in context_object.mcp_actions:
                # Format MCP action with timestamp
                action_dict = action.to_dict() if hasattr(action, 'to_dict') else action
                
                action_type = action_dict.get('type', 'unknown')
                target = action_dict.get('target', 'unknown')
                timestamp = action_dict.get('timestamp', 'N/A')
                status = action_dict.get('status', 'unknown')
                
                action_line = f"[{timestamp}] {action_type}: {target} → {status}"
                combined_reasoning.append(action_line)
                
                # Add error details if failed
                if status == 'error' and 'error' in action_dict:
                    error_msg = action_dict['error'].get('message', 'Unknown error')
                    combined_reasoning.append(f"  ERROR: {error_msg}")
        
        # Section 3: Handoff
        combined_reasoning.append("")
        combined_reasoning.append("-"*80)
        combined_reasoning.append("HANDOFF: Agent A → Agent B")
        combined_reasoning.append("-"*80)
        combined_reasoning.append(f"Decision Rationale: {context_object.decision_rationale}")
        combined_reasoning.append(f"Context Object validated: {context_object.validate()}")
        
        # Section 4: Agent B Synthesis
        combined_reasoning.append("")
        combined_reasoning.append("="*80)
        combined_reasoning.append("AGENT B - CLINICAL SYNTHESIS")
        combined_reasoning.append("="*80)
        
        # Add Agent B's reasoning steps
        for step in agent_b_reasoning:
            combined_reasoning.append(step)
        
        return combined_reasoning
    
    def save_reasoning_log(self, result: AgentResult, output_path: str):
        """
        Save complete reasoning trace to structured log file
        
        This method saves the full reasoning chain to a JSON file for:
        - Paper evidence generation
        - Debugging and analysis
        - Transparency and auditability
        
        Requirements: 8.3, 8.4
        
        Args:
            result: AgentResult with complete reasoning chain
            output_path: Path to save log file
        """
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
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        if self.verbose:
            print(f"\n[LOG] Reasoning log saved to: {output_path}")
    
    def generate_executive_summary(
        self,
        clinical_report: str,
        context_object: ContextObject
    ) -> Dict:
        """
        Generate executive summary using Agent A (Phi-4) for rapid clinical review
        
        This method uses the orchestrator LLM to extract key information from
        the detailed clinical report and structure it into a JSON format suitable
        for dashboard widgets.
        
        Args:
            clinical_report: Full text report from Agent B
            context_object: ContextObject with diagnostic information
        
        Returns:
            Dictionary with structured summary:
            {
                'headline': str,
                'key_findings': List[str],
                'recommended_actions': List[str],
                'risk_level': str
            }
        """
        if self.verbose:
            print(f"\n[SUMMARIZATION] Generating executive summary...")
        
        # Extract key information from context
        prediction = context_object.diagnostic_report.prediction_result
        confidence = context_object.diagnostic_report.confidence
        uq_score = context_object.diagnostic_report.uq_score
        
        # Determine risk level
        if uq_score > 0.8 or confidence < 0.6:
            risk_level = "High"
        elif uq_score > 0.5 or confidence < 0.8:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # If LLM mode is disabled, use rule-based summary
        if not self.use_llm:
            return self._generate_rule_based_summary(
                clinical_report, context_object, risk_level
            )
        
        # Construct prompt for Phi-4
        prompt = f"""You are a Medical Secretary. Read the following clinical report and extract key information into strict JSON format.

CLINICAL REPORT:
{clinical_report[:2000]}  

DIAGNOSTIC DATA:
- Prediction: {prediction}
- Confidence: {confidence:.1%}
- Uncertainty Score: {uq_score:.3f}

Extract the following information and output ONLY valid JSON (no markdown, no explanation):

{{
  "headline": "Short 1-sentence summary (e.g., 'Probable AD with high confidence and hippocampal atrophy')",
  "key_findings": [
    "Finding 1 (focus on top brain regions and their clinical significance)",
    "Finding 2 (mention any anomalies or counterfactual results)",
    "Finding 3 (note uncertainty or confidence issues)"
  ],
  "recommended_actions": [
    "Action 1 (e.g., 'Clinical correlation recommended')",
    "Action 2 (e.g., 'Follow-up imaging in 6 months')"
  ],
  "risk_level": "{risk_level}"
}}

Output ONLY the JSON object, nothing else:"""
        
        try:
            # Use Agent A's LLM (Phi-4) for summarization
            if hasattr(self.agent_a, 'llm') and self.agent_a.llm and hasattr(self.agent_a, 'tokenizer'):
                # Generate using the model directly
                inputs = self.agent_a.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.agent_a.llm.device)
                
                with torch.no_grad():
                    outputs = self.agent_a.llm.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.agent_a.tokenizer.eos_token_id
                    )
                
                response = self.agent_a.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # Parse JSON response
                import json
                import re
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    summary = json.loads(json_match.group(0))
                    
                    # Validate required fields
                    required_fields = ['headline', 'key_findings', 'recommended_actions', 'risk_level']
                    if all(field in summary for field in required_fields):
                        if self.verbose:
                            print(f"   LLM-generated summary: {summary.get('headline', 'N/A')}")
                        return summary
                    else:
                        if self.verbose:
                            print(f"   [WARNING] Incomplete JSON, using rule-based fallback")
                        return self._generate_rule_based_summary(
                            clinical_report, context_object, risk_level
                        )
                else:
                    if self.verbose:
                        print(f"   [WARNING] Failed to parse JSON, using rule-based fallback")
                    return self._generate_rule_based_summary(
                        clinical_report, context_object, risk_level
                    )
            else:
                # No LLM available, use rule-based
                if self.verbose:
                    print(f"   [INFO] LLM not available, using rule-based summary")
                return self._generate_rule_based_summary(
                    clinical_report, context_object, risk_level
                )
        
        except Exception as e:
            if self.verbose:
                print(f"   [WARNING] Summarization failed: {e}, using rule-based fallback")
            
            return self._generate_rule_based_summary(
                clinical_report, context_object, risk_level
            )
    
    def _generate_rule_based_summary(
        self,
        clinical_report: str,
        context_object: ContextObject,
        risk_level: str
    ) -> Dict:
        """
        Generate executive summary using rule-based logic (fallback)
        
        Args:
            clinical_report: Full text report
            context_object: ContextObject with diagnostic information
            risk_level: Pre-calculated risk level
        
        Returns:
            Dictionary with structured summary
        """
        prediction = context_object.diagnostic_report.prediction_result
        confidence = context_object.diagnostic_report.confidence
        uq_score = context_object.diagnostic_report.uq_score
        
        # Generate headline
        diagnosis_map = {
            'AD': 'Alzheimer\'s Disease',
            'MCI': 'Mild Cognitive Impairment',
            'NC': 'Normal Cognition'
        }
        diagnosis_text = diagnosis_map.get(prediction, prediction)
        
        if confidence > 0.8:
            headline = f"Probable {diagnosis_text} with high confidence"
        elif confidence > 0.6:
            headline = f"Possible {diagnosis_text} with moderate confidence"
        else:
            headline = f"Uncertain diagnosis, {diagnosis_text} suggested"
        
        # Extract key findings
        key_findings = []
        
        # Add top features
        top_features = context_object.diagnostic_report.top_features[:3]
        if top_features:
            regions = [_safe_get_attr(f, 'roi_name', 'Unknown') for f in top_features]
            key_findings.append(f"Primary drivers: {', '.join(regions)}")
        
        # Add counterfactual info
        tool_results = context_object.tool_results or {}
        if 'counterfactual' in tool_results:
            cf = tool_results['counterfactual']
            confidence_delta = cf.get('confidence_delta', 0)
            key_findings.append(
                f"Counterfactual analysis shows {abs(confidence_delta):.1%} impact on confidence"
            )
        
        # Add anomaly info
        if 'knowledge_context' in tool_results:
            kc = tool_results['knowledge_context']
            anomalous_regions = kc.get('query_regions', [])
            if anomalous_regions:
                key_findings.append(
                    f"Detected {len(anomalous_regions)} anomalous regions suggesting mixed pathology"
                )
        
        # Add uncertainty info
        if uq_score > 0.8:
            key_findings.append(f"High uncertainty (UQ: {uq_score:.3f}) - additional validation recommended")
        
        # Generate recommended actions
        recommended_actions = []
        
        if uq_score > 0.8:
            recommended_actions.append("Clinical correlation strongly recommended")
            recommended_actions.append("Consider additional imaging or biomarker testing")
        elif uq_score > 0.5:
            recommended_actions.append("Clinical review recommended")
        else:
            recommended_actions.append("Standard clinical follow-up appropriate")
        
        if 'knowledge_context' in tool_results:
            recommended_actions.append("Evaluate for potential mixed pathology")
        
        return {
            'headline': headline,
            'key_findings': key_findings if key_findings else ["Standard diagnostic pattern observed"],
            'recommended_actions': recommended_actions,
            'risk_level': risk_level
        }
    
    def _determine_agent_decision(self, context_object: ContextObject) -> str:
        """
        Determine agent decision type based on context
        
        Args:
            context_object: ContextObject from Agent A
        
        Returns:
            Decision type string
        """
        tool_results = context_object.tool_results or {}
        
        if 'counterfactual' in tool_results:
            return 'SIMULATION_TRIGGERED'
        elif 'knowledge_context' in tool_results:
            return 'ANOMALY_INVESTIGATION'
        else:
            return 'STANDARD_REPORT'
    
    # ========================================================================
    # Legacy Methods (for backward compatibility)
    # ========================================================================
    
    def knowledge_graph_lookup(self, anomalous_regions: List[str]) -> Dict:
        """
        Tool 4: Knowledge Graph Lookup
        
        Queries Neo4j knowledge graph for clinical context about anomalous regions.
        Falls back to mock data if Neo4j is unavailable.
        
        Args:
            anomalous_regions: List of ROI names with anomalies
        
        Returns:
            Dictionary with clinical context
        """
        # Query GraphRAG for each region
        region_contexts = self.graph_rag.query_multiple_regions(
            anomalous_regions,
            max_results=5
        )
        
        # Format contexts for compatibility with existing code
        contexts = []
        for region_info in region_contexts:
            contexts.append({
                'region': region_info['id'],
                'context': {
                    'full_name': region_info.get('full_name', region_info['id']),
                    'function': region_info.get('function', region_info.get('summary', 'Unknown')),
                    'clinical_significance': region_info.get('clinical_significance', region_info.get('summary', 'Requires further investigation')),
                    'related_conditions': region_info.get('related_conditions', []),
                    'is_ad_hotspot': region_info.get('is_ad_hotspot', False)
                }
            })
        
        # Generate summary using GraphRAG
        summary = self.graph_rag.generate_context_summary(region_contexts)
        
        return {
            'query_regions': anomalous_regions,
            'contexts': contexts,
            'summary': summary
        }
    
    def _generate_knowledge_summary(self, contexts: List[Dict]) -> str:
        """Generate natural language summary of knowledge contexts"""
        if not contexts:
            return "No clinical context available."
        
        summaries = []
        for ctx in contexts:
            region = ctx['region']
            info = ctx['context']
            
            if info.get('related_conditions'):
                conditions = ', '.join(info['related_conditions'][:2])
                summary = f"{region} ({info['full_name']}): {info['clinical_significance']}. Related to {conditions}."
            else:
                summary = f"{region}: {info['clinical_significance']}"
            
            summaries.append(summary)
        
        return ' '.join(summaries)
    
    def run_analysis(self, subject_id: str, model_name: str = "NC_MCI_AD") -> AgentResult:
        """
        Main CDDA Agent Analysis (A2A Pattern)
        
        Implements the A2A handoff protocol:
        1. Agent A orchestrates → reads resources, invokes tools, compiles ContextObject
        2. Handoff: Agent A → Agent B via ContextObject
        3. Agent B synthesizes → generates clinical report from context
        4. Aggregate reasoning chains from both agents
        
        Args:
            subject_id: Subject identifier
        
        Returns:
            AgentResult with complete analysis and reasoning chain
        
        Requirements: 1.1, 1.2, 1.3, 3.1, 8.3, 8.4
        """
        if self.verbose:
            print("\n" + "="*80)
            print(f"CDDA AGENT ANALYSIS (A2A): {subject_id}")
            print("="*80)
        
        # ====================================================================
        # PHASE 1: Agent A Orchestration
        # ====================================================================
        
        if self.verbose:
            print(f"\n[PHASE 1] Agent A - Orchestration")
            print("-" * 80)
        
        # [修正 1] 將 model_name 傳遞給 orchestrate
        # Agent A 需要把這個 config 寫入 Graph State，這樣 Tool 才會知道要用哪個 fallback
        context_object = self.agent_a.orchestrate(subject_id, model_name=model_name)
        
        # [修正 2] 確保 Tool 的 Trace Log (包含模型檔名) 被強制保留
        # 有時候 LLM 會忽略這些底層細節，我們手動把它加到 reasoning 的最前面
        # 這樣 comprehensive_statistics.py 就能抓到 "using rf_model_..."
        
        tool_traces = []
        if hasattr(context_object, 'trace_log') and context_object.trace_log:
            tool_traces = context_object.trace_log
        elif hasattr(context_object, 'mcp_actions'):
             # 嘗試從 Action 結果中尋找 Log
             for action in context_object.mcp_actions:
                 # MCPAction 是 dataclass，使用屬性訪問而不是 .get()
                 if hasattr(action, 'result') and action.result and isinstance(action.result, dict):
                     if 'trace_log' in action.result:
                         tool_traces.extend(action.result['trace_log'])

        # 將 Tool Log 注入到 Agent A 的 reasoning 中 (作為 System Observation)
        if tool_traces:
            if self.verbose:
                print(f"   [System] Injected {len(tool_traces)} tool logs into reasoning chain")
            # 加在最前面，模擬系統觀察
            context_object.agent_a_reasoning = tool_traces + context_object.agent_a_reasoning

        if self.verbose:
            print(f"\n[HANDOFF] Agent A → Agent B")
            print(f"   Context compiled: {len(context_object.agent_a_reasoning)} reasoning steps")
        
        # ====================================================================
        # PHASE 2: Agent B Synthesis
        # ====================================================================
        
        if self.verbose:
            print(f"\n[PHASE 2] Agent B - Clinical Synthesis")
            print("-" * 80)
        
        # Agent B synthesizes: generates clinical report from ContextObject
        synthesis_result = self.agent_b.synthesize(context_object)
        
        clinical_report = synthesis_result['clinical_report']
        agent_b_reasoning = synthesis_result['reasoning_chain']
        
        # ====================================================================
        # PHASE 3: Reasoning Chain Aggregation
        # ====================================================================
        
        # Combine reasoning chains
        combined_reasoning = self._aggregate_reasoning_chains(
            context_object=context_object,
            agent_b_reasoning=agent_b_reasoning
        )
        
        # ====================================================================
        # PHASE 4: Post-Processing Summarization
        # ====================================================================
        
        # Generate executive summary
        executive_summary = self.generate_executive_summary(
            clinical_report=clinical_report,
            context_object=context_object
        )
        
        # ====================================================================
        # PHASE 5: Build Final Result
        # ====================================================================
        
        agent_decision = self._determine_agent_decision(context_object)
        
        # Build AgentResult
        result = AgentResult(
            subject_id=subject_id,
            agent_decision=agent_decision,
            prediction=context_object.diagnostic_report.prediction_result,
            confidence=context_object.diagnostic_report.confidence,
            uq_score=context_object.diagnostic_report.uq_score,
            context_object=context_object,
            clinical_report=clinical_report,
            reasoning_chain=combined_reasoning,
            timestamp=datetime.now().isoformat(),
            metadata={
                'agent_a_steps': len(context_object.agent_a_reasoning),
                'agent_b_steps': len(agent_b_reasoning),
                'mcp_actions': len(context_object.mcp_actions),
                'use_llm': self.use_llm,
                'model_used': model_name, # 記錄設定的模型
                'executive_summary': executive_summary
            }
        )
        
        if self.verbose:
            print(f"\n[OK] Analysis complete")
            print(f"   Decision: {agent_decision}")
            print(f"   Prediction: {result.prediction} ({result.confidence:.1%})")
            print("="*80)
        
        return result
    
    def synthesize_simulation_report(
        self, 
        report: Dict, 
        cf_result: Dict
    ) -> Dict:
        """
        Synthesize report emphasizing counterfactual simulation
        
        Logic: "The model was uncertain, but simulation identified the key drivers."
        """
        if self.verbose:
            print(f"\n[SYNTHESIS] Generating simulation-focused report...")
        
        # Calculate impact magnitude
        confidence_change = abs(cf_result['confidence_delta'])
        impact_level = "significant" if confidence_change > 0.1 else "moderate"
        
        # Generate natural language explanation
        explanation = f"""
DIAGNOSTIC ANALYSIS WITH COUNTERFACTUAL SIMULATION

Subject: {report['subject_id']}
Prediction: {report['prediction_result']} (Confidence: {report['confidence']:.1%})

UNCERTAINTY ALERT:
The model exhibited high uncertainty (UQ Score: {report['uq_score']:.3f}), indicating 
the prediction may be sensitive to specific features. To identify key drivers, 
a counterfactual simulation was performed.

COUNTERFACTUAL SIMULATION RESULTS:
Masked Features: {', '.join([_safe_get_attr(f, 'roi_name', 'Unknown') for f in cf_result.get('masked_features', [])[:3]])}

Original Prediction: {cf_result['original_prediction']} ({cf_result['original_confidence']:.1%})
Counterfactual Prediction: {cf_result['new_prediction']} ({cf_result['new_confidence']:.1%})
Confidence Change: {cf_result['confidence_delta']:+.1%}

INTERPRETATION:
{cf_result['interpretation']}

The simulation reveals a {impact_level} impact ({confidence_change:.1%}), suggesting 
these brain regions are {impact_level} contributors to the diagnosis.

TOP CONTRIBUTING FEATURES:
"""
        
        for i, feat in enumerate(report.get('top_features', [])[:5], 1):
            roi_name = _safe_get_attr(feat, 'roi_name', 'Unknown')
            z_score = _safe_get_attr(feat, 'z_score', 0)
            shap_value = _safe_get_attr(feat, 'shap_value', 0)
            explanation += f"\n{i}. {roi_name}: Z-score={z_score:+.2f}, SHAP={shap_value:+.4f}"
        
        explanation += f"""

RECOMMENDATION:
Given the high uncertainty, clinical correlation is recommended. The counterfactual 
analysis suggests focusing on {', '.join([_safe_get_attr(f, 'roi_name', 'Unknown') for f in cf_result.get('masked_features', [])[:2]])} 
for further investigation.
"""
        
        return {
            'subject_id': report['subject_id'],
            'agent_decision': 'SIMULATION_TRIGGERED',
            'prediction': report['prediction_result'],
            'confidence': report['confidence'],
            'uq_score': report['uq_score'],
            'report': report,
            'counterfactual': cf_result,
            'explanation': explanation.strip(),
            'reasoning_chain': [
                f"1. Obtained diagnostic report: {report['prediction_result']} ({report['confidence']:.1%})",
                f"2. Detected high uncertainty: UQ={report['uq_score']:.3f} > {self.uq_threshold}",
                f"3. Triggered counterfactual simulation on top 3 features",
                f"4. Simulation showed {cf_result['confidence_delta']:+.1%} confidence change",
                f"5. Identified key drivers: {', '.join([_safe_get_attr(f, 'roi_name', 'Unknown') for f in cf_result.get('masked_features', [])[:2]])}"
            ],
            'timestamp': datetime.now().isoformat()
        }
    
    def synthesize_anomaly_report(
        self, 
        report: Dict, 
        knowledge_context: Dict
    ) -> Dict:
        """
        Synthesize report acknowledging anomalies with clinical context
        
        Logic: "Model is confident, but data contains unusual patterns. 
                Here is the clinical context."
        """
        if self.verbose:
            print(f"\n[SYNTHESIS] Generating anomaly-focused report...")
        
        # Generate natural language explanation
        explanation = f"""
DIAGNOSTIC ANALYSIS WITH ANOMALY INVESTIGATION

Subject: {report['subject_id']}
Prediction: {report['prediction_result']} (Confidence: {report['confidence']:.1%})

ANOMALY ALERT:
While the model shows reasonable confidence, {len(report['anomaly_status']['anomalous_regions'])} 
brain regions exhibit statistically unusual patterns (|Z-score| > {self.z_score_threshold}).

ANOMALOUS REGIONS:
{', '.join(report['anomaly_status']['anomalous_regions'][:5])}

CLINICAL CONTEXT (from Knowledge Graph):
{knowledge_context['summary']}

DETAILED CONTEXT:
"""
        
        for ctx in knowledge_context['contexts'][:3]:
            region = ctx['region']
            info = ctx['context']
            explanation += f"""
- {region} ({info['full_name']}):
  Function: {info['function']}
  Clinical Significance: {info['clinical_significance']}
"""
            if info.get('related_conditions'):
                explanation += f"  Related Conditions: {', '.join(info['related_conditions'])}\n"
        
        explanation += f"""

TOP CONTRIBUTING FEATURES:
"""
        
        for i, feat in enumerate(report.get('top_features', [])[:5], 1):
            roi_name = _safe_get_attr(feat, 'roi_name', 'Unknown')
            z_score = _safe_get_attr(feat, 'z_score', 0)
            shap_value = _safe_get_attr(feat, 'shap_value', 0)
            anomaly_marker = "⚠️ ANOMALY" if abs(z_score) > self.z_score_threshold else ""
            explanation += f"\n{i}. {roi_name}: Z-score={z_score:+.2f}, SHAP={shap_value:+.4f} {anomaly_marker}"
        
        explanation += f"""

INTERPRETATION:
The presence of anomalous patterns suggests potential:
1. Mixed pathology (e.g., AD with vascular or Lewy body components)
2. Atypical presentation requiring additional clinical correlation
3. Data quality issues that should be reviewed

RECOMMENDATION:
Clinical review recommended to correlate imaging findings with patient symptoms 
and history, particularly regarding {', '.join(report['anomaly_status']['anomalous_regions'][:2])}.
"""
        
        return {
            'subject_id': report['subject_id'],
            'agent_decision': 'ANOMALY_INVESTIGATION',
            'prediction': report['prediction_result'],
            'confidence': report['confidence'],
            'uq_score': report['uq_score'],
            'report': report,
            'knowledge_context': knowledge_context,
            'explanation': explanation.strip(),
            'reasoning_chain': [
                f"1. Obtained diagnostic report: {report['prediction_result']} ({report['confidence']:.1%})",
                f"2. Detected {len(report['anomaly_status']['anomalous_regions'])} anomalous regions",
                f"3. Queried knowledge graph for clinical context",
                f"4. Retrieved context for: {', '.join(report['anomaly_status']['anomalous_regions'][:3])}",
                f"5. Synthesized anomaly-aware diagnostic report"
            ],
            'timestamp': datetime.now().isoformat()
        }
    
    def synthesize_standard_report(self, report: Dict) -> Dict:
        """
        Generate standard diagnostic report
        
        Logic: "Straightforward case with clear prediction and no concerns."
        """
        if self.verbose:
            print(f"\n[SYNTHESIS] Generating standard report...")
        
        # Generate natural language explanation
        explanation = f"""
STANDARD DIAGNOSTIC ANALYSIS

Subject: {report['subject_id']}
Prediction: {report['prediction_result']} (Confidence: {report['confidence']:.1%})

ASSESSMENT:
The model provides a clear prediction with reasonable confidence and low uncertainty 
(UQ Score: {report['uq_score']:.3f}). No statistical anomalies detected.

TOP CONTRIBUTING FEATURES:
"""
        
        for i, feat in enumerate(report.get('top_features', [])[:5], 1):
            roi_name = _safe_get_attr(feat, 'roi_name', 'Unknown')
            z_score = _safe_get_attr(feat, 'z_score', 0)
            shap_value = _safe_get_attr(feat, 'shap_value', 0)
            direction = "↓ Atrophy" if z_score < 0 else "↑ Preserved"
            explanation += f"\n{i}. {roi_name}: Z-score={z_score:+.2f}, SHAP={shap_value:+.4f} {direction}"
        
        explanation += f"""

INTERPRETATION:
The diagnosis is primarily driven by {'atrophy' if report['top_features'][0]['z_score'] < 0 else 'preservation'} 
in {report['top_features'][0]['roi_name']}, which is consistent with typical 
{report['prediction_result']} presentation.

RECOMMENDATION:
Standard clinical follow-up appropriate. The imaging findings align with 
expected patterns for {report['prediction_result']}.
"""
        
        return {
            'subject_id': report['subject_id'],
            'agent_decision': 'STANDARD_REPORT',
            'prediction': report['prediction_result'],
            'confidence': report['confidence'],
            'uq_score': report['uq_score'],
            'report': report,
            'explanation': explanation.strip(),
            'reasoning_chain': [
                f"1. Obtained diagnostic report: {report['prediction_result']} ({report['confidence']:.1%})",
                f"2. Verified low uncertainty: UQ={report['uq_score']:.3f} < {self.uq_threshold}",
                f"3. Confirmed no anomalies detected",
                f"4. Generated standard diagnostic report",
                f"5. Identified primary driver: {report['top_features'][0]['roi_name']}"
            ],
            'timestamp': datetime.now().isoformat()
        }
    
    def print_report(self, result: AgentResult):
        """
        Pretty print the agent's analysis result
        
        Args:
            result: AgentResult from run_analysis()
        """
        print("\n" + "="*80)
        print("CDDA AGENT - FINAL REPORT (A2A)")
        print("="*80)
        print(f"\nSubject: {result.subject_id}")
        print(f"Agent Decision: {result.agent_decision}")
        print(f"Prediction: {result.prediction} ({result.confidence:.1%})")
        print(f"UQ Score: {result.uq_score:.3f}")
        print(f"Timestamp: {result.timestamp}")
        print("\n" + "-"*80)
        print("CLINICAL REPORT:")
        print("-"*80)
        print(result.clinical_report)
        print("\n" + "-"*80)
        print("\nREASONING CHAIN:")
        print("-"*80)
        for step in result.reasoning_chain:
            print(step)
        print("\n" + "="*80 + "\n")


def demo_a2a_standard_case():
    """Demo: A2A Standard case (low uncertainty, no anomalies)"""
    print("\n" + "="*80)
    print("DEMO 1: A2A Standard Case")
    print("="*80)
    
    # Initialize with rule-based fallback (no LLM required)
    agent = CDDAAgent(
        use_llm=False,  # Use rule-based orchestration
        verbose=True
    )
    
    result = agent.run_analysis('sub-0015')
    agent.print_report(result)
    
    # Save reasoning log
    agent.save_reasoning_log(result, "output/demo_standard_reasoning.json")


def demo_a2a_high_uncertainty():
    """Demo: A2A High uncertainty case (triggers counterfactual)"""
    print("\n" + "="*80)
    print("DEMO 2: A2A High Uncertainty Case")
    print("="*80)
    
    # Lower threshold to trigger simulation
    agent = CDDAAgent(
        uq_threshold=0.7,
        use_llm=False,  # Use rule-based orchestration
        verbose=True
    )
    
    result = agent.run_analysis('sub-0005')
    agent.print_report(result)
    
    # Save reasoning log
    agent.save_reasoning_log(result, "output/demo_high_uq_reasoning.json")


def demo_a2a_anomaly_case():
    """Demo: A2A Anomaly case (triggers knowledge lookup)"""
    print("\n" + "="*80)
    print("DEMO 3: A2A Anomaly Case")
    print("="*80)
    
    # Lower z-score threshold to trigger anomaly detection
    agent = CDDAAgent(
        z_score_threshold=1.5,
        use_llm=False,  # Use rule-based orchestration
        verbose=True
    )
    
    result = agent.run_analysis('sub-0005')
    agent.print_report(result)
    
    # Save reasoning log
    agent.save_reasoning_log(result, "output/demo_anomaly_reasoning.json")


def demo_a2a_with_llm():
    """Demo: A2A with LLM-based agents (requires Ollama or HuggingFace models)"""
    print("\n" + "="*80)
    print("DEMO 4: A2A with LLM Agents")
    print("="*80)
    
    try:
        agent = CDDAAgent(
            orchestrator_model="phi-4-mini",
            consultant_model="medgemma-27b",
            use_llm=True,
            verbose=True
        )
        
        result = agent.run_analysis('sub-0005')
        agent.print_report(result)
        
        # Save reasoning log
        agent.save_reasoning_log(result, "output/demo_llm_reasoning.json")
        
    except Exception as e:
        print(f"\n[WARNING] LLM demo failed: {e}")
        print("This is expected if Ollama is not running or models are not installed.")
        print("The agent will fall back to rule-based orchestration.")


if __name__ == "__main__":
    # Run demos
    print("\n" + "="*80)
    print("CDDA AGENT - A2A PATTERN DEMOS")
    print("="*80)
    
    # Demo 1: Standard case
    demo_a2a_standard_case()
    print("\n\n")
    
    # Demo 2: High uncertainty
    demo_a2a_high_uncertainty()
    print("\n\n")
    
    # Demo 3: Anomaly case
    demo_a2a_anomaly_case()
    print("\n\n")
    
    # Demo 4: With LLM (optional)
    print("\n[INFO] Skipping LLM demo by default.")
    print("[INFO] To test LLM mode, run: demo_a2a_with_llm()")
    print("[INFO] Ensure Ollama is running and models are installed first.")
