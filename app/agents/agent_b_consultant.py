"""
Agent B - Clinical Consultant (Medical Specialist)

This module implements Agent B, the clinical consultant in the dual-LLM A2A system.
Agent B is responsible for:
1. Receiving ContextObject from Agent A via handoff
2. Synthesizing clinical narratives from provided context
3. Generating comprehensive diagnostic reports
4. Interpreting counterfactual results and anomalies

Agent B uses Llama3.1-Aloe-Beta-8B, a specialized medical AI model for clinical reasoning.
IMPORTANT: Agent B has NO direct access to MCP server or tools.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 7.2, 7.3, 7.4
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

from app.core.models import ContextObject, DiagnosticReport
from app.core.prompt_loader import PromptLoader
from app.services.llm_providers import ollama, huggingface
from app.services.llm_providers.error_handling import (
    log_llm_error,
    LLMRetryExhausted,
    LLMParsingError
)


# ============================================================================
# Agent B Configuration
# ============================================================================

@dataclass
class AgentBConfig:
    """Configuration for Agent B"""
    model: str = "llama3.1-aloe-beta-8b"  # Llama3.1-Aloe-Beta-8B for medical reasoning
    model_path: Optional[str] = r"D:\hf_models\Llama3.1-Aloe-Beta-8B"  # Path for HuggingFace models
    provider: str = "huggingface"  # "ollama" or "huggingface"
    temperature: float = 0.3  # Higher than Agent A for more creative synthesis
    use_llm: bool = True  # If False, use template-based generation
    load_in_8bit: bool = True  # Use 8-bit quantization to save memory
    verbose: bool = True
    prompt_path: str = "config/prompts/agent_b_consultant.txt"


# ============================================================================
# Agent B - Clinical Consultant
# ============================================================================

class AgentB:
    """
    Agent B - Clinical Consultant (Medical Specialist)
    
    Responsibilities:
    - Receive ContextObject from Agent A
    - Synthesize clinical narratives
    - Generate comprehensive diagnostic reports
    - Interpret counterfactual results
    - Flag anomalies and mixed pathology
    
    IMPORTANT: Agent B has NO direct access to MCP server or tools.
    All context must come from the ContextObject provided by Agent A.
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """
    
    def __init__(self, config: Optional[AgentBConfig] = None):
        """
        Initialize Agent B
        
        Args:
            config: Agent B configuration
        """
        self.config = config or AgentBConfig()
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        
        # Initialize reasoning chain for logging
        self.reasoning_chain: List[str] = []
        
        if self.config.verbose:
            print("\n" + "="*80)
            print("AGENT B - CLINICAL CONSULTANT (Llama3.1-Aloe-Beta-8B)")
            print("="*80)
            print(f"Model: {self.config.model}")
            if self.config.model_path:
                print(f"Path: {self.config.model_path}")
            print(f"Temperature: {self.config.temperature}")
            print(f"LLM Mode: {'Enabled' if self.config.use_llm else 'Template-based fallback'}")
            print("="*80)
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file using PromptLoader"""
        try:
            loader = PromptLoader()
            return loader.load_agent_b_prompt()
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
                return self._get_default_system_prompt()

    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for Agent B (Optimized for Aloe-Beta-8B)"""
        return """You are Agent B, the Clinical Consultant specializing in neuroimaging and dementia diagnosis.
    Your role is to synthesize clinical narratives from the ContextObject provided by Agent A.

    IMPORTANT: You have NO access to tools or resources. You work ONLY with the context provided to you.

    INPUT: ContextObject containing:
    - diagnostic_report: ML prediction, SHAP values, Z-scores, UQ score, anomalies
    - tool_results: Counterfactual simulation results OR knowledge graph context
    - decision_rationale: Why Agent A took certain actions

    YOUR TASK:
    Synthesize all evidence into a professional, evidence-based clinical report.

    CRITICAL RULES (MUST FOLLOW):
    1. **Confidence Calibration**: 
    - If the prediction confidence is **< 60%**, you MUST describe it as "**Low Confidence**" (低信心度) or "**Borderline Result**" (邊緣性結果). 
    - Do NOT say the model is "confident" in these cases. acknowledge the uncertainty.

    2. **Discrepancy Analysis (Rule-out Logic)**: 
    - If the model predicts 'AD' or 'MCI' but key regions (like Hippocampus) have normal Z-scores (|Z| < 1.5), you MUST flag this as a **Discrepancy**.
    - Explicitly state: "While the model predicts X, the preservation of volume in [Region] suggests atypical presentation or potential differential diagnosis."

    3. **Anomaly Interpretation**:
    - If `anomaly_status` is Detected, treat these regions as potential "Mixed Pathology" or "Non-AD causes" and suggest further investigation.

    REPORT STRUCTURE:
    - **Diagnostic Summary**: Clear statement of Prediction, Confidence (calibrated), and UQ Score.
    - **Key Findings**: List top contributing regions. For each, combine SHAP (AI weight) with Z-score (Biological atrophy) to explain *why* it matters.
    - **Discrepancy & Anomaly Analysis**: (Crucial) Highlight any conflict between AI prediction and biological norms.
    - **Clinical Interpretation**: Synthesize the whole picture. Is this a typical or atypical case?
    - **Recommendations**: Evidence-based next steps (e.g., "Due to low confidence, recommend longitudinal follow-up").
    """

    def synthesize(self, context_object: ContextObject) -> Dict[str, Any]:
        """
        Main synthesis method - generate clinical report from ContextObject
        
        This is the entry point for Agent B. It receives the ContextObject
        from Agent A and synthesizes a comprehensive clinical report.
        
        Args:
            context_object: Complete context from Agent A
        
        Returns:
            Dictionary containing clinical_report and reasoning_chain
        
        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
        """
        if self.config.verbose:
            print(f"\n[AGENT B] Synthesizing report for {context_object.subject_id}")
        
        # Reset reasoning chain
        self.reasoning_chain = []
        
        # Log start
        self._log_reasoning(f"Received ContextObject for {context_object.subject_id}")
        self._log_reasoning(f"Prediction: {context_object.diagnostic_report.prediction_result}")
        self._log_reasoning(f"Confidence: {context_object.diagnostic_report.confidence:.1%}")
        self._log_reasoning(f"UQ Score: {context_object.diagnostic_report.uq_score:.3f}")
        
        # Choose synthesis strategy
        if self.config.use_llm:
            clinical_report = self._synthesize_with_llm(context_object)
        else:
            clinical_report = self._synthesize_with_template(context_object)
        
        # Return result
        return {
            'clinical_report': clinical_report,
            'reasoning_chain': self.reasoning_chain.copy()
        }

    
    # ========================================================================
    # LLM Integration (Subtask 4.2)
    # ========================================================================
    
    def _synthesize_with_llm(self, context_object: ContextObject) -> str:
        """
        Synthesize clinical report using LLM
        
        Includes automatic fallback to template-based synthesis if LLM fails.
        
        Requirements: 1.3, 9.2, 10.3
        """
        if self.config.verbose:
            print(f"[AGENT B] Using LLM-based synthesis ({self.config.model})")
        
        try:
            # Format ContextObject for LLM consumption
            formatted_context = self._format_context_for_llm(context_object)
            
            # Call LLM (with retry and error handling)
            clinical_report = self._call_llm(formatted_context)
            
            self._log_reasoning("LLM synthesis completed successfully")
            
            return clinical_report
            
        except (LLMRetryExhausted, Exception) as e:
            # Log the error
            if self.config.verbose:
                print(f"[AGENT B] LLM synthesis failed: {type(e).__name__}: {e}")
                print("[AGENT B] Falling back to template-based synthesis")
            
            # Log error with context
            log_llm_error(
                e,
                {
                    'agent': 'Agent B',
                    'subject_id': context_object.subject_id,
                    'fallback': 'template-based synthesis'
                }
            )
            
            self._log_reasoning(f"LLM synthesis failed: {type(e).__name__}. Using template-based fallback.")
            
            # Fallback to template-based synthesis (Requirement 10.3)
            return self._synthesize_with_template(context_object)
    
    def _format_context_for_llm(self, context_object: ContextObject) -> str:
        """
        Format ContextObject for LLM consumption
        
        Requirements: 5.1
        """
        # Extract key information
        report = context_object.diagnostic_report
        signals = context_object.signals
        tool_results = context_object.tool_results or {}
        
        # Build structured context
        context_dict = {
            'subject_id': context_object.subject_id,
            'prediction': report.prediction_result,
            'confidence': report.confidence,
            'uq_score': report.uq_score,
            'has_anomaly': signals.get('has_anomaly', False),
            'anomalous_regions': signals.get('anomalous_regions', []),
            'top_features': [
                {
                    'roi_name': f.roi_name,
                    'z_score': f.z_score,
                    'shap_value': f.shap_value,
                    'rank': f.rank
                }
                for f in report.top_features[:10]
            ],
            'decision_rationale': context_object.decision_rationale
        }
        
        # Add counterfactual results if present
        if 'counterfactual' in tool_results:
            cf = tool_results['counterfactual']
            context_dict['counterfactual'] = {
                'original_prediction': cf.get('original_prediction'),
                'original_confidence': cf.get('original_confidence'),
                'new_prediction': cf.get('new_prediction'),
                'new_confidence': cf.get('new_confidence'),
                'confidence_delta': cf.get('confidence_delta'),
                'masked_features': [f.get('roi_name') for f in cf.get('masked_features', [])]
            }
        
        # Add knowledge context if present
        if 'knowledge_context' in tool_results:
            kc = tool_results['knowledge_context']
            context_dict['knowledge_context'] = {
                'query_regions': kc.get('query_regions', []),
                'summary': kc.get('summary', ''),
                'contexts': [
                    {
                        'region': ctx.get('region'),
                        'context': ctx.get('context', {})
                    }
                    for ctx in kc.get('contexts', [])
                ]
            }
        
        # Format as JSON string
        return json.dumps(context_dict, indent=2)

    
    def _call_llm(self, formatted_context: str) -> str:
        """
        Call LLM to generate clinical report
        
        Requirements: 1.3, 9.2
        """
        if self.config.verbose:
            print(f"[AGENT B] Calling LLM: {self.config.model}")
            print(f"[AGENT B] Provider: {self.config.provider}")
        
        # Create user prompt
        user_prompt = f"""
Based on the ContextObject below, synthesize a comprehensive clinical report in English.

CONTEXT OBJECT:
{formatted_context}

Generate a clinical report in English following the structure in the system instructions.
Focus on integrating all evidence and providing clear clinical interpretation.

IMPORTANT: Place the actual clinical report content after the <REPORT> marker.
Everything before <REPORT> will be filtered out in post-processing.

<REPORT>

Report structure should include:
1. Diagnostic Summary
2. Key Findings (Brain Region Analysis)
3. Anomaly Analysis (if applicable)
4. Counterfactual Analysis (if applicable)
5. Clinical Interpretation
6. Recommendations

Use simple but professional clinical language.
"""
        
        # Call LLM based on provider
        try:
            if self.config.provider == "huggingface":
                if not self.config.model_path:
                    raise ValueError("model_path required for HuggingFace provider")
                
                # Check if model exists
                model_info = huggingface.get_model_info(self.config.model_path)
                if not model_info['exists']:
                    if self.config.verbose:
                        print(f"[WARNING] Model not found at: {self.config.model_path}")
                        print(f"[INFO] Please ensure the model is downloaded")
                    raise FileNotFoundError(f"Model not found at: {self.config.model_path}")
                
                if self.config.verbose:
                    print(f"[AGENT B] Using HuggingFace model from: {self.config.model_path}")
                    print(f"[AGENT B] 8-bit quantization: {self.config.load_in_8bit}")
                
                response_text = huggingface.handle_text(
                    prompt=user_prompt,
                    model_path=self.config.model_path,
                    system_instruction=self.system_prompt,
                    temperature=self.config.temperature,
                    max_new_tokens=2048,  # Longer for clinical reports
                    load_in_8bit=self.config.load_in_8bit
                )
            else:  # ollama
                # Check if model is available
                if not ollama.check_availability():
                    raise LLMConnectionError("Ollama server is not running")
                
                available_models = ollama.list_models()
                if self.config.model not in available_models:
                    if self.config.verbose:
                        print(f"[WARNING] Model '{self.config.model}' not found in Ollama")
                        print(f"[INFO] Available models: {', '.join(available_models) if available_models else 'None'}")
                        print(f"[INFO] To install: ollama pull {self.config.model}")
                        print(f"[INFO] Or use alternative: ollama pull llama3.1:8b")
                    raise LLMConnectionError(f"Model '{self.config.model}' not found in Ollama")
                
                response_text = ollama.handle_text(
                    prompt=user_prompt,
                    model=self.config.model,
                    system_instruction=self.system_prompt,
                    temperature=self.config.temperature
                )
            
            if self.config.verbose:
                print(f"[AGENT B] LLM response received ({len(response_text)} chars)")
            
            return response_text
            
        except Exception as e:
            if self.config.verbose:
                print(f"[AGENT B] LLM call failed: {type(e).__name__}: {e}")
            raise e

    
    # ========================================================================
    # Template-Based Synthesis (Fallback)
    # ========================================================================
    
    def _synthesize_with_template(self, context_object: ContextObject) -> str:
        """
        Synthesize clinical report using template (fallback)
        
        Includes error annotations in final report.
        
        Requirements: 10.3, 10.5
        """
        if self.config.verbose:
            print("[AGENT B] Using template-based synthesis")
        
        report = context_object.diagnostic_report
        signals = context_object.signals
        tool_results = context_object.tool_results or {}
        context_errors = context_object.errors if hasattr(context_object, 'errors') else []
        
        # Build report sections
        sections = []
        
        # Summary
        sections.append(self._generate_summary_section(report, signals))
        
        # Key Findings
        sections.append(self._generate_key_findings_section(report))
        
        # Anomaly-aware synthesis (Subtask 4.3)
        if signals.get('has_anomaly', False):
            sections.append(self._generate_anomaly_section(report, signals, tool_results))
        
        # Counterfactual explanation (Subtask 4.5)
        if 'counterfactual' in tool_results:
            sections.append(self._generate_counterfactual_section(tool_results['counterfactual']))
        
        # Clinical Context
        if 'knowledge_context' in tool_results:
            sections.append(self._generate_knowledge_section(tool_results['knowledge_context']))
        
        # Interpretation
        sections.append(self._generate_interpretation_section(report, signals, tool_results))
        
        # Recommendations (with error annotations - Requirement 10.5)
        sections.append(self._generate_recommendations_section(report, signals, tool_results, context_errors))
        
        # Combine sections
        clinical_report = "\n\n".join(sections)
        
        self._log_reasoning("Template-based synthesis completed")
        
        if context_errors:
            self._log_reasoning(f"Included {len(context_errors)} error annotation(s) in final report")
        
        return clinical_report

    
    def _generate_summary_section(self, report: DiagnosticReport, signals: Dict) -> str:
        """Generate summary section"""
        return f"""DIAGNOSTIC SUMMARY
Subject: {report.subject_id}
Prediction: {report.prediction_result}
Confidence: {report.confidence:.1%}
Uncertainty Score: {report.uq_score:.3f}
Anomaly Status: {'Detected' if signals.get('has_anomaly') else 'None'}"""
    
    def _generate_key_findings_section(self, report: DiagnosticReport) -> str:
        """Generate key findings section"""
        lines = ["KEY FINDINGS"]
        lines.append("Top Contributing Brain Regions:")
        
        for i, feature in enumerate(report.top_features[:5], 1):
            z_score_desc = "elevated" if feature.z_score > 0 else "reduced"
            lines.append(
                f"{i}. {feature.roi_name}: "
                f"Z-score = {feature.z_score:.2f} ({z_score_desc}), "
                f"SHAP = {feature.shap_value:.3f}"
            )
        
        return "\n".join(lines)

    
    # ========================================================================
    # Anomaly-Aware Synthesis (Subtask 4.3)
    # ========================================================================
    
    def _generate_anomaly_section(
        self, 
        report: DiagnosticReport, 
        signals: Dict,
        tool_results: Dict
    ) -> str:
        """
        Generate anomaly analysis section with mixed pathology detection
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
        """
        lines = ["ANOMALY ANALYSIS"]
        
        anomalous_regions = signals.get('anomalous_regions', [])
        lines.append(f"Detected {len(anomalous_regions)} anomalous regions:")
        
        for region in anomalous_regions[:5]:
            lines.append(f"  - {region}")
        
        # List disease associations for anomalous regions (Requirement 6.3)
        if 'knowledge_context' in tool_results:
            disease_associations = self._list_disease_associations(
                tool_results['knowledge_context']
            )
            
            if disease_associations:
                lines.append("\nDISEASE ASSOCIATIONS:")
                for assoc in disease_associations:
                    lines.append(f"  - {assoc}")
        
        # Check for model-knowledge discrepancies (Requirement 6.1, 6.2)
        if 'knowledge_context' in tool_results:
            discrepancies = self._detect_model_knowledge_discrepancies(
                report, 
                tool_results['knowledge_context']
            )
            
            if discrepancies:
                lines.append("\nPOTENTIAL MIXED PATHOLOGY INDICATORS:")
                for disc in discrepancies:
                    lines.append(f"  - {disc}")
                
                # Log the detection for reasoning chain
                self._log_reasoning(
                    f"Detected {len(discrepancies)} model-knowledge discrepancies "
                    f"suggesting potential mixed pathology"
                )
        
        # Check for SHAP-condition mismatches (Requirement 6.4)
        shap_mismatches = self._detect_shap_condition_mismatches(report, tool_results)
        if shap_mismatches:
            lines.append("\nSHAP-CONDITION MISMATCHES:")
            for mismatch in shap_mismatches:
                lines.append(f"  - {mismatch}")
            
            # Log the detection for reasoning chain
            self._log_reasoning(
                f"Detected {len(shap_mismatches)} SHAP-condition mismatches"
            )
        
        return "\n".join(lines)
    
    def _list_disease_associations(
        self,
        knowledge_context: Dict
    ) -> List[str]:
        """
        List disease associations for anomalous regions
        
        Requirements: 6.3
        """
        associations = []
        
        for ctx in knowledge_context.get('contexts', []):
            region = ctx.get('region', '')
            context_info = ctx.get('context', {})
            related_conditions = context_info.get('related_conditions', [])
            
            if related_conditions:
                # Format: "Region: Condition1, Condition2, ..."
                conditions_str = ', '.join(related_conditions[:3])  # Limit to top 3
                associations.append(f"{region}: {conditions_str}")
        
        return associations
    
    def _detect_model_knowledge_discrepancies(
        self, 
        report: DiagnosticReport,
        knowledge_context: Dict
    ) -> List[str]:
        """
        Detect discrepancies between model prediction and knowledge context
        
        Requirements: 6.1, 6.2
        """
        discrepancies = []
        
        prediction = report.prediction_result
        confidence = report.confidence
        
        # Check if high confidence AD prediction conflicts with knowledge (Requirement 6.1)
        if prediction == "AD" and confidence > 0.8:
            # Check if anomalous regions are associated with non-AD conditions
            for ctx in knowledge_context.get('contexts', []):
                region = ctx.get('region', '')
                context_info = ctx.get('context', {})
                related_conditions = context_info.get('related_conditions', [])
                
                # Flag if related to non-AD conditions
                non_ad_conditions = [c for c in related_conditions if 'AD' not in c and 'Alzheimer' not in c]
                if non_ad_conditions:
                    # Requirement 6.2: Explain the discrepancy
                    discrepancies.append(
                        f"{region} associated with {', '.join(non_ad_conditions[:2])} "
                        f"but model predicts AD with {confidence:.1%} confidence. "
                        f"This suggests potential mixed pathology or atypical presentation."
                    )
        
        # Also check for other prediction types with high confidence
        elif confidence > 0.8:
            for ctx in knowledge_context.get('contexts', []):
                region = ctx.get('region', '')
                context_info = ctx.get('context', {})
                related_conditions = context_info.get('related_conditions', [])
                
                # Check if conditions don't match prediction
                prediction_keywords = {
                    'AD': ['Alzheimer', 'AD'],
                    'MCI': ['MCI', 'Mild Cognitive'],
                    'NC': ['Normal', 'Healthy']
                }
                
                keywords = prediction_keywords.get(prediction, [])
                matching_conditions = [c for c in related_conditions 
                                     if any(kw in c for kw in keywords)]
                
                if related_conditions and not matching_conditions:
                    # Requirement 6.2: Explain the discrepancy
                    discrepancies.append(
                        f"{region} associated with {', '.join(related_conditions[:2])} "
                        f"but model predicts {prediction} with {confidence:.1%} confidence. "
                        f"This discrepancy warrants further investigation."
                    )
        
        return discrepancies
    
    def _detect_shap_condition_mismatches(
        self,
        report: DiagnosticReport,
        tool_results: Dict
    ) -> List[str]:
        """
        Detect mismatches between leading SHAP features and their associated conditions
        
        Requirements: 6.4
        """
        mismatches = []
        
        if 'knowledge_context' not in tool_results:
            return mismatches
        
        knowledge_context = tool_results['knowledge_context']
        prediction = report.prediction_result
        
        # Define prediction keywords for matching
        prediction_keywords = {
            'AD': ['Alzheimer', 'AD'],
            'MCI': ['MCI', 'Mild Cognitive'],
            'NC': ['Normal', 'Healthy', 'Control']
        }
        
        keywords = prediction_keywords.get(prediction, [])
        
        # Check top SHAP features (not just the first one)
        for feature in report.top_features[:3]:  # Check top 3 features
            # Find knowledge context for this region
            for ctx in knowledge_context.get('contexts', []):
                if ctx.get('region') == feature.roi_name:
                    context_info = ctx.get('context', {})
                    related_conditions = context_info.get('related_conditions', [])
                    
                    if not related_conditions:
                        continue
                    
                    # Check if conditions match prediction
                    matching_conditions = [c for c in related_conditions 
                                         if any(kw in c for kw in keywords)]
                    
                    # If no matching conditions, it's a mismatch
                    if not matching_conditions:
                        mismatches.append(
                            f"Feature {feature.roi_name} (SHAP={feature.shap_value:.3f}, rank={feature.rank}) "
                            f"primarily associated with {', '.join(related_conditions[:2])}, "
                            f"which differs from predicted {prediction}. "
                            f"This may indicate mixed pathology."
                        )
                        break  # Only report once per feature
        
        return mismatches

    
    # ========================================================================
    # Counterfactual Explanation (Subtask 4.5)
    # ========================================================================
    
    def _generate_counterfactual_section(self, counterfactual: Dict) -> str:
        """
        Generate counterfactual analysis section with clinical interpretation
        
        This method interprets counterfactual simulation results to identify
        which brain regions are driving the diagnosis. It provides medical
        reasoning about feature impact based on confidence changes.
        
        Requirements: 7.2, 7.3, 7.4
        """
        lines = ["COUNTERFACTUAL ANALYSIS"]
        lines.append("What-if simulation: Testing diagnostic impact of key features")
        lines.append("")
        
        original_pred = counterfactual.get('original_prediction')
        original_conf = counterfactual.get('original_confidence', 0)
        new_pred = counterfactual.get('new_prediction')
        new_conf = counterfactual.get('new_confidence', 0)
        confidence_delta = counterfactual.get('confidence_delta', 0)
        masked_features = counterfactual.get('masked_features', [])
        
        # Show the simulation results
        lines.append(f"Original prediction: {original_pred} ({original_conf:.1%} confidence)")
        lines.append(f"After masking features: {new_pred} ({new_conf:.1%} confidence)")
        lines.append(f"Confidence change: {confidence_delta:+.1%}")
        lines.append(f"\nMasked features: {', '.join([f.get('roi_name', '') for f in masked_features])}")
        
        # Interpret confidence delta with medical reasoning (Requirements 7.2, 7.3, 7.4)
        lines.append("\nCLINICAL INTERPRETATION:")
        
        if abs(confidence_delta) > 0.1:
            # Significant change - key diagnostic drivers (Requirement 7.3)
            lines.append(
                f"The masked features are KEY DIAGNOSTIC DRIVERS. "
                f"Removing them caused a {abs(confidence_delta):.1%} change in confidence, "
                f"indicating they are critical to the {original_pred} diagnosis. "
                f"These regions show pathological changes that strongly support the diagnosis."
            )
            
            # Identify specific drivers with clinical explanations (Requirement 7.2)
            key_drivers = self._identify_key_drivers(masked_features, confidence_delta)
            if key_drivers:
                lines.append("\nDetailed feature impact analysis:")
                for driver in key_drivers:
                    lines.append(f"  • {driver}")
                
                # Add clinical context
                lines.append(
                    f"\nClinical significance: The substantial confidence change ({abs(confidence_delta):.1%}) "
                    f"when these features are neutralized confirms they are primary pathological markers. "
                    f"This validates the model's reliance on these regions for diagnosis."
                )
        
        elif abs(confidence_delta) < 0.05:
            # Minimal change - not primary drivers (Requirement 7.4)
            lines.append(
                f"The masked features are NOT PRIMARY DRIVERS. "
                f"Removing them caused only a {abs(confidence_delta):.1%} change in confidence, "
                f"suggesting other features are more important for the diagnosis. "
                f"While these regions may show some abnormality, they are not the main diagnostic indicators."
            )
            
            # Add clinical context (Requirement 7.2)
            lines.append(
                f"\nClinical significance: The minimal confidence change indicates these features "
                f"have limited diagnostic value in this case. The diagnosis is primarily driven by "
                f"other brain regions not included in this counterfactual simulation."
            )
        
        else:
            # Moderate change (0.05 <= delta <= 0.1)
            lines.append(
                f"The masked features have MODERATE IMPACT on the diagnosis. "
                f"They contribute to the prediction ({abs(confidence_delta):.1%} confidence change) "
                f"but are not the sole drivers. Other features also play important roles."
            )
            
            # Identify drivers with explanations (Requirement 7.2)
            key_drivers = self._identify_key_drivers(masked_features, confidence_delta)
            if key_drivers:
                lines.append("\nFeature impact analysis:")
                for driver in key_drivers:
                    lines.append(f"  • {driver}")
            
            # Add clinical context (Requirement 7.2)
            lines.append(
                f"\nClinical significance: The moderate confidence change suggests these features "
                f"are part of a broader pattern of pathological changes. Consider evaluating "
                f"additional regions to build a complete diagnostic picture."
            )
        
        return "\n".join(lines)
    
    def _identify_key_drivers(self, masked_features: List[Dict], confidence_delta: float) -> List[str]:
        """
        Identify key diagnostic drivers from counterfactual results
        
        This method generates clinical explanations for each masked feature,
        helping clinicians understand which brain regions are driving the diagnosis.
        
        Requirements: 7.2, 7.3
        """
        drivers = []
        
        for feature in masked_features:
            roi_name = feature.get('roi_name', '')
            original_value = feature.get('original_value', 0)
            masked_value = feature.get('masked_value', 0)
            
            # Generate clinical explanation based on confidence delta (Requirement 7.2)
            if abs(confidence_delta) > 0.2:
                impact = "critical"
                clinical_note = "This region is a primary diagnostic driver"
            elif abs(confidence_delta) > 0.1:
                impact = "significant"
                clinical_note = "This region contributes substantially to the diagnosis"
            else:
                impact = "moderate"
                clinical_note = "This region has measurable but limited diagnostic impact"
            
            # Calculate the deviation from normal
            deviation = original_value - masked_value
            deviation_desc = "reduced" if deviation < 0 else "elevated"
            
            # Build clinical explanation (Requirement 7.2)
            explanation = (
                f"{roi_name}: {impact} impact on diagnosis. "
                f"Original value {original_value:.2f} ({deviation_desc} by {abs(deviation):.2f} from population mean). "
                f"{clinical_note}."
            )
            
            drivers.append(explanation)
        
        return drivers

    
    def _generate_knowledge_section(self, knowledge_context: Dict) -> str:
        """Generate clinical knowledge section"""
        lines = ["CLINICAL CONTEXT"]
        
        summary = knowledge_context.get('summary', '')
        if summary:
            lines.append(summary)
        
        contexts = knowledge_context.get('contexts', [])
        if contexts:
            lines.append("\nDetailed Context:")
            for ctx in contexts:
                region = ctx.get('region', '')
                context_info = ctx.get('context', {})
                
                lines.append(f"\n{region}:")
                lines.append(f"  Function: {context_info.get('function', 'Unknown')}")
                lines.append(f"  Clinical Significance: {context_info.get('clinical_significance', 'Unknown')}")
                
                related_conditions = context_info.get('related_conditions', [])
                if related_conditions:
                    lines.append(f"  Related Conditions: {', '.join(related_conditions)}")
        
        return "\n".join(lines)
    
    def _generate_interpretation_section(
        self,
        report: DiagnosticReport,
        signals: Dict,
        tool_results: Dict
    ) -> str:
        """
        Generate interpretation section with anomaly-aware synthesis
        
        Requirements: 5.4, 6.1, 6.2
        """
        lines = ["CLINICAL INTERPRETATION"]
        
        prediction = report.prediction_result
        confidence = report.confidence
        uq_score = report.uq_score
        has_anomaly = signals.get('has_anomaly', False)
        
        # Base interpretation
        if confidence > 0.8:
            conf_desc = "high confidence"
        elif confidence > 0.6:
            conf_desc = "moderate confidence"
        else:
            conf_desc = "low confidence"
        
        lines.append(
            f"The model predicts {prediction} with {conf_desc} ({confidence:.1%}). "
        )
        
        # Uncertainty interpretation
        if uq_score > 0.8:
            lines.append(
                f"However, the high uncertainty score ({uq_score:.3f}) suggests "
                f"the model is less certain about this prediction. "
            )
            if 'counterfactual' in tool_results:
                lines.append(
                    "Counterfactual analysis was performed to identify key diagnostic drivers."
                )
        
        # Anomaly interpretation with mixed pathology detection (Requirements 5.4, 6.1, 6.2)
        if has_anomaly:
            anomalous_count = len(signals.get('anomalous_regions', []))
            lines.append(
                f"Statistical anomalies were detected in {anomalous_count} region(s). "
            )
            
            # Check for mixed pathology and discrepancies
            if 'knowledge_context' in tool_results:
                discrepancies = self._detect_model_knowledge_discrepancies(
                    report,
                    tool_results['knowledge_context']
                )
                
                shap_mismatches = self._detect_shap_condition_mismatches(
                    report,
                    tool_results
                )
                
                if discrepancies or shap_mismatches:
                    # Requirement 5.4: Explicitly address potential mixed pathology or atypical presentation
                    lines.append(
                        "CAUTION: Potential mixed pathology or atypical presentation detected. "
                    )
                    
                    # Requirement 6.2: Explain the discrepancy using medical reasoning
                    if discrepancies:
                        lines.append(
                            f"The anomalous regions show associations with conditions that differ "
                            f"from the predicted {prediction} diagnosis. This discrepancy may indicate:"
                        )
                        lines.append("  - Co-existing pathologies (e.g., AD with vascular changes)")
                        lines.append("  - Atypical disease presentation")
                        lines.append("  - Early-stage disease with mixed features")
                    
                    if shap_mismatches:
                        lines.append(
                            f"Additionally, the leading diagnostic features show associations "
                            f"with conditions beyond {prediction}, further supporting the need "
                            f"for comprehensive clinical evaluation."
                        )
                    
                    # Log this important finding
                    self._log_reasoning(
                        f"Flagged potential mixed pathology: {len(discrepancies)} discrepancies, "
                        f"{len(shap_mismatches)} SHAP mismatches"
                    )
                else:
                    lines.append(
                        "The anomalies may represent normal anatomical variation or "
                        "measurement artifacts. Clinical correlation is recommended."
                    )
        
        return "\n".join(lines)

    
    def _generate_recommendations_section(
        self,
        report: DiagnosticReport,
        signals: Dict,
        tool_results: Dict,
        context_errors: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate recommendations section
        
        Includes error annotations if any fallbacks were used.
        
        Requirements: 6.5, 10.5
        """
        lines = ["RECOMMENDATIONS"]
        
        prediction = report.prediction_result
        confidence = report.confidence
        uq_score = report.uq_score
        has_anomaly = signals.get('has_anomaly', False)
        
        recommendation_num = 1
        
        # Error annotations (Requirement 10.5)
        if context_errors and len(context_errors) > 0:
            lines.append("\nNOTE: The following issues were encountered during analysis:")
            for error in context_errors:
                lines.append(
                    f"  - {error['component']}: {error['type']} - {error['message']}"
                )
            lines.append(
                "\nDespite these issues, the analysis was completed using fallback methods. "
                "Results should be interpreted with appropriate caution."
            )
            lines.append("")
        
        # Standard recommendations
        lines.append(f"{recommendation_num}. Clinical correlation with patient history and symptoms")
        recommendation_num += 1
        lines.append(f"{recommendation_num}. Consider additional neuropsychological testing")
        recommendation_num += 1
        
        # High uncertainty recommendations
        if uq_score > 0.8:
            lines.append(
                f"{recommendation_num}. High uncertainty detected - recommend follow-up imaging "
                f"and longitudinal monitoring"
            )
            recommendation_num += 1
        
        # Anomaly recommendations with mixed pathology detection (Requirement 6.5)
        if has_anomaly:
            # Check for mixed pathology indicators
            has_mixed_pathology = False
            
            if 'knowledge_context' in tool_results:
                discrepancies = self._detect_model_knowledge_discrepancies(
                    report,
                    tool_results['knowledge_context']
                )
                
                shap_mismatches = self._detect_shap_condition_mismatches(
                    report,
                    tool_results
                )
                
                # Multiple pathologies suggested if we have discrepancies or mismatches
                if discrepancies or shap_mismatches:
                    has_mixed_pathology = True
                    
                    # Requirement 6.5: Recommend additional clinical correlation for multiple pathologies
                    lines.append(
                        f"{recommendation_num}. IMPORTANT: Anomalous patterns suggest potential mixed pathology. "
                        f"Recommend comprehensive workup including:"
                    )
                    lines.append(f"   - Additional clinical correlation to differentiate pathologies")
                    lines.append(f"   - Vascular imaging (rule out vascular dementia)")
                    lines.append(f"   - CSF biomarkers (confirm AD pathology)")
                    lines.append(f"   - PET imaging (assess amyloid/tau burden)")
                    lines.append(f"   - Consider other neurodegenerative conditions (Lewy body, FTD)")
                    lines.append(f"   - Longitudinal follow-up to track disease progression")
                    
                    # Log this important recommendation
                    self._log_reasoning(
                        "Generated multiple pathology recommendations due to "
                        f"{len(discrepancies)} discrepancies and {len(shap_mismatches)} SHAP mismatches"
                    )
                    recommendation_num += 1
            
            # Standard anomaly recommendation if no mixed pathology
            if not has_mixed_pathology:
                lines.append(
                    f"{recommendation_num}. Anomalies detected - recommend additional clinical correlation"
                )
                recommendation_num += 1
        
        # Low confidence recommendations
        if confidence < 0.6:
            lines.append(
                f"{recommendation_num}. Low confidence prediction - recommend repeat imaging "
                f"and comprehensive clinical assessment"
            )
            recommendation_num += 1
        
        return "\n".join(lines)
    
    # ========================================================================
    # Reasoning Chain Logging
    # ========================================================================
    
    def _log_reasoning(self, message: str):
        """
        Log reasoning step with timestamp
        
        Requirements: 8.1, 8.2
        """
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [Agent B] {message}"
        
        self.reasoning_chain.append(log_entry)
        
        if self.config.verbose:
            print(f"[REASONING] {message}")



# ============================================================================
# Demo Functions
# ============================================================================

def demo_agent_b_template():
    """Demo: Agent B with template-based synthesis"""
    print("\n" + "="*80)
    print("DEMO: Agent B - Template-Based Synthesis")
    print("="*80)
    
    # Create mock ContextObject
    from app.core.models import Feature, AnomalyStatus
    
    diagnostic_report = DiagnosticReport(
        subject_id='sub-0005',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.75,
        top_features=[
            Feature(
                roi_name='Hippocampus_L',
                feature_name='Hippocampus_L_GM_Vol',
                feature_value=2500.0,
                z_score=-2.8,
                shap_value=0.15,
                rank=1
            ),
            Feature(
                roi_name='Hippocampus_R',
                feature_name='Hippocampus_R_GM_Vol',
                feature_value=2450.0,
                z_score=-2.6,
                shap_value=0.12,
                rank=2
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=False,
            anomalous_regions=[]
        )
    )
    
    context_object = ContextObject(
        subject_id='sub-0005',
        diagnostic_report=diagnostic_report,
        decision_rationale="Standard case: low uncertainty, no anomalies.",
        signals={
            'uq_score': 0.75,
            'has_anomaly': False,
            'prediction': 'AD',
            'confidence': 0.85
        },
        agent_a_reasoning=["Read diagnostic report", "Evaluated signals", "Compiled context"]
    )
    
    # Initialize Agent B (template mode)
    config = AgentBConfig(use_llm=False, verbose=True)
    agent_b = AgentB(config=config)
    
    # Synthesize report
    result = agent_b.synthesize(context_object)
    
    # Print results
    print("\n" + "-"*80)
    print("CLINICAL REPORT:")
    print("-"*80)
    print(result['clinical_report'])
    print("-"*80)


def demo_agent_b_with_llm():
    """Demo: Agent B with LLM-based synthesis"""
    print("\n" + "="*80)
    print("DEMO: Agent B - LLM-Based Synthesis (MedGemma-27B)")
    print("="*80)
    
    # Check if Ollama is available
    if not ollama.check_availability():
        print("[WARNING] Ollama not available. Skipping LLM demo.")
        print("To run this demo:")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Pull model: ollama pull medgemma-27b")
        print("     (Alternative: ollama pull llama3.1:8b)")
        print("  3. Start server: ollama serve")
        return
    
    # Create mock ContextObject (same as template demo)
    from app.core.models import Feature, AnomalyStatus
    
    diagnostic_report = DiagnosticReport(
        subject_id='sub-0005',
        prediction_result='AD',
        confidence=0.85,
        uq_score=0.75,
        top_features=[
            Feature(
                roi_name='Hippocampus_L',
                feature_name='Hippocampus_L_GM_Vol',
                feature_value=2500.0,
                z_score=-2.8,
                shap_value=0.15,
                rank=1
            )
        ],
        anomaly_status=AnomalyStatus(
            has_anomaly=False,
            anomalous_regions=[]
        )
    )
    
    context_object = ContextObject(
        subject_id='sub-0005',
        diagnostic_report=diagnostic_report,
        decision_rationale="Standard case: low uncertainty, no anomalies.",
        signals={
            'uq_score': 0.75,
            'has_anomaly': False,
            'prediction': 'AD',
            'confidence': 0.85
        },
        agent_a_reasoning=["Read diagnostic report", "Evaluated signals", "Compiled context"]
    )
    
    # Initialize Agent B (LLM mode)
    config = AgentBConfig(use_llm=True, verbose=True)
    agent_b = AgentB(config=config)
    
    # Synthesize report
    result = agent_b.synthesize(context_object)
    
    # Print results
    print("\n" + "-"*80)
    print("CLINICAL REPORT:")
    print("-"*80)
    print(result['clinical_report'])
    print("-"*80)


if __name__ == "__main__":
    # Run demos
    demo_agent_b_template()
    print("\n\n")
    demo_agent_b_with_llm()
