from typing import List
from app.graph.state import AgentState, BrainRegionInfo
from app.services.llm_providers import llm_response


def format_regions_for_prompt(regions: List[BrainRegionInfo]) -> str:
    if not regions:
        return "No significant brain region activations were identified."

    text_parts = ["Key Activated Regions and Their Known Associations:\n"]
    for region in regions[:15]: 
        name = region.get("region_name", "N/A")
        score = region.get("activation_score", 0)
        
        # Safely handle None values
        networks_list = region.get("associated_networks") or []
        functions_list = region.get("known_functions") or []
        
        # Ensure we have lists before joining
        if not isinstance(networks_list, list):
            networks_list = []
        if not isinstance(functions_list, list):
            functions_list = []
            
        networks = ", ".join(networks_list) if networks_list else "N/A"
        functions = ", ".join(functions_list) if functions_list else "N/A"
        
        text_parts.append(
            f"- **{name}** (Activation Score: {score:.3f})\n"
            f"  - Associated Networks: {networks}\n"
            f"  - Known Functions: {functions}\n"
        )
    return "\n".join(text_parts)

def generate_structural_report(state: AgentState) -> dict:
    """
    Generate report for structural MRI analysis
    """
    print("\n--- Node: Structural MRI Report Generator ---")
    
    subject_id = state.get("subject_id")
    classification = state.get("classification_result")
    confidence = state.get("prediction_confidence", 0)
    activated_regions = state.get("activated_regions", [])
    
    # Format top regions
    top_regions_text = format_regions_for_prompt(activated_regions[:10])
    
    # Build structured prompt for structural MRI
    synthesis_prompt = f"""
    You are a professional neuroradiologist AI. Generate a structured clinical report in JSON format for structural MRI analysis.
    
    **Subject ID**: {subject_id}
    **Classification**: {classification}
    **Model Confidence**: {confidence:.1%}
    **Model Type**: Random Forest (32 AAL ROIs)
    
    **Key Brain Regions**:
    {top_regions_text}
    
    **CRITICAL**: You MUST return ONLY a valid JSON object. Do NOT include any text before or after the JSON. Do NOT use markdown code blocks. Return the raw JSON directly.
    
    **Required JSON structure**:
    
    {{
        "risk_assessment": {{
            "level": "High Risk" or "Low Risk",
            "confidence": {confidence},
            "primary_finding": "1-2 sentence summary of the main finding"
        }},
        "key_findings": {{
            "structural_changes": [
                {{
                    "finding": "Brief description of structural change",
                    "severity": "Mild" or "Moderate" or "Severe",
                    "significance": "High" or "Medium" or "Low"
                }}
            ],
            "volumetric_analysis": [
                {{
                    "region": "Brain region name",
                    "change": "Description of change",
                    "percentage": "Estimated percentage (e.g., -12%)"
                }}
            ]
        }},
        "clinical_interpretation": {{
            "summary": "2-3 sentence clinical summary",
            "ad_indicators": ["indicator1", "indicator2", "indicator3"],
            "protective_factors": ["factor1", "factor2"]
        }},
        "recommendations": {{
            "immediate_actions": ["action1", "action2"],
            "monitoring": ["item1", "item2"],
            "additional_tests": ["test1", "test2"]
        }},
        "limitations": ["limitation1", "limitation2"]
    }}
    
    Guidelines:
    - If classification is "AD" or 1, set risk_level to "High Risk"
    - If classification is "NC" or 0, set risk_level to "Low Risk"
    - Base findings on the activated brain regions provided
    - Be specific and clinically relevant
    - Keep language professional and concise
    - Return ONLY the JSON object, no additional text
    """
    
    print("  - Generating structured MRI report...")
    try:
        import json
        
        # Use bedrock (default) for report generation
        report_en_raw = llm_response(prompt=synthesis_prompt, llm_provider="aws_bedrock")
        print("  - English report generated.")
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            report_en_raw = report_en_raw.strip()
            if report_en_raw.startswith("```json"):
                report_en_raw = report_en_raw.split("```json")[1].split("```")[0].strip()
            elif report_en_raw.startswith("```"):
                report_en_raw = report_en_raw.split("```")[1].split("```")[0].strip()
            
            structured_report_en = json.loads(report_en_raw)
            print("  - Successfully parsed structured report.")
        except json.JSONDecodeError as e:
            print(f"  - Warning: Failed to parse JSON: {e}")
            print(f"  - Raw response: {report_en_raw[:200]}...")
            # Fallback: Create a basic structured report from the text
            structured_report_en = {
                "risk_assessment": {
                    "level": "High Risk" if classification in ["AD", "1", 1] else "Low Risk",
                    "confidence": confidence,
                    "primary_finding": f"Based on structural MRI analysis, the classification is {classification} with {confidence:.1%} confidence. Detailed analysis available in raw report."
                },
                "key_findings": {
                    "structural_changes": [
                        {
                            "finding": f"Analysis of {len(activated_regions)} brain regions completed",
                            "severity": "Moderate",
                            "significance": "Medium"
                        }
                    ],
                    "volumetric_analysis": []
                },
                "clinical_interpretation": {
                    "summary": report_en_raw[:300] if len(report_en_raw) > 300 else report_en_raw,
                    "ad_indicators": ["Hippocampal changes", "Temporal lobe alterations"] if classification in ["AD", "1", 1] else [],
                    "protective_factors": ["Preserved frontal function"] if classification in ["NC", "0", 0] else []
                },
                "recommendations": {
                    "immediate_actions": ["Clinical follow-up recommended"],
                    "monitoring": ["Regular MRI monitoring", "Cognitive assessment"],
                    "additional_tests": ["Consider additional biomarker analysis"]
                },
                "limitations": ["Structured report generation encountered formatting issues", "Please review detailed analysis"]
            }
        
        # Generate Chinese translation
        translation_prompt = f"""Translate this JSON structure to Traditional Chinese. Keep the JSON structure intact, only translate the text values.

{json.dumps(structured_report_en, indent=2, ensure_ascii=False)}

Return ONLY the translated JSON, no additional text."""
        
        report_zh_raw = llm_response(prompt=translation_prompt, llm_provider="aws_bedrock")
        print("  - Chinese translation generated.")
        
        # Parse Chinese JSON
        try:
            report_zh_raw = report_zh_raw.strip()
            if report_zh_raw.startswith("```json"):
                report_zh_raw = report_zh_raw.split("```json")[1].split("```")[0].strip()
            elif report_zh_raw.startswith("```"):
                report_zh_raw = report_zh_raw.split("```")[1].split("```")[0].strip()
            
            structured_report_zh = json.loads(report_zh_raw)
            print("  - Successfully parsed Chinese report.")
        except json.JSONDecodeError as e:
            print(f"  - Warning: Failed to parse Chinese JSON: {e}")
            # Use English version as fallback
            structured_report_zh = structured_report_en
        
        trace = "Node: Structured MRI report generation complete."
        return {
            "structured_report": {
                "en": structured_report_en,
                "zh": structured_report_zh
            },
            "trace_log": state.get("trace_log", []) + [trace]
        }
    except Exception as e:
        error_message = f"Node (Structural Report Generator) Error: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}


def generate_final_report(state: AgentState) -> dict:
    """
    Node: Synthesizes all information from the state into a final,
    structured report using a real LLM call.
    
    Routes to appropriate report generator based on analysis mode.
    """
    print("\n--- Node: Final Report Generator ---")
    
    # Check analysis mode
    analysis_mode = state.get("analysis_mode", "functional")
    
    if analysis_mode == "structural":
        return generate_structural_report(state)
    
    # Original functional MRI report generation
    # 1. 收集所有資訊 (保持不變)
    classification = state.get("classification_result")
    enriched_regions = state.get("activated_regions")
    image_explanation_obj = state.get("image_explanation", {})
    image_explanation_text = image_explanation_obj.get("text", "No visual analysis available.")
    subject_id = state.get("subject_id")
    
    # 2. 建立 Prompt (保持不變)
    formatted_regions_text = format_regions_for_prompt(enriched_regions)
    synthesis_prompt = f"""
    You are a professional neuroradiologist AI. Your task is to generate a comprehensive clinical fMRI report for subject: {subject_id}.
    Synthesize all the following information into a single, fluent, and well-structured report.

    **--- Primary Finding ---**
    The initial deep learning model classification for this subject is: **{classification}**

    **--- Visual Analysis of Activation Maps ---**
    An expert vision model provided the following summary of the fMRI activation maps:
    "{image_explanation_text}"

    **--- Detailed Brain Region Analysis (Data from KG) ---**
    {formatted_regions_text}

    **--- YOUR TASK ---**
    Based on ALL the information above, write the final clinical summary report. The report must be structured with the following sections EXACTLY:
    - **Primary Assessment Finding**
    - **Interpretation of Brain Activity Patterns**
    - **Correlation with Established Neurological Knowledge**
    - **Conclusion**
    """

    print("  - Sending final synthesis prompt to LLM...")
    try:
        # --- REPLACE MOCK DATA WITH REAL LLM CALLS ---
        
        # Call for English report
        final_report_en = llm_response(
            prompt=synthesis_prompt,
            llm_provider="gemini" # or "gpt-oss-20b" depending on your setup
        )
        print("  - English report generated.")

        # Call for Chinese translation
        translation_prompt = f"Please translate the following clinical report into fluent, professional Traditional Chinese and reply with content only:\n\n---\n\n{final_report_en}"
        final_report_zh = llm_response(
            prompt=translation_prompt,
            llm_provider="gemini"
        )
        print("  - Chinese translation complete.")
        
        trace = "Node: Final report synthesis complete."
        return {
            "generated_reports": {"en": final_report_en, "zh": final_report_zh},
            "trace_log": state.get("trace_log", []) + [trace]
        }
    except Exception as e:
        error_message = f"Node (Report Generator) Error: {e}"
        print(f"\n[ERROR] {error_message}")
        return {"error_log": state.get("error_log", []) + [error_message]}