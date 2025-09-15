from pydantic import BaseModel, Field

from agents.client.agent_client import create_llm_agent

INSTRUCTION = """
You are a clinical report generator responsible for summarizing multi-stage analysis results from an fMRI Alzheimer pipeline.

Use the following context to write the final report:

---

**Step 1 Results:**
{map_act_brain_result}

**Step 2 Results:**
{image_explain_result}

**Step 3 Results:**
{graph_rag_result}

---

Your report must include:

1. **Model Classification Output**  
   - Predicted label and explanation for selected activation layer.
   - Activation heatmap image path.

2. **Activation Pattern Interpretation**  
   - Anatomical regions, asymmetry, functional roles, and clinical significance.

3. **Graph Reasoning Summary**  
   - Network-level and disease associations per region from the knowledge graph.

Be precise, medically relevant, and clearly structured.

**Your Final Output Mandate:**
Your **sole final task** is to generate a JSON object that strictly adheres to the `FinalReport` schema, which has two keys: `final_report_markdown` and `visualization_path`.

**Specific Field Instructions:**
1.  **`classification_result`**: First, identify the model's classification prediction (e.g., "AD" or "CN") from the context, likely within {map_act_brain_result}, and place it in this field.
2.  **`visualization_path`**: Find the primary activation map PNG file path from the context (e.g., `figures/.../activation_map_...png`) and place it in this field.
2.  **`final_report_markdown`**: Next, you MUST synthesize ALL the information from ALL context fields into a single, comprehensive, and coherent narrative report formatted in Markdown. This report should seamlessly integrate the model's findings, the visual interpretation, and the knowledge graph insights, following the professional structure of a clinical analysis.
4.  **`final_report_chinese`**: Translate the entire `final_report_markdown` content into **TRADITIONAL Chinese**, ensuring all medical terminology and nuances are accurately preserved.

**CRITICAL RULE:** If any information from the context is missing or inadequate, generate the report with the information you have. Do not mention the missing parts in the report. Your output MUST be a single, valid JSON object and nothing else. Do not write any text or explanation before or after the JSON object.
"""
class FinalReport(BaseModel):
   """
   The simplified, final output of the entire pipeline,
   designed for direct use in a UI.
   """
   classification_result: str = Field(
      ...,
      description="The classification result of the model (e.g., 'AD' or 'CN')."
   )
   visualization_path: str = Field(
      ...,
      description="The file path to the primary activation map visualization PNG image for display."
   )
   final_report_markdown: str = Field(
      ..., 
      description="The complete, final clinical report formatted in Markdown, integrating all findings from all previous agents."
   )
   final_report_chinese: str = Field(
      ..., 
      description="The complete, final clinical report formatted in traditional Chinese, integrating all findings from all previous agents."
   )


report_generator_agent = create_llm_agent(
   name="ReportGeneratorAgent",
   # model=LiteLlm(model="ollama_chat/gpt-oss:20b"),
   # model="gemini-2.5-flash-lite",
   description="Integrates outputs from previous steps and writes the final clinical report.",
   instruction=INSTRUCTION,
   output_schema=FinalReport,
   output_key="final_report",
)
