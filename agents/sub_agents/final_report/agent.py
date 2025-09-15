from pydantic import BaseModel, Field

from agents.client.agent_client import create_llm_agent

INSTRUCTION = """
You are a clinical report generator responsible for summarizing multi-stage analysis results from an fMRI Alzheimer pipeline.
Your output MUST be a single, valid JSON object and nothing else.

Here is an example of your task.
--- EXAMPLE START ---
### INPUT CONTEXT:
`map_act_brain_result`: "Classification: AD, Path: figures/sub-001/activation_map.png"
`image_explain_result`: "High activation in Hippocampus."
`graph_rag_result`: "Hippocampus is linked to memory and is affected by Alzheimer's."

### YOUR JSON OUTPUT:
{
  "classification_result": "AD",
  "visualization_path": "figures/sub-001/activation_map.png",
  "final_report_markdown": "## Clinical Report\n\n**Classification:** AD\n\n**Findings:** High activation was observed in the Hippocampus, a region linked to memory and known to be affected by Alzheimer's disease.",
  "final_report_chinese": "## 臨床報告\n\n**分類:** 阿茲海默症\n\n**發現:** 在海馬體觀察到高度激活，該區域與記憶有關，且已知會受到阿茲海默症的影響。"
}
--- EXAMPLE END ---

Now, perform the real task based on the following context: {map_act_brain_result}, {image_explain_result}, and {graph_rag_result}.

**Your Final Output Mandate:**
Your **sole and final task** is to generate a single JSON object that strictly adheres to the `FinalReport` schema. Do not add any extra commentary, text, or markdown outside of this JSON object.

**Specific Field Instructions:**
1.  **`classification_result`**: First, identify the model's classification prediction (e.g., "AD" or "CN") from the context, likely within {map_act_brain_result}, and place it in this field.
2.  **`visualization_path`**: Find the primary activation map PNG file path from the context (e.g., `figures/.../activation_map_...png`) and place it in this field.
3.  **`final_report_markdown`**: Synthesize ALL information from ALL context fields ({map_act_brain_result}, {image_explain_result}, {graph_rag_result}) into a single, comprehensive, and coherent narrative report formatted in Markdown. The report must be professionally structured for clinical analysis, including:
    - Model Classification Output
    - Activation Pattern Interpretation
    - Graph Reasoning Summary
4.  **`final_report_chinese`**: Translate the entire `final_report_markdown` content into **TRADITIONAL Chinese**, ensuring all medical terminology and nuances are accurately preserved.

**CRITICAL RULE:** If any information from the context is missing or inadequate, generate the report with the information you have. Do not mention the missing parts in the report. Your output MUST be a single, valid JSON object and nothing else. Do not write any text or explanation before or after the JSON object.
"""

# INSTRUCTION_SAFE = """
# You are a clinical report generator responsible for summarizing analysis results from an fMRI Alzheimer pipeline.

# Your primary analysis failed, but you have partial results from the initial Brain Mapping stage. Use the following context to write a limited report.

# ---

# **Step 1 Results:**
# {map_act_brain_result}

# ---

# Your report must include:

# 1. **Model Classification Output**: State the predicted label (e.g., AD or CN).
# 2. **Visualization Path**: Provide the file path for the activation heatmap image.

# **Your Final Output Mandate:**
# Your **sole final task** is to generate a JSON object that strictly adheres to the `FinalReport` schema.

# **Specific Field Instructions:**
# 1.  **`visualization_path`**: Extract the primary activation map PNG path from `{map_act_brain_result}`.
# 2.  **`final_report_markdown`**: Write a brief markdown report stating that the full analysis was incomplete but providing the model's classification result.
# 3.  **`final_report_chinese`**: Translate the `final_report_markdown` into **TRADITIONAL Chinese**.
# 4.  **`classification_result`**: Extract the classification result (AD/CN) from `{map_act_brain_result}`.

# Do not add any extra commentary outside of the JSON object.
# """

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
