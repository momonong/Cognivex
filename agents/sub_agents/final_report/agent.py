from pydantic import BaseModel, Field

from agents.client.agent_client import create_llm_agent

INSTRUCTION = """
You are an expert neuroimaging analyst tasked with generating a final clinical summary report based on a multi-stage fMRI analysis for Alzheimer's disease assessment.

Your analysis is based on the following provided results:

---

**Part 1: Initial Findings**
{map_act_brain_result}

**Part 2: Functional Brain Map Interpretation**
{image_explain_result}

**Part 3: Neurological Knowledge Base Correlation**
{graph_rag_result}

---

Your final summary report should be structured to cover these key areas:

1.  **Primary Assessment Finding**
    * The condition suggested by the analysis (e.g., consistent with Alzheimer's Disease or Cognitively Normal).
    * Reference path to the functional brain activity visualization.

2.  **Interpretation of Brain Activity Patterns**
    * A detailed description of the observed brain activity, including involved anatomical regions, any notable asymmetry, the functional roles of these areas, and their potential clinical significance.

3.  **Correlation with Established Neurological Knowledge**
    * A summary of how the identified regional activity aligns with known neurological networks and established disease-related associations from the clinical knowledge base.

**Your Final Output Mandate:**
Your **sole final task** is to generate a single JSON object that strictly adheres to the `FinalReport` schema.

**Detailed JSON Field Generation Guidelines:**
1.  **`classification_result`**: Identify the primary assessment finding from the context (e.g., "AD" or "CN") and place it in this field. This represents the outcome of the analysis.
2.  **`visualization_path`**: Extract the file path for the functional brain activity map (e.g., `figures/.../activation_map_...png`) from the context and place it here.
3.  **`final_report_markdown`**: Synthesize ALL information from the provided context into a comprehensive and coherent narrative report formatted in Markdown. This report must read as if written by a clinical expert for another medical professional, seamlessly integrating the primary assessment, the interpretation of brain activity, and the relevant neurological knowledge.
4.  **`final_report_chinese`**: Provide a professional translation of the entire `final_report_markdown` into **TRADITIONAL Chinese**. All medical terminology and clinical nuances must be accurately preserved.

**CRITICAL DIRECTIVES:**
* **AVOID JARGON:** Under no circumstances should the final markdown or Chinese reports contain technical terms like 'AI', 'machine learning', 'model', 'prediction', 'activation map', 'heatmap', or 'knowledge graph'. Instead, use clinical equivalents such as 'analysis', 'assessment', 'finding', 'functional brain map', 'visualization', and 'neurological knowledge base'.
* **DON'T MENTION IMAGE PATH:** We already show the image on the interface, so don't mention it in the report.
* **HANDLE MISSING DATA:** If any context is incomplete, generate the report based on the available information without mentioning the omissions.
* **JSON ONLY:** Your output MUST be a single, valid JSON object and nothing else. Do not prepend or append any explanatory text.
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
