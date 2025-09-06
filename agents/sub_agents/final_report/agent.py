from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent
from sympy import true

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
1.  **`visualization_path`**: First, you MUST find the primary activation map PNG path from the provided context (likely from the `{map_act_brain_result}`) and place it in this field.
2.  **`final_report_markdown`**: Next, you MUST synthesize ALL the information from ALL context fields into a single, comprehensive, and coherent narrative report formatted in Markdown. This report should seamlessly integrate the model's findings, the visual interpretation, and the knowledge graph insights, following the professional structure of a clinical analysis.

Do not add any extra commentary outside of the JSON object.
"""

class FinalReport(BaseModel):
    """
    The simplified, final output of the entire pipeline,
    designed for direct use in a UI.
    """
    final_report_markdown: str = Field(
        ..., 
        description="The complete, final clinical report formatted in Markdown, integrating all findings from all previous agents."
    )
    visualization_path: str = Field(
        ..., 
        description="The file path to the primary activation map visualization PNG image for display."
    )

report_generator_agent = LlmAgent(
   name="ReportGeneratorAgent",
   model="gemini-2.5-flash-lite",
   description="Integrates outputs from previous steps and writes the final clinical report.",
   instruction=INSTRUCTION,
   output_schema=FinalReport,
   output_key="final_report",
   disallow_transfer_to_peers=True,
   disallow_transfer_to_parent=True,
)
