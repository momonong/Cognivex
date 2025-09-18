from pydantic import BaseModel, Field
from typing import List
from agents.client.agent_client import create_llm_agent

INSTRUCTION = """
You will be given the output {map_act_brain_result} from a brain imaging analysis agent that includes both the main report and one or more activation map image file paths (typically ending in .png).

Your goals are:

1. **Accurately extract all relevant image file paths.**
   - Identify all primary activation map visualization PNG file paths. These are usually found in paths like `figures/`, `output/`, or those containing the phrase `activation_map` or similar.
   - Aggregate these PNG paths into a List[str] for the field `img_path`.

2. **Copy the main report contents into the output.**
   - Take the agent’s primary report body, including all text and markdown, and place it in the `contents` field.
   - Do NOT rewrite or summarize; transfer the content as given.

**Output format (must strictly follow this schema, and output JSON only):**
{
 "img_path": [ ... ],   // All extracted activation map PNG file paths.
 "contents": "<main report text, markdown, or summary>"
}

If there are no image paths, leave `img_path` as an empty list; if there is no report content, leave `contents` as an empty string.
"""


class ImgPath(BaseModel):
   """
   The simplified, final output of the entire pipeline,
   designed for direct use in a UI.
   """
   img_path: List[str] = Field(
        ...,
        description="The file paths to the primary activation map visualization PNG image for display."
   )
   contents: str = Field(
        ...,
        description="The main content of the last agent output."
   )

retrieve_img_path_agent = create_llm_agent(
   name="RetrieveImgAgent",
   # model=LiteLlm(model="ollama_chat/gpt-oss:20b"),
   # model="gemini-2.5-flash-lite",
   description="Retrieve the image path within the first agent and pass it to second agent.",
   instruction=INSTRUCTION,
   output_schema=ImgPath,
   output_key="img_path_with_report",
)
