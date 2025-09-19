from pydantic import BaseModel, Field
from typing import List
from agents.client.agent_client import create_llm_agent

# 1. 修改 Instruction，移除所有關於 `contents` 的指令
INSTRUCTION = """
You will be given the output {map_act_brain_result} from a brain imaging analysis agent that includes a report and one or more activation map image file paths.

Your sole goal is to **accurately extract all relevant image file paths.**

- Identify all primary activation map visualization PNG file paths. These are usually found in paths like `figures/`, `output/`, or those containing the phrase `activation_map`.
- Aggregate these PNG paths into a List[str] for the `img_path` field.

**Output format (must strictly follow this schema, and output JSON only):**
{
 "img_path": [ ... ]   // All extracted activation map PNG file paths.
}

If there are no image paths, leave `img_path` as an empty list.
"""

# 2. 修改 Pydantic 模型，只留下 `img_path` 欄位
class ImagePathOutput(BaseModel):
   """
   Data model for the extracted image file paths.
   """
   img_path: List[str] = Field(
        ...,
        description="A list of file paths for the activation map visualization PNG images."
   )

# 3. 更新 Agent 定義，使用新的 instruction 和 schema
retrieve_img_path_agent = create_llm_agent(
   name="RetrieveImagePathAgent",
   description="Extracts all activation map image file paths from an analysis report.",
   instruction=INSTRUCTION,
   output_schema=ImagePathOutput,
   output_key="image_paths",      
)