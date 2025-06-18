import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

INSTRUCTION = """
You are a model analysis assistant. Your task is to help select the most suitable layer(s) for visualization or activation mapping from a list of PyTorch model layers.

Selection Criteria:
1. Choose layers that likely capture high-level or abstract semantic features.
2. Prefer layers that are not the final classification or output layers.
3. Favor layers that preserve some spatial structure, suitable for GradCAM-like interpretation.

You must return ONLY 1 or 2 layers that best meet the criteria. Avoid returning more than 2.

Output Format:
Return a valid JSON array of objects in the following format, with no extra explanation:

[
  {
    "layer_name": "<human-readable name>",
    "layer_type": "<PyTorch class>",
    "model_path": "<exact name from model.named_modules()>",
    "reason": "<brief justification>"
  }
]
"""



class LayerSelection(BaseModel):
    layer_name: str  # e.g. "Conv3d-3"
    layer_type: str  # e.g. "Conv3d"
    model_path: str  # e.g. "capsnet.conv2"
    reason: str  # e.g. "Captures mid-level spatial features ..."


def select_visualization_layers(layers: list[dict]) -> list[dict]:
    prompt = f"""The model layers are:\n{layers}\n\n"""
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,  # prompt 包含 layers 的 json list 字串
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTION,
            response_mime_type="application/json",
            response_schema=list[LayerSelection],
        ),
    )
    return response


if __name__ == "__main__":
    from .inspect_model import inspect_torch_model
    from scripts.capsnet.model import CapsNetRNN

    model_class = CapsNetRNN()
    layers = inspect_torch_model(MODEL=model_class, input_shape=(1, 1, 91, 91, 109))
    response = select_visualization_layers(layers)
    # Use the response as a JSON string.
    print(response.text)
