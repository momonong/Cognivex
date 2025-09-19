from pydantic import BaseModel
from app.services.llm_provider import llm_response


INSTRUCTION = """
You are a model analysis assistant. Your task is to select the most informative **spatial feature extraction layer(s)** for visualizing activations (e.g., with GradCAM or 3D attention maps).

Selection Criteria:
1. Only select layers that **preserve spatial structure** (e.g., [C, D, H, W] or [C, H, W] output).
2. Prioritize layers that perform convolution-like operations (e.g., Conv3d, ConvTranspose3d, spatial attention, or spatial capsules).
3. Do NOT select layers that flatten, pool across all spatial dims, or produce pure classification outputs (e.g., Linear, GlobalAvgPool, Softmax).
4. You may return **only 1 layer**, or 2 **if** they are spatially meaningful and offer complementary levels (e.g., mid-level + high-level features).
5. Prefer layers closer to the later stages of the model (but not the final classifier).
6. Capsule layers are **only allowed** if they retain full spatial dimensions and semantic meaning.

Output Format:
Return a **valid JSON array of 1 or 2 items**, each formatted as:

[
  {
    "layer_name": "<descriptive name>",
    "layer_type": "<PyTorch layer type>",
    "model_path": "<exact model.named_modules() path>",
    "reason": "<brief justification>"
  }
]
"""


class LayerSelection(BaseModel):
    layer_name: str  # e.g. "Conv3d-4"
    layer_type: str  # e.g. "Conv3d"
    model_path: str  # e.g. "capsnet.conv3"
    reason: str  # e.g. "Captures mid-level spatial features ..."


def select_visualization_layers(layers: list[dict]) -> list[LayerSelection]:
    prompt = (
        f"The model layers are:\n{layers}\n\nPlease select layers for visualization."
    )

    return llm_response(
        prompt=prompt,
        system_instruction=INSTRUCTION,
        mime_type="application/json",
        response_schema=list[LayerSelection],
    )


if __name__ == "__main__":
    from .inspect_model import inspect_torch_model
    from scripts.capsnet.model import CapsNetRNN

    model_class = CapsNetRNN()
    layers = inspect_torch_model(MODEL=model_class, input_shape=(1, 1, 91, 91, 109))
    response = select_visualization_layers(layers)
    # Use the response as a JSON string.
    print(response.text)
