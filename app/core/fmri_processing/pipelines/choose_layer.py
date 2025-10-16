from pydantic import BaseModel
from app.services.llm_providers import llm_response


# Strategy-based layer selection instructions
STRATEGY_INSTRUCTIONS = {
    "capsule_focused": """
You are a model analysis assistant for Capsule Networks. Your task is to select the most informative layers for visualizing capsule-based activations.

Selection Criteria for Capsule Networks:
1. Prioritize **capsule layers** that retain spatial structure and semantic meaning.
2. Include key **convolutional layers** that feed into capsules (e.g., feature extraction convolutions).
3. Select layers that preserve spatial structure (e.g., [C, D, H, W] or [Caps, Dim, D, H, W] output).
4. Do NOT select layers that flatten completely or produce pure classification outputs.
5. You may return 1-3 layers: typically early conv + mid-level capsule + high-level capsule.
6. Focus on layers that capture both spatial and part-whole relationships.

Output Format:
Return a **valid JSON array**, each formatted as:
[
  {
    "layer_name": "<descriptive name>",
    "layer_type": "<PyTorch layer type>", 
    "model_path": "<exact model.named_modules() path>",
    "reason": "<brief justification>"
  }
]
""",
    
    "conv_focused": """
You are a model analysis assistant for Convolutional Networks. Your task is to select the most informative **spatial feature extraction layers** for visualizing activations.

Selection Criteria for CNNs:
1. Only select layers that **preserve spatial structure** (e.g., [C, D, H, W] or [C, H, W] output).
2. Prioritize **convolutional layers** that perform spatial feature extraction.
3. Do NOT select layers that flatten, pool across all spatial dims, or produce pure classification outputs.
4. You may return 1-2 layers that offer complementary levels (e.g., mid-level + high-level features).
5. Prefer layers closer to the later stages of the model (but not the final classifier).
6. Focus on layers with rich spatial representations.

Output Format:
Return a **valid JSON array**, each formatted as:
[
  {
    "layer_name": "<descriptive name>",
    "layer_type": "<PyTorch layer type>",
    "model_path": "<exact model.named_modules() path>",
    "reason": "<brief justification>"
  }
]
""",

    "default": """
You are a model analysis assistant. Your task is to select the most informative **spatial feature extraction layer(s)** for visualizing activations (e.g., with GradCAM or 3D attention maps).

General Selection Criteria:
1. Only select layers that **preserve spatial structure** (e.g., [C, D, H, W] or [C, H, W] output).
2. Prioritize layers that perform convolution-like operations (e.g., Conv3d, ConvTranspose3d, spatial attention).
3. Do NOT select layers that flatten, pool across all spatial dims, or produce pure classification outputs (e.g., Linear, GlobalAvgPool, Softmax).
4. You may return **only 1 layer**, or 2 **if** they are spatially meaningful and offer complementary levels.
5. Prefer layers closer to the later stages of the model (but not the final classifier).

Output Format:
Return a **valid JSON array**, each formatted as:
[
  {
    "layer_name": "<descriptive name>",
    "layer_type": "<PyTorch layer type>",
    "model_path": "<exact model.named_modules() path>",
    "reason": "<brief justification>"
  }
]
""",

    "improved_capsule": """
You are a model analysis assistant for Capsule Networks with IMPROVED selection criteria based on visualization analysis.

IMPROVED Selection Criteria for Capsule Networks:
1. **Avoid very early layers** (conv1) - they tend to show low-level noise rather than meaningful patterns.
2. **STRONGLY PRIORITIZE conv3 over conv2** - conv3 contains the most meaningful spatial features for visualization.
3. **Select ONE primary capsule layer** (caps1) - avoid redundant capsule selections.
4. **Focus on layers that preserve spatial dimensions** and show interpretable patterns.
5. **conv3 is the PREFERRED choice** for convolutional layer visualization.

Recommended selection pattern: **conv3 + caps1** (prioritize conv3 as primary visualization layer)

Output Format:
Return a **valid JSON array** with 2-3 layers, each formatted as:
[
  {
    "layer_name": "<descriptive name>",
    "layer_type": "<PyTorch layer type>",
    "model_path": "<exact model.named_modules() path>",
    "reason": "<brief justification focusing on visualization quality>"
  }
]
""",

    "improved_conv": """
You are a model analysis assistant for Convolutional Networks with IMPROVED selection criteria.

IMPROVED Selection Criteria for CNNs:
1. **Focus on middle-to-late convolutional layers** - they provide the best balance of spatial detail and semantic meaning.
2. **Avoid first layer** (conv0) - usually too low-level and noisy for meaningful visualization.
3. **Avoid final layers** before classification - they may be too abstracted.
4. **Select layers with good spatial resolution** but sufficient feature complexity.
5. **Prefer layers that are most relevant to the classification task**.

For typical CNN: Select conv1 (mid-level) AND conv2 (high-level) if both exist.

Output Format:
Return a **valid JSON array** with 1-2 layers, each formatted as:
[
  {
    "layer_name": "<descriptive name>",
    "layer_type": "<PyTorch layer type>",
    "model_path": "<exact model.named_modules() path>",
    "reason": "<brief justification focusing on visualization quality>"
  }
]
"""
}


class LayerSelection(BaseModel):
    layer_name: str  # e.g. "Conv3d-4"
    layer_type: str  # e.g. "Conv3d"
    model_path: str  # e.g. "capsnet.conv3"
    reason: str  # e.g. "Captures mid-level spatial features ..."


def select_visualization_layers(layers: list[dict], strategy: str = "default") -> str:
    """
    Select visualization layers using specified strategy.
    
    Args:
        layers: List of layer information from model inspection
        strategy: Selection strategy ('capsule_focused', 'conv_focused', 'default')
        
    Returns:
        JSON string of selected layers
    """
    # Get strategy-specific instruction
    instruction = STRATEGY_INSTRUCTIONS.get(strategy, STRATEGY_INSTRUCTIONS["default"])
    
    prompt = (
        f"The model layers are:\n{layers}\n\nPlease select layers for visualization."
    )

    return llm_response(
        prompt=prompt,
        system_instruction=instruction,
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
