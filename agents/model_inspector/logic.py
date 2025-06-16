import torch
import importlib
from agents.model_inspector.interface import ModelInspectInput, ModelInspectOutput, LayerInfo

def inspect_model_layers(inputs: ModelInspectInput) -> ModelInspectOutput:
    model_module = importlib.import_module("scripts.capsnet.model")
    model_class = getattr(model_module, inputs.model_class)
    model = model_class()
    model.load_state_dict(torch.load(inputs.model_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(*inputs.dummy_input_shape)

    layer_info = []
    hooks = []

    def register_hook(name):
        def hook_fn(module, input, output):
            shape = list(output.shape) if isinstance(output, torch.Tensor) else []
            layer_info.append(LayerInfo(name=name, type=module.__class__.__name__, output_shape=shape))
        return hook_fn

    for name, module in model.named_modules():
        if name == "":
            continue
        if inputs.filter_keywords:
            if not any(k in name for k in inputs.filter_keywords):
                continue
        hooks.append(module.register_forward_hook(register_hook(name)))

    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()

    return ModelInspectOutput(layers=layer_info)
