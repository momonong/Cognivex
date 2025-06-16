from agents.model_inspector.interface import ModelInspectInput, ModelInspectOutput
from agents.model_inspector.logic import inspect_model_layers

def handler(inputs: ModelInspectInput) -> ModelInspectOutput:
    return inspect_model_layers(inputs)
