from pydantic import BaseModel
from typing import List, Optional

class ModelInspectInput(BaseModel):
    model_path: str
    model_class: str = "CapsNetRNN"
    dummy_input_shape: List[int] = [1, 1, 5, 32, 64, 64]
    filter_keywords: Optional[List[str]] = None

class LayerInfo(BaseModel):
    name: str
    type: str
    output_shape: List[int]

class ModelInspectOutput(BaseModel):
    layers: List[LayerInfo]
