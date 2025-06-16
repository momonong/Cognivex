from pydantic import BaseModel


class InferenceInput(BaseModel):
    model_path: str
    model_class: str
    model_module: str
    nii_path: str
    layer_name: str
    window: int = 5
    stride: int = 3
    device: str = "auto"  # optional, can be auto-selected


class InferenceOutput(BaseModel):
    pred: int
    activation_path: str
    subject_id: str
    shape: tuple
