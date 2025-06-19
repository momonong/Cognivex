import os
import json
import torch
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

INSTRUCTION = """
You are a model activation filter assistant.

Your task is to select the most informative and meaningful layers for **visualizing model activations** (e.g., using GradCAM or 3D attention maps), based on their activation statistics and metadata.

Input: A list of layers, each containing:
- layer_name: string
- layer_type: string (e.g., Conv3d, CapsuleLayer3D)
- model_path: string (e.g., capsnet.conv3)
- nonzero_ratio: float
- mean_activation: float
- std_activation: float
- max_activation: float

Selection Guidelines:
1. Favor layers with:
   - nonzero_ratio > 0.1 (indicates broad spatial activation)
   - mean_activation > 0.001 (avoids near-zero useless maps)
2. Drop layers with:
   - Extremely sparse or weak activations
   - Final classifier layers or non-spatial layers (like Linear, GlobalPool)
3. Capsule layers are allowed **only** if they retain spatial structure.

Output:
A JSON array of the **selected layers**, each with:
- model_path: exact identifier of the layer (string)
- reason: brief justification for selecting this layer (string)
"""


class SelectedLayer(BaseModel):
    model_path: str
    reason: str


def get_activation_stats(act_path: str):
    act = torch.load(act_path)
    act = act[0] if act.dim() == 5 else act
    return {
        "nonzero_ratio": round((act > 1e-4).float().mean().item(), 6),
        "mean_activation": round(act.mean().item(), 6),
        "std_activation": round(act.std().item(), 6),
        "max_activation": round(act.max().item(), 6),
    }


def filter_layers_by_gemini(
    selected_layers: list[dict],
    activation_dir: str,
    save_name_prefix: str,
    delete_rejected=True,
):
    """Call Gemini to decide which layers to keep based on activation stats + semantic metadata"""
    layer_inputs = []

    for layer in selected_layers:
        model_path = layer["model_path"]
        safe_name = model_path.replace(".", "_")
        act_path = os.path.join(activation_dir, f"{save_name_prefix}_{safe_name}.pt")
        if not os.path.exists(act_path):
            print(f"[Skip] Activation not found: {act_path}")
            continue
        stats = get_activation_stats(act_path)
        layer_inputs.append(
            {
                "model_path": model_path,
                "layer_name": layer["layer_name"],
                "layer_type": layer["layer_type"],
                "reason": layer.get("reason", ""),
                **stats,
            }
        )

    print("[Activation Layer Stats Before Gemini Filtering]")
    for layer in layer_inputs:
        print(f"• {layer['model_path']:30} | mean={layer['mean_activation']:.6f} | nonzero={layer['nonzero_ratio']:.4f}")

    prompt = f"The model layer activations are as follows:\n{layer_inputs}\n\nWhich ones should we keep?"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTION,
            response_mime_type="application/json",
            response_schema=list[SelectedLayer],
        ),
    )

    keep_entries = json.loads(response.text)
    keep_model_paths = [entry["model_path"] for entry in keep_entries]

    print(f"[Gemini Selected Layers]:")
    for entry in keep_entries:
        print(f"✔ {entry['model_path']:30} — {entry['reason']}")

    final_layers = []
    for layer in selected_layers:
        if layer["model_path"] in keep_model_paths:
            final_layers.append(layer)
        else:
            print(f"✘ Dropping: {layer['model_path']} - {layer['reason']}")
            safe_name = layer["model_path"].replace(".", "_")
            act_path = os.path.join(activation_dir, f"{save_name_prefix}_{safe_name}.pt")
            if delete_rejected and os.path.exists(act_path):
                os.remove(act_path)
                print(f"Deleted: {act_path}")

    return final_layers
