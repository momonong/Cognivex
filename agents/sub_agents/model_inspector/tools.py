import importlib
import torch
import google.generativeai as genai
from dotenv import load_dotenv
import os
from .prompt import INSTRUCTION
from .utils import parse_gemini_json_response 

load_dotenv()

def extract_model_layers(model_path: str, model_class: str, model_module: str):
    # Step 1: Initialize Gemini API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found in environment variables"}
    
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")

    # Step 2: Load model dynamically
    try:
        model_mod = importlib.import_module(model_module)
        model_cls = getattr(model_mod, model_class)
        model = model_cls()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    # Step 3: fetch model layers
    layers = [name for name, _ in model.named_modules() if name != ""]
    if not layers:
        return {"error": "No layers found in model"}

    # Step 4: Setup prompt for Gemini
    prompt = f"""{INSTRUCTION}

    model layers' list:
    {layers}
    """

    try:
        response = model_gemini.generate_content(prompt)
        response_text = response.text.strip()
    except Exception as e:
        return {"layers": layers, "error": f"Gemini failed: {e}"}

    # Step 5: Try to parse JSON (simplified version)
    try:
        response = model_gemini.generate_content(prompt)
        gemini_raw = response.text.strip()
        parsed = parse_gemini_json_response(gemini_raw)

        if parsed and parsed.get("layer_name") in layers:
            recommended_layer = parsed["layer_name"]
            reason = parsed.get("reason", "No reason provided")
        else:
            recommended_layer = layers[-1] if layers else None
            reason = "Failed to parse Gemini response format, using fallback to last layer"

    except Exception as e:
        recommended_layer = layers[-1] if layers else None
        gemini_raw = None
        reason = f"Error using Gemini: {e}"

    return {
        "layers": layers,
        "recommended_layer": recommended_layer,
        "reason": reason,
        "gemini_response": gemini_raw,
    }