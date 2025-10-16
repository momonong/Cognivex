import os
from typing import Any, Union, Optional, Type
from pathlib import Path
import mimetypes
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 初始化 ---
load_dotenv()
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
gemini_client = genai.Client(
    vertexai=True,
    project=str(GOOGLE_CLOUD_PROJECT),
    location=str(GOOGLE_CLOUD_LOCATION),
)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


# --- Gemini 專屬輔助函式 (從 utils.py 移入) ---
def build_gemini_config(
    *,
    mime_type: Optional[str] = "text/plain",
    system_instruction: Optional[str] = None,
    response_schema: Optional[Type] = None,
    temperature: float = 0,
    seed: int = 42,
    **kwargs,  
) -> types.GenerateContentConfig:
    config_kwargs = {
        "response_mime_type": mime_type,
        "system_instruction": system_instruction,
        "response_schema": response_schema,
        "temperature": temperature,
        "seed": seed,
    }
    return types.GenerateContentConfig(**config_kwargs)


def prepare_image_parts_from_paths(
    image_path: Union[str, Path, list[Union[str, Path]]],
) -> list[types.Part]:
    """從路徑載入圖片並回傳 Gemini Part 物件列表。"""
    if isinstance(image_path, (str, Path)):
        image_path_list = [Path(image_path)]
    else:
        image_path_list = [Path(p) for p in image_path]

    parts = []
    for img_path in image_path_list:
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        mime_type, _ = mimetypes.guess_type(str(img_path))
        if mime_type is None:
            raise ValueError(f"Unable to determine MIME type for {img_path}")
        with open(img_path, "rb") as f:
                    image_data = f.read()
        parts.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))
    return parts


def handle_text(
    prompt: str | list, *, model: str = DEFAULT_GEMINI_MODEL, **kwargs
) -> str | Any:
    """(內部) Vertex AI Gemini 聊天函式。"""
    config = build_gemini_config(**kwargs)
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return getattr(response, "text", response.candidates[0].content.parts[0].text)


def handle_image(
    prompt: str,
    *,
    image_path: Union[str, Path, list[Union[str, Path]]],
    model: str = DEFAULT_GEMINI_MODEL,
    **kwargs,
) -> str:
    """(內部) 使用 Gemini 處理一個或多個圖片 + 提示詞。"""
    image_parts = prepare_image_parts_from_paths(image_path)
    parts = [types.Part(text=prompt)] + image_parts
    config = build_gemini_config(**kwargs)
    response = gemini_client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": parts}],
        config=config,
    )
    return getattr(response, "text", response.candidates[0].content.parts[0].text)
