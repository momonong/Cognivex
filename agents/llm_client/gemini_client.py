import os
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Optional, Any, Type

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

DEFAULT_MODEL = "gemini-2.5-flash"


def build_gemini_config(
    *,
    mime_type: Optional[str] = "text/plain",
    system_instruction: Optional[str] = None,
    response_schema: Optional[Type] = None,
    input_schema: Optional[Type] = None,
) -> types.GenerateContentConfig:
    kwargs = {
        "response_mime_type": mime_type,
        "system_instruction": system_instruction,
        "response_schema": response_schema,
    }

    if input_schema is not None:
        kwargs["input_schema"] = input_schema

    return types.GenerateContentConfig(**kwargs)


def gemini_chat(
    prompt: str | list,
    *,
    model: str = DEFAULT_MODEL,
    mime_type: Optional[str] = "text/plain",
    system_instruction: Optional[str] = None,
    response_schema: Optional[Type] = None,
    input_schema: Optional[Type] = None,
) -> str | Any:
    """
    Call Gemini model with flexible configuration.
    Returns either plain text or structured object depending on mime_type and response_schema.
    """
    config = build_gemini_config(
        mime_type=mime_type,
        system_instruction=system_instruction,
        response_schema=response_schema,
        input_schema=input_schema,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    return response.candidates[0].content.parts[0].text


def gemini_image(
    image_path: str,
    prompt: str = "What can you see in this image?",
    model: str = DEFAULT_MODEL,  # 選用支援 image 的 model
    mime_type: Optional[str] = None,
) -> str:
    """
    Call Gemini with an image + prompt and return the generated response text.

    Args:
        image_path (str): Path to the image file (JPEG/PNG).
        prompt (str): Prompt or question for the image.
        model (str): Gemini model that supports vision.
        mime_type (Optional[str]): Optional MIME type, auto-detected if None.

    Returns:
        str: Model's textual response.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as f:
        image_data = f.read()

    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            raise ValueError("Unable to determine MIME type of the image.")

    response = client.models.generate_content(
        model=model,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": image_data}},
                ],
            }
        ],
    )

    return response.candidates[0].content.parts[0].text
