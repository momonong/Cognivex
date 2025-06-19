# agents/gemini_client.py

import os
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

    if mime_type == "application/json":
        return response.candidates[0].content.parts[0].text  # 結構化 JSON 回傳
