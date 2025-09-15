from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from typing import Optional, Type, Any


load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION")


client = genai.Client(vertexai=True,  project=str(GOOGLE_PROJECT_ID), location=str(GOOGLE_LOCATION))

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

def gemini_chat_vertexai(
    prompt: str | list,
    *,
    model: str = "gemini-2.5-flash-lite",
    mime_type: Optional[str] = "text/plain",
    system_instruction: Optional[str] = None,
    response_schema: Optional[Type] = None,
    input_schema: Optional[Type] = None,
) -> str | Any:
    """
    Vertex AI Gemini 聊天函式，支援進階 config (mime_type、system_instruction、schema 等)。
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
    # SDK 有時 response.text，有時 response.candidates
    return getattr(response, "text", response.candidates[0].content.parts[0].text)

# 用法範例
result = gemini_chat_vertexai(
    prompt="請用中文解釋氣泡排序。",
    mime_type="text/plain",
    system_instruction="只用繁體中文回答。",
)

print(result)
