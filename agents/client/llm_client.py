import os
from pathlib import Path
from typing import Union, Optional
from dotenv import load_dotenv
from google import genai
from typing import Optional, Any, Type
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI


from agents.client.utils import build_gemini_config
from agents.client.utils import prepare_image_parts_from_paths

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

DEFAULT_MODEL = "gemini-2.5-flash-lite"


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
    prompt: str,
    *,
    image_path: Union[str, Path, list[Union[str, Path]]],
    system_instruction: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    mime_type: Optional[str] = "text/plain",
    response_schema: Optional[Type] = None,
    input_schema: Optional[Type] = None,
) -> str:
    """
    Call Gemini with one or multiple images + prompt and return generated response.

    Args:
        image_path (Union[str, Path, List[str]]): Path(s) to image(s).
        prompt (str): Prompt or question for the image(s).
        model (str): Gemini model that supports vision.
        system_instruction (Optional[str]): Optional system instruction.
        response_schema (Optional[Type]): Structured output schema class (Pydantic).
        input_schema (Optional[Type]): Structured input schema class (Pydantic).

    Returns:
        str: Model's textual response.
    """

    image_parts = prepare_image_parts_from_paths(image_path)
    parts = [types.Part(text=prompt)] + image_parts

    # Optional schema configuration
    config = build_gemini_config(
        mime_type=mime_type,
        system_instruction=system_instruction,
        response_schema=response_schema,
        input_schema=input_schema,
    )

    # Send multimodal request
    response = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": parts}],
        config=config,
    )

    # Return raw text (structured parsing可另接)
    return response.text


def llm_response(
    prompt: str | list,
    *,
    llm_provider: str = "gemini",
    model: str = DEFAULT_MODEL,
    mime_type: Optional[str] = "text/plain",
    system_instruction: Optional[str] = None,
    response_schema: Optional[Type] = None,
    input_schema: Optional[Type] = None,
) -> str | Any:

    if llm_provider.startswith("gemini"):    
        response = gemini_chat(
            prompt=prompt,
            model=model,
            mime_type=mime_type,
            system_instruction=system_instruction,
            response_schema=response_schema,
            input_schema=input_schema,
    )
    return response

def load_llm(model="gemini-2.5-flash-lite") -> ChatGoogleGenerativeAI:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=0,
    )
    return llm