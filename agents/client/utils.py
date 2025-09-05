from typing import Optional, Type, Union
from pathlib import Path
import mimetypes
from google.genai import types

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


def prepare_image_parts_from_paths(
    image_path: Union[str, Path, list[Union[str, Path]]]
) -> list[types.Part]:
    """
    Load one or more images and return a list of Gemini `Part` objects with inline image data.

    Args:
        image_path: Single or list of image file paths (str or Path).

    Returns:
        List of `types.Part` containing inline image data.
    """
    if isinstance(image_path, (str, Path)):
        image_path_list = [Path(image_path)]
    else:
        image_path_list = [Path(p) for p in image_path]

    parts = []
    for img_path in image_path_list:
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        with open(img_path, "rb") as f:
            image_data = f.read()

        mime_type, _ = mimetypes.guess_type(str(img_path))
        if mime_type is None:
            raise ValueError(f"Unable to determine MIME type for {img_path}")

        parts.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))

    return parts

