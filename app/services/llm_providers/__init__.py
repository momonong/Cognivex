from typing import Any, List, Optional, Union
from pathlib import Path

from app.services.llm_providers import bedrock

DEFAULT_LLM_PROVIDER = "aws_bedrock"

def llm_response(
    prompt: str | list,
    *,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    model: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    (純文字) 根據 llm_provider 分派請求到對應的 LLM 服務。
    """
    if llm_provider == "aws_bedrock":
        return bedrock.handle_text(prompt=prompt, model=model, **kwargs)


def llm_image_response(
    prompt: str,
    *,
    image_path: Union[str, Path, List[Union[str, Path]]],
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    model: Optional[str] = None,
    **kwargs,
) -> str:
    """
    (文字+圖片) 根據 llm_provider 分派多模態請求。
    """
    if llm_provider == "aws_bedrock":
        return bedrock.handle_image(
            prompt=prompt, image_path=image_path, model=model, **kwargs
        )
