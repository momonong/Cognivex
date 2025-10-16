from typing import Any, List, Optional, Union
from pathlib import Path

from app.services.llm_providers import gemini, bedrock, ollama

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
    elif llm_provider == "gemini":
        return gemini.handle_text(prompt=prompt, model_id=model, **kwargs)
    elif llm_provider == "gpt-oss-20b":
        if not model:
            raise ValueError("Ollama 需要指定模型")
        return ollama.handle_text(prompt=prompt, model=model, **kwargs)
    else:
        raise ValueError(f"不支援的 LLM 供應商: {llm_provider}")


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
    elif llm_provider == "gemini":
        return gemini.handle_image(
            prompt=prompt, image_path=image_path, model_id=model, **kwargs
        )
    else:
        raise ValueError(f"供應商 {llm_provider} 的圖像處理功能尚未實現。")
