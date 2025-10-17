# 檔案路徑: app/services/llm_providers/bedrock.py
# 說明: 最終版 Bedrock 適配器，包含自動 JSON 清理功能。

import os
import json
import re  # 匯入正規表示式模組
import base64
from typing import Any, Union, Optional, List, Type, get_origin, get_args
from pathlib import Path
from pydantic import BaseModel, Field, create_model
import traceback

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

# --- 初始化 ---
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

chat_client = ChatBedrock(
    model_id=DEFAULT_BEDROCK_MODEL_ID,
    region_name=REGION,
    model_kwargs={"temperature": 0.1, "max_tokens": 4096},
)


# --- Bedrock 專屬輔助函式 ---

def _extract_json_from_text(text: str) -> str:
    """
    從包含閒聊文字和 markdown 區塊的字串中，提取出純淨的 JSON 字串。
    """
    # 尋找被 ```json ... ``` 或 ``` ... ``` 包裹的內容
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    
    if match:
        # 如果找到，回傳第一個捕獲組的內容，並去除前後空白
        return match.group(1).strip()
    
    # 如果沒找到 markdown 區塊，檢查整個字串是否本身就是一個 JSON
    stripped_text = text.strip()
    if (stripped_text.startswith('[') and stripped_text.endswith(']')) or \
       (stripped_text.startswith('{') and stripped_text.endswith('}')):
        return stripped_text

    # 如果以上方法都失敗，回傳原始文字 (這將會讓 Node 2 的 json.loads 失敗，但這是預期行為)
    return text


def _image_to_base64(image_path: Union[str, Path]) -> str:
    # ... (程式碼保持不變)
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _build_content_parts(prompt: str, image_paths: Optional[List[Union[str, Path]]] = None) -> List[dict]:
    # ... (程式碼保持不變)
    parts = [{"type": "text", "text": prompt}]
    if image_paths:
        for p in image_paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at path: {path}")
            media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            parts.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": _image_to_base64(path)}})
    return parts


def _invoke_bedrock(
    prompt: str, *, model_id: str, system_instruction: Optional[str],
    response_schema: Optional[Type[BaseModel]], image_paths: Optional[List[Union[str, Path]]] = None,
) -> str:
    print("\n" + "="*20 + " [BEDROCK INVOKE START] " + "="*20)
    print(f"[DEBUG] Model ID: {model_id}")
    print(f"[DEBUG] Response Schema Provided: {response_schema}")
    
    try:
        client = chat_client if model_id == DEFAULT_BEDROCK_MODEL_ID else ChatBedrock(model_id=model_id, region_name=REGION)
        messages = [SystemMessage(content=system_instruction)] if system_instruction else []
        content_parts = _build_content_parts(prompt, image_paths=image_paths)
        messages.append(HumanMessage(content=content_parts))
        
        if response_schema:
            # --- SCHEMA PROVIDED: LangChain 處理結構化輸出 ---
            # ... (原有邏輯不變)
            schema_to_use, is_list_schema = response_schema, False
            origin = get_origin(response_schema)
            if origin in [list, List]:
                is_list_schema = True
                inner_type = get_args(response_schema)[0]
                schema_to_use = create_model("DynamicListWrapper", output=(List[inner_type], Field(description="A list of items.")))
            structured_llm = client.with_structured_output(schema_to_use)
            result = structured_llm.invoke(messages)
            if is_list_schema:
                pydantic_objects = getattr(result, 'output', [])
                dict_list = [obj.model_dump() for obj in pydantic_objects]
                return json.dumps(dict_list, ensure_ascii=False, indent=2)
            else:
                return result.model_dump_json(indent=2)
        else:
            # --- NO SCHEMA: 執行標準聊天並清理輸出 ---
            response = client.invoke(messages)
            raw_output = response.content
            print(f"[DEBUG] Raw LLM output (before cleaning):\n---\n{raw_output}\n---")
            
            # 【核心修正】呼叫清理函式
            clean_output = _extract_json_from_text(raw_output)
            print(f"[DEBUG] Cleaned JSON output (after cleaning):\n---\n{clean_output}\n---")
            return clean_output

    except Exception as e:
        print("\n" + "X"*20 + " [BEDROCK INVOKE FAILED] " + "X"*20)
        traceback.print_exc()
        raise e
    finally:
        print("="*21 + " [BEDROCK INVOKE END] " + "="*21 + "\n")


# --- 公開介面 (Public API) ---

def handle_text(prompt: Union[str, List], *, model: Optional[str] = None, **kwargs) -> Union[str, Any]:
    prompt_text = prompt[-1] if isinstance(prompt, list) else prompt
    model_to_use = model or DEFAULT_BEDROCK_MODEL_ID
    system_instruction = kwargs.pop("system_instruction", None)
    response_schema = kwargs.pop("response_schema", None)
    
    return _invoke_bedrock(
        prompt=prompt_text, model_id=model_to_use,
        system_instruction=system_instruction, response_schema=response_schema
    )

def handle_image(prompt: str, *, image_path: Union[str, Path, List[Union[str, Path]]], model: Optional[str] = None, **kwargs) -> str:
    image_paths = [image_path] if isinstance(image_path, (str, Path)) else image_path
    model_to_use = model or DEFAULT_BEDROCK_MODEL_ID
    system_instruction = kwargs.pop("system_instruction", None)
    response_schema = kwargs.pop("response_schema", None)

    return _invoke_bedrock(
        prompt=prompt, image_paths=image_paths, model_id=model_to_use,
        system_instruction=system_instruction, response_schema=response_schema
    )