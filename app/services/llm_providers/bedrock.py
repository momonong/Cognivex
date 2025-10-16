import json
import boto3
import base64
import re
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, get_origin, get_args

# --- Client Initialization ---
bedrock_client = boto3.client(
    service_name="bedrock-runtime", region_name="us-east-1"
)
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# --- Helper Functions ---

def _image_to_base64(image_path: Union[str, Path]) -> str:
    """
    將圖片檔案轉換為 Base64 編碼字串，並確保有正確的填充 (padding)。
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            encoded_string = encoded_bytes.decode("utf-8")
            
            # --- 核心修正點 #2：確保 Base64 填充 ---
            # AWS API 要求 Base64 字串長度必須是 4 的倍數。
            missing_padding = len(encoded_string) % 4
            if missing_padding:
                encoded_string += '=' * (4 - missing_padding)
            return encoded_string
            
    except Exception as e:
        raise IOError(f"無法讀取或編碼圖片: {image_path}") from e

def _clean_json_string(s: str) -> str:
    """清理字串，移除所有 ASCII 控制字元。"""
    return re.sub(r'[\x00-\x1F]', '', s)

def _convert_schema_if_needed(schema: Any) -> Optional[Dict[str, Any]]:
    """將 Pydantic 模型轉換為 Bedrock API 要求的 JSON schema。"""
    if schema is None: return None
    target_schema = None
    origin = get_origin(schema)
    if origin in [list, List]:
        inner_type = get_args(schema)[0]
        if hasattr(inner_type, 'model_json_schema'):
            item_schema = inner_type.model_json_schema()
            target_schema = {"type": "array", "items": item_schema}
    elif hasattr(schema, 'model_json_schema'):
        target_schema = schema.model_json_schema()
    elif isinstance(schema, dict):
        target_schema = schema
    else:
        raise TypeError(f"Unsupported type for response_schema: {type(schema)}.")
    return {"type": "object", "properties": {"output": target_schema}, "required": ["output"]}

def _parse_bedrock_response(response_body_bytes: bytes) -> Union[str, Dict, List]:
    """從 Bedrock 回傳中解析出純文字或 Python 物件，並能從混雜文字中提取 JSON。"""
    if not response_body_bytes:
        raise ValueError("從 AWS Bedrock API 收到空的回傳 body。")

    try:
        response_body = json.loads(response_body_bytes)
    except json.JSONDecodeError:
        raise ValueError(f"無法從 Bedrock 解析 JSON。原始回傳內容: {response_body_bytes.decode('utf-8')}")

    content_list = response_body.get("content", [])
    
    if response_body.get("stop_reason") == "tool_use":
        tool_use_block = next((block for block in content_list if block.get("type") == "tool_use"), None)
        if tool_use_block and "input" in tool_use_block:
            response_input = tool_use_block["input"]
            return response_input.get("output", response_input)

    text_block = next((block for block in content_list if block.get("type") == "text"), None)
    if text_block and "text" in text_block:
        raw_text = text_block["text"]
        
        # --- 核心修正點 #1：智慧提取 JSON ---
        # 嘗試從文字中尋找 ```json ... ``` 區塊或獨立的 [ ... ] / { ... }
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```|(\[[\s\S]*\]|\{[\s\S]*\})', raw_text, re.DOTALL)
        if json_match:
            # 優先選擇第一個捕獲組 (```json...```)，否則選擇第二個 ([...])
            json_str = json_match.group(1) or json_match.group(2)
            try:
                # 驗證它是否是有效的 JSON
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 如果提取的不是有效 JSON，則回傳原始文字
                return raw_text
        return raw_text

    raise ValueError(f"在 Bedrock 的回傳中找不到有效的 'text' 或 'tool_use' 區塊。Stop Reason: '{response_body.get('stop_reason')}'.")

# --- 公開處理函式 ---

def handle_text(
    prompt: str | list, *, model_id: str = DEFAULT_BEDROCK_MODEL_ID,
    system_instruction: Optional[str] = None, response_schema: Optional[Any] = None, **kwargs
) -> str:
    """呼叫 AWS Bedrock API，並確保回傳乾淨、可用的 JSON 字串或一般字串。"""
    schema_dict = _convert_schema_if_needed(response_schema)
    prompt_text = str(prompt[-1]) if isinstance(prompt, list) else str(prompt)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    request_body = {
        "messages": messages, "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096, "temperature": 0.1,
    }
    if system_instruction: request_body["system"] = system_instruction
    if schema_dict:
        request_body["tools"] = [{"name": "structured_output", "description": "A tool to format the response.", "input_schema": schema_dict}]
    
    try:
        response = bedrock_client.invoke_model(modelId=model_id, body=json.dumps(request_body))
        response_bytes = response.get("body").read()
        parsed_response = _parse_bedrock_response(response_bytes)

        if isinstance(parsed_response, (dict, list)):
            return _clean_json_string(json.dumps(parsed_response, ensure_ascii=False))
        return _clean_json_string(str(parsed_response))
        
    except Exception as e:
        print(f"[Bedrock Error] 處理文字時發生錯誤: {e}")
        raise

def handle_image(
    prompt: str, *, image_path: Union[str, Path, List[Union[str, Path]]],
    model_id: str = DEFAULT_BEDROCK_MODEL_ID, system_instruction: Optional[str] = None,
    response_schema: Optional[Any] = None, **kwargs
) -> str:
    """呼叫 AWS Bedrock API 處理圖片，並確保回傳乾淨、可用的 JSON 字串或一般字串。"""
    schema_dict = _convert_schema_if_needed(response_schema)
    image_paths = [image_path] if isinstance(image_path, (str, Path)) else image_path
    content_blocks = [{"type": "text", "text": prompt}]
    for path in image_paths:
        media_type = "image/jpeg" if str(path).lower().endswith((".jpg", ".jpeg")) else "image/png"
        content_blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": _image_to_base64(path)}})
    messages = [{"role": "user", "content": content_blocks}]
    request_body = {
        "messages": messages, "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096, "temperature": 0.1,
    }
    if system_instruction: request_body["system"] = system_instruction
    if schema_dict:
        request_body["tools"] = [{"name": "image_analysis_output", "description": "A tool to format the analysis.", "input_schema": schema_dict}]
    
    try:
        response = bedrock_client.invoke_model(modelId=model_id, body=json.dumps(request_body))
        response_bytes = response.get("body").read()
        parsed_response = _parse_bedrock_response(response_bytes)

        if isinstance(parsed_response, (dict, list)):
            return _clean_json_string(json.dumps(parsed_response, ensure_ascii=False))
        return _clean_json_string(str(parsed_response))
        
    except Exception as e:
        print(f"[Bedrock Error] 處理圖片時發生錯誤: {e}")
        raise