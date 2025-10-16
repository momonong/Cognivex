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
    """將圖片檔案轉換為 Base64 編碼字串，並確保有正確的填充 (padding)。"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            encoded_string = encoded_bytes.decode("utf-8")
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


def _extract_json_from_string(text: str) -> Optional[Union[Dict, List]]:
    """
    從可能混雜了其他文字的字串中，精準地提取出第一個完整的 JSON 物件或陣列。
    """
    first_bracket_pos = -1
    first_brace_pos = text.find('{')
    first_square_pos = text.find('[')

    if first_brace_pos == -1:
        first_bracket_pos = first_square_pos
    elif first_square_pos == -1:
        first_bracket_pos = first_brace_pos
    else:
        first_bracket_pos = min(first_brace_pos, first_square_pos)
        
    if first_bracket_pos == -1:
        return None

    start_char = text[first_bracket_pos]
    end_char = '}' if start_char == '{' else ']'
    
    open_brackets = 0
    for i in range(first_bracket_pos, len(text)):
        if text[i] == start_char:
            open_brackets += 1
        elif text[i] == end_char:
            open_brackets -= 1
        
        if open_brackets == 0:
            potential_json = text[first_bracket_pos : i + 1]
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError:
                return None
    
    return None

def _parse_bedrock_response(response_body_bytes: bytes) -> Union[str, Dict, List]:
    """從 Bedrock 回傳中解析出純文字或 Python 物件，並能從混雜文字中提取 JSON。"""
    if not response_body_bytes:
        raise ValueError("從 AWS Bedrock API 收到空的回傳 body。")

    response_body = json.loads(response_body_bytes.decode('utf-8'))
    content_list = response_body.get("content", [])
    
    if response_body.get("stop_reason") == "tool_use":
        tool_use_block = next((block for block in content_list if block.get("type") == "tool_use"), None)
        if tool_use_block and "input" in tool_use_block:
            output_data = tool_use_block["input"].get("output", tool_use_block["input"])
            
            # --- 核心修正點：無差別淨化程序 ---
            # 如果 tool_use 的輸出是個字串，就必須把它當成髒數據，送進提取器淨化
            if isinstance(output_data, str):
                return _extract_json_from_string(output_data)
            
            # 如果它已經是個 dict 或 list，就直接回傳
            return output_data

    text_block = next((block for block in content_list if block.get("type") == "text"), None)
    if text_block and "text" in text_block:
        raw_text = text_block["text"]
        
        extracted_json = _extract_json_from_string(raw_text)
        if extracted_json is not None:
            return extracted_json
        
        return raw_text

    raise ValueError(f"在 Bedrock 的回傳中找不到有效的 'text' 或 'tool_use' 區塊。")

# --- 公開處理函式 ---

def handle_text(
    prompt: str | list, *, model_id: str = DEFAULT_BEDROCK_MODEL_ID,
    system_instruction: Optional[str] = None, response_schema: Optional[Any] = None, **kwargs
) -> str:
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