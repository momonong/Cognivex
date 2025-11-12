from typing import Any, List, Optional, Type

# --- 嘗試導入 ollama，如果失敗則設為 None ---
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False
    print("[WARNING] ollama not installed. Ollama provider will not be available.")

def handle_text(prompt: str | list, *, model: str, system_instruction: Optional[str] = None, response_schema: Optional[Type] = None, **kwargs) -> str:
    """(內部) 呼叫本地 Ollama API。"""
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("ollama is not installed. Cannot use Ollama provider.")
    if isinstance(prompt, list):
        messages = [{'role': 'user', 'content': m} for m in prompt]
        if system_instruction:
            messages.insert(0, {'role': 'system', 'content': system_instruction})
        
        ollama_kwargs = {'model': model, 'messages': messages}
        if response_schema:
            ollama_kwargs['format'] = response_schema
        response = ollama.chat(**ollama_kwargs)
        return response['message']['content']
    else:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']