import ollama
from typing import Any, List, Optional, Type

def handle_text(prompt: str | list, *, model: str, system_instruction: Optional[str] = None, response_schema: Optional[Type] = None, **kwargs) -> str:
    """(內部) 呼叫本地 Ollama API。"""
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