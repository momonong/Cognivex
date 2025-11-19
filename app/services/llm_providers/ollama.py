"""
Ollama LLM Provider

Local LLM provider for privacy-preserving inference.
Supports structured output and JSON formatting.
"""

import json
import re
from typing import Any, List, Optional, Type, Union
from pathlib import Path
from pydantic import BaseModel

# --- 嘗試導入 ollama，如果失敗則設為 None ---
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False
    print("[WARNING] ollama not installed. Ollama provider will not be available.")
    print("To install: pip install ollama")


# Default configuration
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_BASE_URL = "http://localhost:11434"


def _extract_json_from_text(text: str) -> str:
    """
    Extract JSON from text that may contain markdown code blocks
    
    Similar to Bedrock's JSON extraction logic
    """
    # Try to find JSON in markdown code blocks
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    
    if match:
        return match.group(1).strip()
    
    # Check if the whole text is JSON
    stripped_text = text.strip()
    if (stripped_text.startswith('[') and stripped_text.endswith(']')) or \
       (stripped_text.startswith('{') and stripped_text.endswith('}')):
        return stripped_text
    
    return text


def check_availability() -> bool:
    """Check if Ollama is available and running"""
    if not OLLAMA_AVAILABLE:
        return False
    
    try:
        # Try to list models to check if server is running
        ollama.list()
        return True
    except Exception:
        return False


def list_models() -> List[str]:
    """List available Ollama models"""
    if not OLLAMA_AVAILABLE:
        return []
    
    try:
        models = ollama.list()
        return [model['name'] for model in models.get('models', [])]
    except Exception:
        return []


def handle_text(
    prompt: Union[str, List[str]], 
    *, 
    model: str = DEFAULT_OLLAMA_MODEL,
    system_instruction: Optional[str] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    temperature: float = 0.1,
    **kwargs
) -> str:
    """
    Handle text-only prompts with Ollama
    
    Args:
        prompt: Text prompt or list of prompts
        model: Ollama model name
        system_instruction: System instruction for the model
        response_schema: Pydantic model for structured output
        temperature: Sampling temperature
        **kwargs: Additional arguments
    
    Returns:
        Model response as string (JSON if response_schema provided)
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError(
            "ollama is not installed. Cannot use Ollama provider.\n"
            "Install with: pip install ollama"
        )
    
    if not check_availability():
        raise RuntimeError(
            "Ollama server is not running.\n"
            "Start with: ollama serve\n"
            "Or install from: https://ollama.ai"
        )
    
    print("\n" + "="*20 + " [OLLAMA INVOKE START] " + "="*20)
    print(f"[DEBUG] Model: {model}")
    print(f"[DEBUG] Response Schema: {response_schema}")
    print(f"[DEBUG] Temperature: {temperature}")
    
    try:
        # Prepare messages
        messages = []
        
        # Add system instruction
        if system_instruction:
            messages.append({
                'role': 'system',
                'content': system_instruction
            })
        
        # Add user prompts
        if isinstance(prompt, list):
            for p in prompt:
                messages.append({
                    'role': 'user',
                    'content': p
                })
        else:
            messages.append({
                'role': 'user',
                'content': prompt
            })
        
        # Prepare options
        options = {
            'temperature': temperature,
        }
        
        # Add JSON format instruction if schema provided
        if response_schema:
            # Add JSON format instruction to the last message
            json_instruction = (
                "\n\nPlease respond with valid JSON only. "
                "Do not include any explanatory text before or after the JSON."
            )
            messages[-1]['content'] += json_instruction
            
            # Request JSON format
            options['format'] = 'json'
        
        # Call Ollama
        response = ollama.chat(
            model=model,
            messages=messages,
            options=options
        )
        
        raw_output = response['message']['content']
        print(f"[DEBUG] Raw output:\n---\n{raw_output}\n---")
        
        # Clean output if needed
        if response_schema:
            clean_output = _extract_json_from_text(raw_output)
            print(f"[DEBUG] Cleaned JSON:\n---\n{clean_output}\n---")
            
            # Validate against schema
            try:
                parsed = json.loads(clean_output)
                validated = response_schema(**parsed)
                return validated.model_dump_json(indent=2)
            except Exception as e:
                print(f"[WARNING] Schema validation failed: {e}")
                return clean_output
        
        return raw_output
        
    except Exception as e:
        print("\n" + "X"*20 + " [OLLAMA INVOKE FAILED] " + "X"*20)
        print(f"[ERROR] {type(e).__name__}: {e}")
        raise e
    finally:
        print("="*21 + " [OLLAMA INVOKE END] " + "="*21 + "\n")


def handle_image(
    prompt: str,
    *,
    image_path: Union[str, Path, List[Union[str, Path]]],
    model: str = "llava:7b",  # Vision model
    system_instruction: Optional[str] = None,
    **kwargs
) -> str:
    """
    Handle image + text prompts with Ollama vision models
    
    Args:
        prompt: Text prompt
        image_path: Path to image or list of image paths
        model: Ollama vision model name (e.g., llava, bakllava)
        system_instruction: System instruction
        **kwargs: Additional arguments
    
    Returns:
        Model response as string
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("ollama is not installed. Cannot use Ollama provider.")
    
    if not check_availability():
        raise RuntimeError("Ollama server is not running.")
    
    print("\n" + "="*20 + " [OLLAMA VISION INVOKE START] " + "="*20)
    print(f"[DEBUG] Model: {model}")
    
    try:
        # Prepare image paths
        if isinstance(image_path, (str, Path)):
            image_paths = [str(image_path)]
        else:
            image_paths = [str(p) for p in image_path]
        
        # Prepare messages
        messages = []
        
        if system_instruction:
            messages.append({
                'role': 'system',
                'content': system_instruction
            })
        
        # Add user message with images
        messages.append({
            'role': 'user',
            'content': prompt,
            'images': image_paths
        })
        
        # Call Ollama
        response = ollama.chat(
            model=model,
            messages=messages
        )
        
        output = response['message']['content']
        print(f"[DEBUG] Output:\n---\n{output}\n---")
        
        return output
        
    except Exception as e:
        print("\n" + "X"*20 + " [OLLAMA VISION INVOKE FAILED] " + "X"*20)
        print(f"[ERROR] {type(e).__name__}: {e}")
        raise e
    finally:
        print("="*21 + " [OLLAMA VISION INVOKE END] " + "="*21 + "\n")


def pull_model(model: str) -> bool:
    """
    Pull (download) an Ollama model
    
    Args:
        model: Model name (e.g., "llama3.2:3b")
    
    Returns:
        True if successful
    """
    if not OLLAMA_AVAILABLE:
        print("[ERROR] ollama not installed")
        return False
    
    try:
        print(f"[INFO] Pulling model: {model}")
        print("[INFO] This may take a while...")
        
        # Pull model with progress
        for progress in ollama.pull(model, stream=True):
            status = progress.get('status', '')
            if 'total' in progress and 'completed' in progress:
                percent = (progress['completed'] / progress['total']) * 100
                print(f"\r[PROGRESS] {status}: {percent:.1f}%", end='')
            else:
                print(f"\r[PROGRESS] {status}", end='')
        
        print("\n[OK] Model pulled successfully")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to pull model: {e}")
        return False