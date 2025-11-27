"""
HuggingFace LLM Provider

Local LLM provider using HuggingFace transformers library.
Supports loading models directly from local directories in SafeTensors format.
"""

import json
import torch
from typing import Optional, Dict, Any
from pathlib import Path

# Import error handling
from .error_handling import (
    retry_with_backoff,
    parse_json_with_recovery,
    log_llm_error,
    LLMConnectionError,
    LLMParsingError
)

# Try to import transformers
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        pipeline,
        TextGenerationPipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Define placeholder types
    TextGenerationPipeline = None
    print("[WARNING] transformers not installed. HuggingFace provider will not be available.")
    print("To install: pip install transformers torch accelerate")


# Model cache
_model_cache: Dict[str, Any] = {}
_tokenizer_cache: Dict[str, Any] = {}


def check_availability() -> bool:
    """Check if transformers is available"""
    return TRANSFORMERS_AVAILABLE


def load_model(
    model_path: str,
    device: str = "cuda:0",
    torch_dtype: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = True
) -> tuple:
    """
    Load model and tokenizer from local path (transformers 4.57.1+ compatible)
    
    Args:
        model_path: Path to model directory (e.g., "D:/hf_models/Llama3.1-Aloe-Beta-8B")
        device: Device to load model on ("auto", "cuda", "cpu")
        torch_dtype: Data type ("auto", "float16", "bfloat16", "float32")
        load_in_8bit: Load model in 8-bit quantization
        load_in_4bit: Load model in 4-bit quantization (recommended)
        trust_remote_code: Trust remote code in model
    
    Returns:
        (model, tokenizer) tuple
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed")
    
    # Check cache
    cache_key = f"{model_path}_{device}_{torch_dtype}_{load_in_4bit}_{load_in_8bit}"
    if cache_key in _model_cache:
        print(f"[INFO] Using cached model: {model_path}")
        return _model_cache[cache_key], _tokenizer_cache[cache_key]
    
    print(f"\n[INFO] Loading model from: {model_path}")
    print(f"[INFO] Device map: {device}")
    print(f"[INFO] Dtype: {torch_dtype}")
    
    # Convert torch_dtype string to torch dtype
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype_obj = dtype_map.get(torch_dtype, torch.float16)
    
    # Load tokenizer
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )
    
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 2025 API (transformers 4.57.1+)
    print("[INFO] Loading model (this may take a few minutes)...")
    
    model_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "device_map": device,
        "trust_remote_code": trust_remote_code
    }
    
    # Add quantization configuration (2025 API)
    if load_in_4bit:
        print("[INFO] Using 4-bit quantization (NF4 with double quantization)")
        model_kwargs.update({
            "load_in_4bit": True,
            "torch_dtype": torch.float16,  # Required for 4-bit
            "bnb_4bit_use_double_quant": True,  # Double quantization for better accuracy
            "bnb_4bit_quant_type": "nf4",  # NF4 quantization type
            "bnb_4bit_compute_dtype": torch.float16  # Compute dtype
        })
    elif load_in_8bit:
        print("[INFO] Using 8-bit quantization")
        model_kwargs.update({
            "load_in_8bit": True,
            "torch_dtype": torch.float16
        })
    else:
        # No quantization
        model_kwargs["torch_dtype"] = torch_dtype_obj
    
    # Try loading with quantization config first, fallback if model is already quantized
    try:
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        print("[OK] Model loaded successfully")
        
        # Print model info
        if hasattr(model, 'config'):
            print(f"[INFO] Model type: {model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'}")
        
        # Print device info
        if hasattr(model, 'device'):
            print(f"[INFO] Model device: {model.device}")
        
    except ValueError as e:
        if "quantized" in str(e).lower() and (load_in_8bit or load_in_4bit):
            # Model is already quantized (e.g., with Mxfp4Config), remove quantization params
            print(f"[INFO] Model is already quantized, loading without additional quantization config")
            model_kwargs = {
                "pretrained_model_name_or_path": model_path,
                "device_map": device,
                "torch_dtype": torch_dtype_obj,
                "trust_remote_code": trust_remote_code
            }
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            print("[OK] Model loaded successfully (using model's native quantization)")
        else:
            raise
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print(f"[INFO] Attempting fallback loading without quantization...")
        # Fallback: try loading without quantization
        model_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "device_map": device,
            "torch_dtype": torch_dtype_obj,
            "trust_remote_code": trust_remote_code
        }
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        print("[OK] Model loaded successfully (fallback mode)")
    
    # Cache model and tokenizer
    _model_cache[cache_key] = model
    _tokenizer_cache[cache_key] = tokenizer
    
    return model, tokenizer


@retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=10.0,
    exceptions=(Exception,),
    verbose=True
)
def handle_text(
    prompt: str,
    *,
    model_path: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.1,
    max_new_tokens: int = 512,
    top_p: float = 0.9,
    top_k: int = 40,
    device: str = "cuda:0",
    torch_dtype: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    **kwargs
) -> str:
    """
    Generate text using HuggingFace model
    
    Includes automatic retry with exponential backoff and error logging.
    
    Args:
        prompt: Input prompt
        model_path: Path to model directory
        system_instruction: System instruction (prepended to prompt)
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        device: Device to use
        torch_dtype: Data type
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
    
    Returns:
        Generated text
    
    Raises:
        LLMConnectionError: If transformers is not available
    
    Requirements: 10.1
    """
    if not TRANSFORMERS_AVAILABLE:
        error = LLMConnectionError("transformers not installed")
        log_llm_error(error, {'provider': 'huggingface', 'issue': 'not_installed'})
        raise error
    
    print("\n" + "="*20 + " [HUGGINGFACE INVOKE START] " + "="*20)
    print(f"[DEBUG] Model: {model_path}")
    print(f"[DEBUG] Temperature: {temperature}")
    print(f"[DEBUG] Max tokens: {max_new_tokens}")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        
        # Format prompt with system instruction
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        print(f"[DEBUG] Prompt length: {len(full_prompt)} chars")
        
        # Tokenize (2025 API)
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # Move to model's device (handles auto device_map correctly)
        inputs = inputs.to(model.device)
        
        print(f"[DEBUG] Input device: {inputs['input_ids'].device}")
        print(f"[DEBUG] Model device: {model.device}")
        
        # Generate (2025 API)
        print("[INFO] Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.1,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output
        if generated_text.startswith(full_prompt):
            generated_text = generated_text[len(full_prompt):].strip()
        
        print(f"[DEBUG] Generated {len(generated_text)} chars")
        print(f"[DEBUG] Output preview: {generated_text[:100]}...")
        
        return generated_text
        
    except Exception as e:
        print("\n" + "X"*20 + " [HUGGINGFACE INVOKE FAILED] " + "X"*20)
        print(f"[ERROR] {type(e).__name__}: {e}")
        
        # Log error with context
        log_llm_error(
            e,
            {
                'provider': 'huggingface',
                'model_path': model_path,
                'temperature': temperature,
                'max_new_tokens': max_new_tokens,
                'device': device
            }
        )
        raise
    finally:
        print("="*21 + " [HUGGINGFACE INVOKE END] " + "="*21 + "\n")


def create_pipeline(
    model_path: str,
    task: str = "text-generation",
    device: str = "auto",
    torch_dtype: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
):
    """
    Create a HuggingFace pipeline for text generation
    
    Args:
        model_path: Path to model directory
        task: Pipeline task (default: "text-generation")
        device: Device to use
        torch_dtype: Data type
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
    
    Returns:
        TextGenerationPipeline
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed")
    
    model, tokenizer = load_model(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit
    )
    
    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer
    )
    
    return pipe


def clear_cache():
    """Clear model cache to free memory"""
    global _model_cache, _tokenizer_cache
    
    print("[INFO] Clearing model cache...")
    
    # Delete models
    for model in _model_cache.values():
        del model
    
    # Clear caches
    _model_cache.clear()
    _tokenizer_cache.clear()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("[OK] Cache cleared")


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about a model
    
    Args:
        model_path: Path to model directory
    
    Returns:
        Dictionary with model info
    """
    model_path = Path(model_path)
    
    info = {
        "path": str(model_path),
        "exists": model_path.exists(),
        "files": []
    }
    
    if model_path.exists():
        # List files
        info["files"] = [f.name for f in model_path.glob("*")]
        
        # Check for config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                info["config"] = json.load(f)
        
        # Check for SafeTensors files
        safetensors_files = list(model_path.glob("*.safetensors"))
        info["safetensors_count"] = len(safetensors_files)
        info["safetensors_files"] = [f.name for f in safetensors_files]
    
    return info


# ============================================================================
# Demo Functions
# ============================================================================

def demo_load_model():
    """Demo: Load a model"""
    print("\n" + "="*80)
    print("DEMO: Load HuggingFace Model")
    print("="*80)
    
    model_path = "D:/hf_models/gpt-oss-20b"
    
    # Get model info
    info = get_model_info(model_path)
    print(f"\nModel Path: {info['path']}")
    print(f"Exists: {info['exists']}")
    print(f"SafeTensors files: {info['safetensors_count']}")
    
    if not info['exists']:
        print("[ERROR] Model not found")
        return
    
    # Load model
    try:
        model, tokenizer = load_model(
            model_path=model_path,
            device="cuda:0",
            torch_dtype="auto",
            load_in_8bit=True  # Use 8-bit to save memory
        )
        print("\n[OK] Model loaded successfully")
        print(f"Model type: {type(model)}")
        print(f"Tokenizer type: {type(tokenizer)}")
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")


def demo_generate_text():
    """Demo: Generate text"""
    print("\n" + "="*80)
    print("DEMO: Generate Text with HuggingFace")
    print("="*80)
    
    model_path = "D:/hf_models/gpt-oss-20b"
    
    try:
        response = handle_text(
            prompt="What is the role of the hippocampus in memory?",
            model_path=model_path,
            system_instruction="You are a helpful medical AI assistant.",
            temperature=0.7,
            max_new_tokens=100,
            load_in_8bit=True
        )
        
        print("\n[Response]")
        print(response)
        
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")


if __name__ == "__main__":
    if not TRANSFORMERS_AVAILABLE:
        print("[ERROR] transformers not installed")
        print("Install with: pip install transformers torch accelerate")
    else:
        # Run demos
        demo_load_model()
        # Uncomment to test generation (requires model to be loaded)
        # demo_generate_text()
