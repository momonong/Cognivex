"""
LLM Factory - Centralized Model Loading for CDDA Agents

This module provides factory methods for loading LLM models with optimized
configurations for the CDDA dual-agent system:
- Orchestrator Agent (Phi-4-mini): Tool calling and decision making
- Consultant Agent (MedGemma-27b): Medical reasoning and report synthesis

All models use 4-bit quantization to fit within 24GB VRAM.
"""

import torch
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] transformers not installed. LLM factory will not be available.")
    print("To install: pip install transformers torch accelerate bitsandbytes")


class LLMFactory:
    """
    Factory for loading and caching LLM models
    
    Provides optimized configurations for:
    - Phi-4-mini-instruct: Orchestrator (tool calling, JSON output)
    - MedGemma-27b: Consultant (medical reasoning, report generation)
    """
    
    # Model cache to avoid reloading
    _model_cache: Dict[str, Any] = {}
    _tokenizer_cache: Dict[str, Any] = {}
    
    @staticmethod
    def get_4bit_config() -> 'BitsAndBytesConfig':
        """
        Get 4-bit quantization config for memory efficiency
        
        This configuration enables loading large models (20B-27B parameters)
        within 24GB VRAM by using 4-bit quantization.
        
        Returns:
            BitsAndBytesConfig for 4-bit quantization
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    @staticmethod
    def get_orchestrator(
        model_path: str = "D:/hf_models/Phi-4-mini-instruct",
        use_4bit: bool = True,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Load Phi-4-mini-instruct for Orchestrator Agent
        
        Phi-4-mini is optimized for:
        - Tool calling and function invocation
        - Structured JSON output
        - Fast inference with low latency
        - Deterministic decision making (low temperature)
        
        Args:
            model_path: Path to Phi-4-mini model directory
            use_4bit: Use 4-bit quantization (recommended for VRAM)
            device_map: Device mapping strategy
            max_new_tokens: Maximum tokens to generate (512 for short JSON)
            temperature: Sampling temperature (0.1 for deterministic)
        
        Returns:
            Dictionary with 'model', 'tokenizer', 'pipeline', and 'config'
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")
        
        cache_key = f"phi4_{model_path}"
        
        # Check cache
        if cache_key in LLMFactory._model_cache:
            print(f"[INFO] Using cached Phi-4-mini model")
            return {
                'model': LLMFactory._model_cache[cache_key],
                'tokenizer': LLMFactory._tokenizer_cache[cache_key],
                'pipeline': None,  # Will be created on demand
                'config': {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'model_name': 'Phi-4-mini-instruct'
                }
            }
        
        print(f"\n[INFO] Loading Phi-4-mini-instruct from: {model_path}")
        print(f"[INFO] Quantization: {'4-bit' if use_4bit else 'None'}")
        print(f"[INFO] Device map: {device_map}")
        
        # Load tokenizer
        print("[INFO] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Prepare model kwargs
        model_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "device_map": device_map,
            "trust_remote_code": True,  # Phi-4 requires this
            "torch_dtype": torch.float16
        }
        
        # Add quantization config if requested
        if use_4bit:
            try:
                model_kwargs["quantization_config"] = LLMFactory.get_4bit_config()
                print("[INFO] Using 4-bit quantization (BitsAndBytes)")
            except Exception as e:
                print(f"[WARNING] Failed to set quantization config: {e}")
                print("[INFO] Will try loading without explicit quantization config")
        
        # Load model with fallback for pre-quantized models
        print("[INFO] Loading model (this may take a few minutes)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            print("[OK] Phi-4-mini loaded successfully")
        except ValueError as e:
            if "quantized" in str(e).lower() and use_4bit:
                # Model is already quantized, remove quantization config
                print(f"[INFO] Model is already quantized, loading without additional config")
                model_kwargs.pop("quantization_config", None)
                model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                print("[OK] Phi-4-mini loaded successfully (using native quantization)")
            else:
                raise
        
        # Cache model and tokenizer
        LLMFactory._model_cache[cache_key] = model
        LLMFactory._tokenizer_cache[cache_key] = tokenizer
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'pipeline': None,  # Will be created on demand
            'config': {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'model_name': 'Phi-4-mini-instruct'
            }
        }
    
    @staticmethod
    def get_medgemma(
        model_path: str = "D:/hf_models/medgemma-27b-text-it",
        use_4bit: bool = True,
        device_map: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Load MedGemma-27b for Consultant Agent
        
        MedGemma is optimized for:
        - Medical domain reasoning
        - Clinical report generation
        - Long-form text synthesis
        - Nuanced medical interpretation (higher temperature)
        
        Args:
            model_path: Path to MedGemma model directory
            use_4bit: Use 4-bit quantization (recommended for VRAM)
            device_map: Device mapping strategy
            max_new_tokens: Maximum tokens to generate (2048 for reports)
            temperature: Sampling temperature (0.3 for creative synthesis)
        
        Returns:
            Dictionary with 'model', 'tokenizer', 'pipeline', and 'config'
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")
        
        cache_key = f"medgemma_{model_path}"
        
        # Check cache
        if cache_key in LLMFactory._model_cache:
            print(f"[INFO] Using cached MedGemma model")
            return {
                'model': LLMFactory._model_cache[cache_key],
                'tokenizer': LLMFactory._tokenizer_cache[cache_key],
                'pipeline': None,  # Will be created on demand
                'config': {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'model_name': 'MedGemma-27b'
                }
            }
        
        print(f"\n[INFO] Loading MedGemma-27b from: {model_path}")
        print(f"[INFO] Quantization: {'4-bit' if use_4bit else 'None'}")
        print(f"[INFO] Device map: {device_map}")
        
        # Load tokenizer
        print("[INFO] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Prepare model kwargs
        model_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "device_map": device_map,
            "trust_remote_code": True,
            "torch_dtype": torch.float16
        }
        
        # Add quantization config if requested
        if use_4bit:
            try:
                model_kwargs["quantization_config"] = LLMFactory.get_4bit_config()
                print("[INFO] Using 4-bit quantization (BitsAndBytes)")
            except Exception as e:
                print(f"[WARNING] Failed to set quantization config: {e}")
                print("[INFO] Will try loading without explicit quantization config")
        
        # Load model with fallback for pre-quantized models
        print("[INFO] Loading model (this may take a few minutes)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            print("[OK] MedGemma-27b loaded successfully")
        except ValueError as e:
            if "quantized" in str(e).lower() and use_4bit:
                # Model is already quantized, remove quantization config
                print(f"[INFO] Model is already quantized, loading without additional config")
                model_kwargs.pop("quantization_config", None)
                model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                print("[OK] MedGemma-27b loaded successfully (using native quantization)")
            else:
                raise
        
        # Cache model and tokenizer
        LLMFactory._model_cache[cache_key] = model
        LLMFactory._tokenizer_cache[cache_key] = tokenizer
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'pipeline': None,  # Will be created on demand
            'config': {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'model_name': 'MedGemma-27b'
            }
        }
    
    @staticmethod
    def clear_cache():
        """Clear model cache to free VRAM"""
        print("[INFO] Clearing LLM cache...")
        
        # Delete models
        for model in LLMFactory._model_cache.values():
            del model
        
        # Clear caches
        LLMFactory._model_cache.clear()
        LLMFactory._tokenizer_cache.clear()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[OK] Cache cleared")
    
    @staticmethod
    def get_vram_usage() -> Dict[str, float]:
        """
        Get current VRAM usage
        
        Returns:
            Dictionary with VRAM statistics in GB
        """
        if not torch.cuda.is_available():
            return {'available': False}
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            'available': True,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated
        }


# ============================================================================
# Demo Functions
# ============================================================================

def demo_load_orchestrator():
    """Demo: Load Phi-4-mini for Orchestrator"""
    print("\n" + "="*80)
    print("DEMO: Load Phi-4-mini-instruct (Orchestrator)")
    print("="*80)
    
    try:
        # Show initial VRAM
        vram = LLMFactory.get_vram_usage()
        if vram['available']:
            print(f"\nInitial VRAM: {vram['allocated_gb']:.2f}GB / {vram['total_gb']:.2f}GB")
        
        # Load model
        orchestrator = LLMFactory.get_orchestrator(
            model_path="D:/hf_models/Phi-4-mini-instruct",
            use_4bit=True
        )
        
        print(f"\n[OK] Orchestrator loaded: {orchestrator['config']['model_name']}")
        print(f"   Max tokens: {orchestrator['config']['max_new_tokens']}")
        print(f"   Temperature: {orchestrator['config']['temperature']}")
        
        # Show VRAM after loading
        vram = LLMFactory.get_vram_usage()
        if vram['available']:
            print(f"\nVRAM after loading: {vram['allocated_gb']:.2f}GB / {vram['total_gb']:.2f}GB")
            print(f"Free VRAM: {vram['free_gb']:.2f}GB")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load Orchestrator: {e}")


def demo_load_consultant():
    """Demo: Load MedGemma-27b for Consultant"""
    print("\n" + "="*80)
    print("DEMO: Load MedGemma-27b (Consultant)")
    print("="*80)
    
    try:
        # Show initial VRAM
        vram = LLMFactory.get_vram_usage()
        if vram['available']:
            print(f"\nInitial VRAM: {vram['allocated_gb']:.2f}GB / {vram['total_gb']:.2f}GB")
        
        # Load model
        consultant = LLMFactory.get_medgemma(
            model_path="D:/hf_models/medgemma-27b",
            use_4bit=True
        )
        
        print(f"\n[OK] Consultant loaded: {consultant['config']['model_name']}")
        print(f"   Max tokens: {consultant['config']['max_new_tokens']}")
        print(f"   Temperature: {consultant['config']['temperature']}")
        
        # Show VRAM after loading
        vram = LLMFactory.get_vram_usage()
        if vram['available']:
            print(f"\nVRAM after loading: {vram['allocated_gb']:.2f}GB / {vram['total_gb']:.2f}GB")
            print(f"Free VRAM: {vram['free_gb']:.2f}GB")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load Consultant: {e}")


def demo_load_both():
    """Demo: Load both models to test VRAM usage"""
    print("\n" + "="*80)
    print("DEMO: Load Both Models (Phi-4 + MedGemma)")
    print("="*80)
    
    try:
        # Show initial VRAM
        vram = LLMFactory.get_vram_usage()
        if vram['available']:
            print(f"\nInitial VRAM: {vram['allocated_gb']:.2f}GB / {vram['total_gb']:.2f}GB")
        
        # Load Orchestrator
        print("\n[1/2] Loading Orchestrator (Phi-4-mini)...")
        orchestrator = LLMFactory.get_orchestrator(use_4bit=True)
        
        vram = LLMFactory.get_vram_usage()
        if vram['available']:
            print(f"VRAM after Orchestrator: {vram['allocated_gb']:.2f}GB / {vram['total_gb']:.2f}GB")
        
        # Load Consultant
        print("\n[2/2] Loading Consultant (MedGemma-27b-text-it)...")
        consultant = LLMFactory.get_medgemma(use_4bit=True)
        
        vram = LLMFactory.get_vram_usage()
        if vram['available']:
            print(f"\nFinal VRAM: {vram['allocated_gb']:.2f}GB / {vram['total_gb']:.2f}GB")
            print(f"Free VRAM: {vram['free_gb']:.2f}GB")
            
            if vram['allocated_gb'] > 24:
                print("\n[WARNING] VRAM usage exceeds 24GB!")
                print("[SUGGESTION] Both models may not fit simultaneously")
            else:
                print(f"\n[OK] Both models fit within 24GB VRAM")
        
        print("\n[SUCCESS] Both models loaded successfully")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load models: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not TRANSFORMERS_AVAILABLE:
        print("[ERROR] transformers not installed")
        print("Install with: pip install transformers torch accelerate bitsandbytes")
    else:
        # Run demos
        print("\nSelect demo:")
        print("1. Load Orchestrator (Phi-4-mini)")
        print("2. Load Consultant (MedGemma-27b)")
        print("3. Load Both Models")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            demo_load_orchestrator()
        elif choice == "2":
            demo_load_consultant()
        elif choice == "3":
            demo_load_both()
        else:
            print("Invalid choice. Running demo 3 (load both)...")
            demo_load_both()
