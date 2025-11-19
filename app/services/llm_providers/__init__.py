"""
LLM Providers Module

Unified interface for multiple LLM providers:
- AWS Bedrock (cloud)
- Ollama (local, privacy-preserving)
- Gemini (cloud)
"""

from typing import Any, List, Optional, Union
from pathlib import Path

from app.services.llm_providers import gemini, bedrock, ollama
from app.services.llm_providers.config import (
    get_default_config,
    get_config_by_name,
    LLMConfig,
    PRIVACY_MODE
)

# Get default configuration
_default_config = get_default_config()
DEFAULT_LLM_PROVIDER = _default_config.provider


def llm_response(
    prompt: Union[str, List[str]],
    *,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    **kwargs,
) -> Any:
    """
    Unified LLM text response interface
    
    Args:
        prompt: Text prompt or list of prompts
        llm_provider: Provider name ('aws_bedrock', 'ollama', 'gemini')
                     If None, uses default from config
        model: Model name (overrides config)
        config: LLMConfig object (overrides llm_provider and model)
        **kwargs: Additional arguments passed to provider
    
    Returns:
        Model response (string or structured output)
    
    Examples:
        # Use default provider
        response = llm_response("Hello")
        
        # Use specific provider
        response = llm_response("Hello", llm_provider="ollama")
        
        # Use specific model
        response = llm_response("Hello", llm_provider="ollama", model="llama3.1:8b")
        
        # Use config object
        config = get_config_by_name("ollama_large")
        response = llm_response("Hello", config=config)
        
        # Privacy mode (force local)
        import os
        os.environ['PRIVACY_MODE'] = 'true'
        response = llm_response("Hello")  # Will use Ollama
    """
    # Determine configuration
    if config is None:
        if llm_provider is None:
            config = get_default_config()
        else:
            try:
                config = get_config_by_name(llm_provider)
            except ValueError:
                # Fallback to creating basic config
                config = LLMConfig(provider=llm_provider, model=model or "default")
    
    # Override model if specified
    if model is not None:
        config.model = model
    
    # Route to appropriate provider
    provider = config.provider
    
    if provider == "aws_bedrock":
        return bedrock.handle_text(
            prompt=prompt,
            model=config.model,
            **kwargs
        )
    
    elif provider == "ollama":
        # Check if Ollama is available
        if not ollama.OLLAMA_AVAILABLE:
            print("[WARNING] Ollama not available, falling back to Bedrock")
            return bedrock.handle_text(prompt=prompt, model=None, **kwargs)
        
        if not ollama.check_availability():
            print("[WARNING] Ollama server not running, falling back to Bedrock")
            return bedrock.handle_text(prompt=prompt, model=None, **kwargs)
        
        return ollama.handle_text(
            prompt=prompt,
            model=config.model,
            temperature=config.temperature,
            **kwargs
        )
    
    elif provider == "gemini":
        return gemini.handle_text(
            prompt=prompt,
            model_id=config.model,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def llm_image_response(
    prompt: str,
    *,
    image_path: Union[str, Path, List[Union[str, Path]]],
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    **kwargs,
) -> str:
    """
    Unified LLM image + text response interface
    
    Args:
        prompt: Text prompt
        image_path: Path to image or list of image paths
        llm_provider: Provider name
        model: Model name
        config: LLMConfig object
        **kwargs: Additional arguments
    
    Returns:
        Model response as string
    """
    # Determine configuration
    if config is None:
        if llm_provider is None:
            config = get_default_config()
        else:
            try:
                config = get_config_by_name(llm_provider)
            except ValueError:
                config = LLMConfig(provider=llm_provider, model=model or "default")
    
    # Override model if specified
    if model is not None:
        config.model = model
    
    # Route to appropriate provider
    provider = config.provider
    
    if provider == "aws_bedrock":
        return bedrock.handle_image(
            prompt=prompt,
            image_path=image_path,
            model=config.model,
            **kwargs
        )
    
    elif provider == "ollama":
        # Check if Ollama is available
        if not ollama.OLLAMA_AVAILABLE or not ollama.check_availability():
            print("[WARNING] Ollama not available, falling back to Bedrock")
            return bedrock.handle_image(
                prompt=prompt,
                image_path=image_path,
                model=None,
                **kwargs
            )
        
        # Use vision model for Ollama
        vision_model = model or "llava:7b"
        return ollama.handle_image(
            prompt=prompt,
            image_path=image_path,
            model=vision_model,
            **kwargs
        )
    
    elif provider == "gemini":
        return gemini.handle_image(
            prompt=prompt,
            image_path=image_path,
            model_id=config.model,
            **kwargs
        )
    
    else:
        raise ValueError(f"Provider {provider} does not support image processing")


# Convenience functions
def use_local_llm():
    """Switch to local Ollama provider for privacy"""
    global DEFAULT_LLM_PROVIDER
    DEFAULT_LLM_PROVIDER = "ollama"
    print("[INFO] Switched to local Ollama provider")


def use_cloud_llm():
    """Switch to cloud provider (Bedrock)"""
    global DEFAULT_LLM_PROVIDER
    DEFAULT_LLM_PROVIDER = "aws_bedrock"
    print("[INFO] Switched to cloud Bedrock provider")
