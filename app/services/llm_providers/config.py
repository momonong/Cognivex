"""
LLM Provider Configuration

Centralized configuration for LLM providers with support for:
- AWS Bedrock (cloud)
- Ollama (local)
- Gemini (cloud)
"""

import os
from typing import Literal, Optional
from dataclasses import dataclass
from pathlib import Path


# LLM Provider Types
LLMProvider = Literal["aws_bedrock", "ollama", "gemini"]


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider
    model: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    system_instruction: Optional[str] = None
    
    # Ollama specific
    ollama_base_url: str = "http://localhost:11434"
    
    # AWS Bedrock specific
    aws_region: str = "us-east-1"
    
    @property
    def is_local(self) -> bool:
        """Check if provider is local (privacy-preserving)"""
        return self.provider == "ollama"
    
    @property
    def is_cloud(self) -> bool:
        """Check if provider is cloud-based"""
        return self.provider in ["aws_bedrock", "gemini"]


# ============================================================================
# Predefined Configurations
# ============================================================================

# AWS Bedrock (Cloud) - Current default
BEDROCK_CONFIG = LLMConfig(
    provider="aws_bedrock",
    model="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0.1,
    aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

# Ollama (Local) - Privacy-preserving
OLLAMA_CONFIG = LLMConfig(
    provider="ollama",
    model="llama3.2:3b",  # 輕量級模型，適合本地運行
    temperature=0.1,
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

# Ollama with larger model for better quality
OLLAMA_LARGE_CONFIG = LLMConfig(
    provider="ollama",
    model="llama3.1:8b",  # 更大的模型，更好的質量
    temperature=0.1,
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

# Ollama with medical-specific model (if available)
OLLAMA_MEDICAL_CONFIG = LLMConfig(
    provider="ollama",
    model="meditron:7b",  # 醫療專用模型
    temperature=0.1,
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)


# ============================================================================
# Configuration Selection
# ============================================================================

# Default provider (can be overridden by environment variable)
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Privacy mode (force local models)
PRIVACY_MODE = os.getenv("PRIVACY_MODE", "true").lower() == "false"


def get_default_config() -> LLMConfig:
    """
    Get default LLM configuration
    
    Priority:
    1. PRIVACY_MODE=true -> Use Ollama
    2. LLM_PROVIDER env var
    3. Default to AWS Bedrock
    """
    if PRIVACY_MODE:
        print("[INFO] Privacy mode enabled - using local Ollama")
        return OLLAMA_CONFIG
    
    if DEFAULT_PROVIDER == "ollama":
        return OLLAMA_CONFIG
    elif DEFAULT_PROVIDER == "gemini":
        from app.services.llm_providers.gemini import GEMINI_AVAILABLE
        if not GEMINI_AVAILABLE:
            print("[WARNING] Gemini not available, falling back to Bedrock")
            return BEDROCK_CONFIG
        return LLMConfig(provider="gemini", model="gemini-pro")
    else:
        return BEDROCK_CONFIG


def get_config_by_name(name: str) -> LLMConfig:
    """
    Get LLM configuration by name
    
    Args:
        name: Configuration name
            - 'bedrock' or 'aws_bedrock'
            - 'ollama' or 'ollama_small'
            - 'ollama_large'
            - 'ollama_medical'
            - 'gemini'
    
    Returns:
        LLMConfig object
    """
    name = name.lower()
    
    if name in ["bedrock", "aws_bedrock"]:
        return BEDROCK_CONFIG
    elif name in ["ollama", "ollama_small"]:
        return OLLAMA_CONFIG
    elif name == "ollama_large":
        return OLLAMA_LARGE_CONFIG
    elif name == "ollama_medical":
        return OLLAMA_MEDICAL_CONFIG
    elif name == "gemini":
        return LLMConfig(provider="gemini", model="gemini-pro")
    else:
        raise ValueError(f"Unknown configuration: {name}")


def check_ollama_availability() -> bool:
    """Check if Ollama is available and running"""
    try:
        import requests
        config = OLLAMA_CONFIG
        response = requests.get(f"{config.ollama_base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def list_ollama_models() -> list:
    """List available Ollama models"""
    try:
        import requests
        config = OLLAMA_CONFIG
        response = requests.get(f"{config.ollama_base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except Exception:
        return []


def print_config_info(config: LLMConfig = None):
    """Print LLM configuration information"""
    if config is None:
        config = get_default_config()
    
    print("="*80)
    print("LLM Provider Configuration")
    print("="*80)
    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"Temperature: {config.temperature}")
    print(f"Privacy Mode: {'✓ Local' if config.is_local else '✗ Cloud'}")
    
    if config.provider == "ollama":
        print(f"Ollama URL: {config.ollama_base_url}")
        print(f"Ollama Available: {check_ollama_availability()}")
        
        models = list_ollama_models()
        if models:
            print(f"Available Models: {', '.join(models)}")
    
    elif config.provider == "aws_bedrock":
        print(f"AWS Region: {config.aws_region}")
    
    print("="*80)


# ============================================================================
# Recommended Models for Medical Applications
# ============================================================================

RECOMMENDED_MODELS = {
    "ollama": {
        "small": {
            "model": "llama3.2:3b",
            "description": "輕量級，適合快速推理",
            "ram_required": "4GB",
            "speed": "快"
        },
        "medium": {
            "model": "llama3.1:8b",
            "description": "平衡性能和質量",
            "ram_required": "8GB",
            "speed": "中"
        },
        "large": {
            "model": "llama3.1:70b",
            "description": "最佳質量，需要強大硬件",
            "ram_required": "64GB",
            "speed": "慢"
        },
        "medical": {
            "model": "meditron:7b",
            "description": "醫療專用模型",
            "ram_required": "8GB",
            "speed": "中"
        }
    },
    "aws_bedrock": {
        "fast": {
            "model": "anthropic.claude-3-haiku-20240307-v1:0",
            "description": "快速且經濟",
            "cost": "低"
        },
        "balanced": {
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "description": "平衡性能和成本",
            "cost": "中"
        },
        "best": {
            "model": "anthropic.claude-3-opus-20240229-v1:0",
            "description": "最佳質量",
            "cost": "高"
        }
    }
}


def print_recommended_models():
    """Print recommended models for medical applications"""
    print("\n" + "="*80)
    print("Recommended Models for Medical Applications")
    print("="*80)
    
    print("\n[LOCAL - Ollama] Privacy-Preserving")
    print("-"*80)
    for name, info in RECOMMENDED_MODELS["ollama"].items():
        print(f"\n{name.upper()}:")
        print(f"  Model: {info['model']}")
        print(f"  Description: {info['description']}")
        print(f"  RAM Required: {info['ram_required']}")
        print(f"  Speed: {info['speed']}")
    
    print("\n[CLOUD - AWS Bedrock]")
    print("-"*80)
    for name, info in RECOMMENDED_MODELS["aws_bedrock"].items():
        print(f"\n{name.upper()}:")
        print(f"  Model: {info['model']}")
        print(f"  Description: {info['description']}")
        print(f"  Cost: {info['cost']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Print current configuration
    print_config_info()
    
    # Print recommended models
    print_recommended_models()
    
    # Check Ollama availability
    if check_ollama_availability():
        print("\n[OK] Ollama is running and available")
        models = list_ollama_models()
        if models:
            print(f"[OK] Found {len(models)} models: {', '.join(models)}")
    else:
        print("\n[WARNING] Ollama is not available")
        print("To install Ollama:")
        print("  1. Visit: https://ollama.ai")
        print("  2. Download and install")
        print("  3. Run: ollama pull llama3.2:3b")
