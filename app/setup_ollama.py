"""
Ollama Setup and Testing Script

This script helps you:
1. Check if Ollama is installed and running
2. Install recommended models
3. Test Ollama integration
4. Switch between cloud and local providers
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_ollama_installation():
    """Check if Ollama is installed"""
    print("\n" + "="*80)
    print("Step 1: Checking Ollama Installation")
    print("="*80)
    
    from app.services.llm_providers import ollama
    
    if not ollama.OLLAMA_AVAILABLE:
        print("[ERROR] Ollama Python package not installed")
        print("\nTo install:")
        print("  pip install ollama")
        return False
    
    print("[OK] Ollama Python package installed")
    
    if not ollama.check_availability():
        print("[WARNING] Ollama server is not running")
        print("\nTo start Ollama:")
        print("  1. Install Ollama from: https://ollama.ai")
        print("  2. Start the server:")
        print("     - Windows/Mac: Ollama runs automatically after installation")
        print("     - Linux: Run 'ollama serve' in terminal")
        return False
    
    print("[OK] Ollama server is running")
    return True


def list_available_models():
    """List currently available models"""
    print("\n" + "="*80)
    print("Step 2: Available Models")
    print("="*80)
    
    from app.services.llm_providers import ollama
    
    models = ollama.list_models()
    
    if not models:
        print("[INFO] No models installed yet")
        return []
    
    print(f"[OK] Found {len(models)} installed models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    return models


def install_recommended_models():
    """Install recommended models for medical applications"""
    print("\n" + "="*80)
    print("Step 3: Install Recommended Models")
    print("="*80)
    
    from app.services.llm_providers import ollama
    from app.services.llm_providers.config import RECOMMENDED_MODELS
    
    print("\nRecommended models for medical applications:")
    print("\n1. llama3.2:3b (Small, Fast)")
    print("   - RAM: 4GB")
    print("   - Speed: Fast")
    print("   - Best for: Quick inference, testing")
    
    print("\n2. llama3.1:8b (Medium, Balanced)")
    print("   - RAM: 8GB")
    print("   - Speed: Medium")
    print("   - Best for: Production use, good quality")
    
    print("\n3. meditron:7b (Medical Specialist)")
    print("   - RAM: 8GB")
    print("   - Speed: Medium")
    print("   - Best for: Medical text generation")
    
    print("\n4. llava:7b (Vision Model)")
    print("   - RAM: 8GB")
    print("   - Speed: Medium")
    print("   - Best for: Image analysis")
    
    choice = input("\nWhich model would you like to install? (1-4, or 'all', or 'skip'): ").strip().lower()
    
    models_to_install = []
    if choice == '1':
        models_to_install = ['llama3.2:3b']
    elif choice == '2':
        models_to_install = ['llama3.1:8b']
    elif choice == '3':
        models_to_install = ['meditron:7b']
    elif choice == '4':
        models_to_install = ['llava:7b']
    elif choice == 'all':
        models_to_install = ['llama3.2:3b', 'llama3.1:8b', 'meditron:7b', 'llava:7b']
    elif choice == 'skip':
        print("[INFO] Skipping model installation")
        return
    else:
        print("[INFO] Invalid choice, skipping")
        return
    
    for model in models_to_install:
        print(f"\n[INFO] Installing {model}...")
        success = ollama.pull_model(model)
        if success:
            print(f"[OK] {model} installed successfully")
        else:
            print(f"[ERROR] Failed to install {model}")


def test_ollama_inference():
    """Test Ollama inference"""
    print("\n" + "="*80)
    print("Step 4: Test Ollama Inference")
    print("="*80)
    
    from app.services.llm_providers import llm_response
    
    # Check available models
    from app.services.llm_providers import ollama
    models = ollama.list_models()
    
    if not models:
        print("[ERROR] No models available for testing")
        print("Please install a model first")
        return
    
    # Use first available model
    test_model = models[0]
    print(f"\n[INFO] Testing with model: {test_model}")
    
    # Test prompt
    test_prompt = "What is Alzheimer's disease? Please provide a brief medical explanation."
    
    print(f"\n[TEST] Prompt: {test_prompt}")
    print("\n[INFO] Generating response...")
    
    try:
        response = llm_response(
            test_prompt,
            llm_provider="ollama",
            model=test_model
        )
        
        print("\n[RESPONSE]")
        print("-"*80)
        print(response)
        print("-"*80)
        
        print("\n[OK] Ollama inference test passed!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def configure_privacy_mode():
    """Configure privacy mode"""
    print("\n" + "="*80)
    print("Step 5: Configure Privacy Mode")
    print("="*80)
    
    print("\nPrivacy Mode Options:")
    print("1. Enable Privacy Mode (Use local Ollama only)")
    print("2. Disable Privacy Mode (Use cloud providers)")
    print("3. Skip")
    
    choice = input("\nYour choice (1-3): ").strip()
    
    if choice == '1':
        os.environ['PRIVACY_MODE'] = 'true'
        print("\n[OK] Privacy Mode ENABLED")
        print("All LLM requests will use local Ollama")
        print("\nTo make this permanent, add to your environment:")
        print("  export PRIVACY_MODE=true  # Linux/Mac")
        print("  set PRIVACY_MODE=true     # Windows")
        
    elif choice == '2':
        os.environ['PRIVACY_MODE'] = 'false'
        print("\n[OK] Privacy Mode DISABLED")
        print("LLM requests will use cloud providers (AWS Bedrock)")
        
    else:
        print("\n[INFO] Skipping privacy mode configuration")


def show_usage_examples():
    """Show usage examples"""
    print("\n" + "="*80)
    print("Usage Examples")
    print("="*80)
    
    print("""
# Example 1: Use default provider (respects PRIVACY_MODE)
from app.services.llm_providers import llm_response

response = llm_response("What is Alzheimer's disease?")

# Example 2: Explicitly use Ollama
response = llm_response(
    "What is Alzheimer's disease?",
    llm_provider="ollama",
    model="llama3.2:3b"
)

# Example 3: Use configuration object
from app.services.llm_providers.config import get_config_by_name

config = get_config_by_name("ollama_large")
response = llm_response("What is Alzheimer's disease?", config=config)

# Example 4: Enable privacy mode programmatically
import os
os.environ['PRIVACY_MODE'] = 'true'
response = llm_response("What is Alzheimer's disease?")  # Will use Ollama

# Example 5: Switch providers at runtime
from app.services.llm_providers import use_local_llm, use_cloud_llm

use_local_llm()  # Switch to Ollama
response = llm_response("What is Alzheimer's disease?")

use_cloud_llm()  # Switch back to Bedrock
response = llm_response("What is Alzheimer's disease?")
""")


def main():
    """Main setup workflow"""
    print("="*80)
    print("Ollama Setup and Testing")
    print("="*80)
    print("\nThis script will help you set up Ollama for local, privacy-preserving LLM inference.")
    
    # Step 1: Check installation
    if not check_ollama_installation():
        print("\n[INFO] Please install and start Ollama, then run this script again")
        return
    
    # Step 2: List models
    models = list_available_models()
    
    # Step 3: Install models (if needed)
    if not models:
        install_recommended_models()
    else:
        choice = input("\nWould you like to install additional models? (y/n): ").strip().lower()
        if choice == 'y':
            install_recommended_models()
    
    # Step 4: Test inference
    test_ollama_inference()
    
    # Step 5: Configure privacy mode
    configure_privacy_mode()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "="*80)
    print("[SUCCESS] Ollama setup complete!")
    print("="*80)
    print("\nYou can now use Ollama in your application.")
    print("See the usage examples above for how to use it in your code.")


if __name__ == "__main__":
    main()
