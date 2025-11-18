"""
Test Ollama Integration

Quick test script to verify Ollama integration works correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_ollama_availability():
    """Test if Ollama is available"""
    print("\n" + "="*80)
    print("Test 1: Ollama Availability")
    print("="*80)
    
    from app.services.llm_providers import ollama
    
    print(f"Ollama package installed: {ollama.OLLAMA_AVAILABLE}")
    
    if not ollama.OLLAMA_AVAILABLE:
        print("[FAIL] Ollama package not installed")
        print("Install with: pip install ollama")
        return False
    
    print(f"Ollama server running: {ollama.check_availability()}")
    
    if not ollama.check_availability():
        print("[FAIL] Ollama server not running")
        print("Start with: ollama serve")
        return False
    
    print("[PASS] Ollama is available and running")
    return True


def test_list_models():
    """Test listing available models"""
    print("\n" + "="*80)
    print("Test 2: List Available Models")
    print("="*80)
    
    from app.services.llm_providers import ollama
    
    models = ollama.list_models()
    
    if not models:
        print("[WARN] No models installed")
        print("Install a model with: ollama pull llama3.2:3b")
        return False
    
    print(f"[PASS] Found {len(models)} models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    return True


def test_simple_inference():
    """Test simple text inference"""
    print("\n" + "="*80)
    print("Test 3: Simple Text Inference")
    print("="*80)
    
    from app.services.llm_providers import llm_response
    
    test_prompt = "What is 2+2? Answer with just the number."
    
    print(f"Prompt: {test_prompt}")
    print("Generating response...")
    
    try:
        response = llm_response(
            test_prompt,
            llm_provider="ollama"
        )
        
        print(f"\nResponse: {response}")
        print("\n[PASS] Simple inference works")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Inference failed: {e}")
        return False


def test_medical_inference():
    """Test medical domain inference"""
    print("\n" + "="*80)
    print("Test 4: Medical Domain Inference")
    print("="*80)
    
    from app.services.llm_providers import llm_response
    
    test_prompt = """What is Alzheimer's disease? 
Provide a brief medical explanation in 2-3 sentences."""
    
    print(f"Prompt: {test_prompt}")
    print("Generating response...")
    
    try:
        response = llm_response(
            test_prompt,
            llm_provider="ollama"
        )
        
        print(f"\nResponse:")
        print("-"*80)
        print(response)
        print("-"*80)
        
        print("\n[PASS] Medical inference works")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Medical inference failed: {e}")
        return False


def test_privacy_mode():
    """Test privacy mode"""
    print("\n" + "="*80)
    print("Test 5: Privacy Mode")
    print("="*80)
    
    # Enable privacy mode
    os.environ['PRIVACY_MODE'] = 'true'
    
    from app.services.llm_providers import llm_response
    from app.services.llm_providers.config import get_default_config
    
    config = get_default_config()
    
    print(f"Privacy mode enabled: {os.environ.get('PRIVACY_MODE')}")
    print(f"Default provider: {config.provider}")
    print(f"Is local: {config.is_local}")
    
    if config.provider != "ollama":
        print("[FAIL] Privacy mode not working - not using Ollama")
        return False
    
    print("\nTesting inference with privacy mode...")
    
    try:
        response = llm_response("Hello, how are you?")
        print(f"Response: {response[:100]}...")
        print("\n[PASS] Privacy mode works")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Privacy mode inference failed: {e}")
        return False


def test_provider_switching():
    """Test switching between providers"""
    print("\n" + "="*80)
    print("Test 6: Provider Switching")
    print("="*80)
    
    from app.services.llm_providers import use_local_llm, use_cloud_llm
    from app.services.llm_providers import DEFAULT_LLM_PROVIDER
    
    print(f"Initial provider: {DEFAULT_LLM_PROVIDER}")
    
    # Switch to local
    use_local_llm()
    from app.services.llm_providers import DEFAULT_LLM_PROVIDER as provider_after_local
    print(f"After use_local_llm(): {provider_after_local}")
    
    # Switch to cloud
    use_cloud_llm()
    from app.services.llm_providers import DEFAULT_LLM_PROVIDER as provider_after_cloud
    print(f"After use_cloud_llm(): {provider_after_cloud}")
    
    print("\n[PASS] Provider switching works")
    return True


def test_config_system():
    """Test configuration system"""
    print("\n" + "="*80)
    print("Test 7: Configuration System")
    print("="*80)
    
    from app.services.llm_providers.config import (
        get_config_by_name,
        OLLAMA_CONFIG,
        BEDROCK_CONFIG
    )
    
    # Test predefined configs
    configs_to_test = [
        "ollama",
        "ollama_large",
        "ollama_medical",
        "bedrock"
    ]
    
    for config_name in configs_to_test:
        try:
            config = get_config_by_name(config_name)
            print(f"[OK] {config_name}: provider={config.provider}, model={config.model}")
        except Exception as e:
            print(f"[FAIL] {config_name}: {e}")
            return False
    
    print("\n[PASS] Configuration system works")
    return True


def run_all_tests():
    """Run all tests"""
    print("="*80)
    print("Ollama Integration Test Suite")
    print("="*80)
    
    tests = [
        ("Ollama Availability", test_ollama_availability),
        ("List Models", test_list_models),
        ("Simple Inference", test_simple_inference),
        ("Medical Inference", test_medical_inference),
        ("Privacy Mode", test_privacy_mode),
        ("Provider Switching", test_provider_switching),
        ("Configuration System", test_config_system),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Ollama Integration")
    parser.add_argument(
        '--test',
        choices=['all', 'availability', 'models', 'simple', 'medical', 'privacy', 'switching', 'config'],
        default='all',
        help='Which test to run'
    )
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_tests()
        sys.exit(0 if success else 1)
    elif args.test == 'availability':
        test_ollama_availability()
    elif args.test == 'models':
        test_list_models()
    elif args.test == 'simple':
        test_simple_inference()
    elif args.test == 'medical':
        test_medical_inference()
    elif args.test == 'privacy':
        test_privacy_mode()
    elif args.test == 'switching':
        test_provider_switching()
    elif args.test == 'config':
        test_config_system()
