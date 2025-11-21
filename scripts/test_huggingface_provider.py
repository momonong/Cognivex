"""
Test HuggingFace Provider

Quick test to verify HuggingFace models can be loaded and used.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.llm_providers import huggingface


def test_model_info():
    """Test getting model information"""
    print("\n" + "="*80)
    print("Test 1: Get Model Information")
    print("="*80)
    
    models = {
        "GPT-OSS-20B": "D:/hf_models/gpt-oss-20b",
        "MedGemma-27B": "D:/hf_models/medgemma-27b-text-it"
    }
    
    for name, path in models.items():
        print(f"\n[{name}]")
        info = huggingface.get_model_info(path)
        print(f"  Path: {info['path']}")
        print(f"  Exists: {info['exists']}")
        if info['exists']:
            print(f"  SafeTensors files: {info['safetensors_count']}")
            if 'config' in info:
                print(f"  Model type: {info['config'].get('model_type', 'Unknown')}")
                print(f"  Architecture: {info['config'].get('architectures', ['Unknown'])[0]}")


def test_load_model():
    """Test loading a model"""
    print("\n" + "="*80)
    print("Test 2: Load Model")
    print("="*80)
    
    model_path = "D:/hf_models/gpt-oss-20b"
    
    print(f"\n[INFO] Attempting to load: {model_path}")
    print("[INFO] Using 8-bit quantization to save memory")
    print("[INFO] This may take a few minutes on first load...")
    
    try:
        model, tokenizer = huggingface.load_model(
            model_path=model_path,
            device="auto",
            torch_dtype="auto",
            load_in_8bit=True  # Use 8-bit to save memory
        )
        
        print("\n[OK] Model loaded successfully!")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Tokenizer type: {type(tokenizer).__name__}")
        print(f"  Vocab size: {len(tokenizer)}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        print("\nPossible solutions:")
        print("1. Install required packages:")
        print("   pip install transformers torch accelerate bitsandbytes")
        print("2. Ensure model files are complete")
        print("3. Check available memory (model requires ~16GB RAM)")
        return False


def test_generate_simple():
    """Test simple text generation"""
    print("\n" + "="*80)
    print("Test 3: Simple Text Generation")
    print("="*80)
    
    model_path = "D:/hf_models/gpt-oss-20b"
    
    print(f"\n[INFO] Generating text with: {model_path}")
    
    try:
        response = huggingface.handle_text(
            prompt="Hello, how are you?",
            model_path=model_path,
            temperature=0.7,
            max_new_tokens=50,
            load_in_8bit=True
        )
        
        print("\n[Response]")
        print(response)
        print("\n[OK] Generation successful!")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        return False


def test_json_generation():
    """Test JSON generation (for Agent A)"""
    print("\n" + "="*80)
    print("Test 4: JSON Generation (Agent A Style)")
    print("="*80)
    
    model_path = "D:/hf_models/gpt-oss-20b"
    
    system_prompt = """You are Agent A, an orchestrator that decides which actions to take.
Respond with valid JSON only."""
    
    user_prompt = """Based on this diagnostic data, decide which actions to take:
- Subject: sub-0005
- Prediction: AD
- Confidence: 0.72
- UQ Score: 0.74
- Has Anomaly: True

Respond with JSON containing 'actions' and 'decision_rationale'."""
    
    print(f"\n[INFO] Testing JSON generation...")
    
    try:
        response = huggingface.handle_text(
            prompt=user_prompt,
            model_path=model_path,
            system_instruction=system_prompt,
            temperature=0.1,
            max_new_tokens=256,
            load_in_8bit=True
        )
        
        print("\n[Response]")
        print(response)
        
        # Try to parse as JSON
        import json
        try:
            parsed = json.loads(response)
            print("\n[OK] Valid JSON generated!")
            print(f"  Keys: {list(parsed.keys())}")
            return True
        except json.JSONDecodeError:
            print("\n[WARNING] Response is not valid JSON")
            print("  This is normal - may need prompt tuning")
            return False
        
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("HuggingFace Provider Test Suite")
    print("="*80)
    
    # Check if transformers is available
    if not huggingface.check_availability():
        print("\n[ERROR] transformers not installed")
        print("\nInstall with:")
        print("  pip install transformers torch accelerate bitsandbytes")
        return
    
    print("\n[OK] transformers is available")
    
    # Run tests
    results = {}
    
    # Test 1: Model info
    test_model_info()
    
    # Test 2: Load model
    print("\n" + "="*80)
    print("Ready to load model?")
    print("="*80)
    print("\nThis will:")
    print("- Load GPT-OSS-20B (~13GB with 8-bit quantization)")
    print("- Require ~16GB RAM")
    print("- Take 2-5 minutes on first load")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("\n[INFO] Skipping model loading tests")
        return
    
    results['load'] = test_load_model()
    
    if results['load']:
        # Test 3: Simple generation
        results['simple'] = test_generate_simple()
        
        # Test 4: JSON generation
        results['json'] = test_json_generation()
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    if all(results.values()):
        print("\n[OK] All tests passed!")
        print("\nNext steps:")
        print("1. Test Agent A with HuggingFace:")
        print("   python app/agents/agent_a_orchestrator.py")
    else:
        print("\n[WARNING] Some tests failed")
        print("Check error messages above for details")


if __name__ == "__main__":
    main()
