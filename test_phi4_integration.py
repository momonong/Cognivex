#!/usr/bin/env python3
"""
Test Script: Phi-4-mini + MedGemma-27b Integration

This script verifies that the CDDA system is correctly configured to use:
- Phi-4-mini-instruct as Orchestrator Agent
- MedGemma-27b as Consultant Agent
- 4-bit quantization for VRAM efficiency
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*80)
    print("TEST 1: Verify Imports")
    print("="*80)
    
    try:
        from app.agents.llm_factory import LLMFactory
        print("✓ LLMFactory imported successfully")
        
        from app.agents.cdda_agent import CDDAAgent
        print("✓ CDDAAgent imported successfully")
        
        from app.agents.agent_a_orchestrator import AgentA, AgentAConfig
        print("✓ AgentA imported successfully")
        
        from app.agents.agent_b_consultant import AgentB, AgentBConfig
        print("✓ AgentB imported successfully")
        
        print("\n[PASS] All imports successful")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_factory():
    """Test 2: Verify LLM Factory configuration"""
    print("\n" + "="*80)
    print("TEST 2: Verify LLM Factory Configuration")
    print("="*80)
    
    try:
        from app.agents.llm_factory import LLMFactory
        
        # Check if transformers is available
        try:
            import transformers
            print(f"✓ Transformers version: {transformers.__version__}")
        except ImportError:
            print("✗ Transformers not installed")
            print("  Install with: pip install transformers torch accelerate bitsandbytes")
            return False
        
        # Check if torch is available
        try:
            import torch
            print(f"✓ PyTorch version: {torch.__version__}")
            print(f"✓ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"✓ CUDA version: {torch.version.cuda}")
                print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✓ Total VRAM: {total_vram:.2f}GB")
        except ImportError:
            print("✗ PyTorch not installed")
            return False
        
        # Check 4-bit config
        try:
            config = LLMFactory.get_4bit_config()
            print(f"✓ 4-bit quantization config created")
            print(f"  - Compute dtype: {config.bnb_4bit_compute_dtype}")
            print(f"  - Quant type: {config.bnb_4bit_quant_type}")
            print(f"  - Double quant: {config.bnb_4bit_use_double_quant}")
        except Exception as e:
            print(f"✗ Failed to create 4-bit config: {e}")
            return False
        
        print("\n[PASS] LLM Factory configuration verified")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] LLM Factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_paths():
    """Test 3: Verify model paths exist"""
    print("\n" + "="*80)
    print("TEST 3: Verify Model Paths")
    print("="*80)
    
    phi4_path = Path("D:/hf_models/Phi-4-mini-instruct")
    medgemma_path = Path("D:/hf_models/medgemma-27b-text-it")
    
    # Check Phi-4-mini
    if phi4_path.exists():
        print(f"✓ Phi-4-mini path exists: {phi4_path}")
        config_file = phi4_path / "config.json"
        if config_file.exists():
            print(f"  ✓ config.json found")
        else:
            print(f"  ✗ config.json not found")
        
        # Check for model files
        safetensors = list(phi4_path.glob("*.safetensors"))
        if safetensors:
            print(f"  ✓ Found {len(safetensors)} safetensors file(s)")
        else:
            print(f"  ✗ No safetensors files found")
    else:
        print(f"✗ Phi-4-mini path does not exist: {phi4_path}")
        print(f"  Download with: huggingface-cli download microsoft/Phi-4-mini-instruct --local-dir {phi4_path}")
    
    # Check MedGemma
    if medgemma_path.exists():
        print(f"✓ MedGemma path exists: {medgemma_path}")
        config_file = medgemma_path / "config.json"
        if config_file.exists():
            print(f"  ✓ config.json found")
        else:
            print(f"  ✗ config.json not found")
        
        # Check for model files
        safetensors = list(medgemma_path.glob("*.safetensors"))
        if safetensors:
            print(f"  ✓ Found {len(safetensors)} safetensors file(s)")
        else:
            print(f"  ✗ No safetensors files found")
    else:
        print(f"✗ MedGemma path does not exist: {medgemma_path}")
        print(f"  Download with: huggingface-cli download google/medgemma-27b-text-it --local-dir {medgemma_path}")
    
    both_exist = phi4_path.exists() and medgemma_path.exists()
    
    if both_exist:
        print("\n[PASS] Both model paths exist")
    else:
        print("\n[WARN] Some model paths are missing")
        print("       System will fall back to rule-based orchestration")
    
    return both_exist


def test_cdda_agent_config():
    """Test 4: Verify CDDA Agent default configuration"""
    print("\n" + "="*80)
    print("TEST 4: Verify CDDA Agent Configuration")
    print("="*80)
    
    try:
        from app.agents.cdda_agent import CDDAAgent
        
        # Create agent with rule-based fallback (no model loading)
        print("Creating CDDA Agent with rule-based fallback...")
        agent = CDDAAgent(
            use_llm=False,  # Don't load models
            verbose=False
        )
        
        print(f"✓ CDDA Agent created successfully")
        print(f"  - UQ Threshold: {agent.uq_threshold}")
        print(f"  - Z-Score Threshold: {agent.z_score_threshold}")
        print(f"  - Use LLM: {agent.use_llm}")
        
        # Check Agent A config
        print(f"\nAgent A (Orchestrator) Configuration:")
        print(f"  - Model: {agent.agent_a.config.model}")
        print(f"  - Model Path: {agent.agent_a.config.model_path}")
        print(f"  - Provider: {agent.agent_a.config.provider}")
        print(f"  - Temperature: {agent.agent_a.config.temperature}")
        
        # Check Agent B config
        print(f"\nAgent B (Consultant) Configuration:")
        print(f"  - Model: {agent.agent_b.config.model}")
        print(f"  - Model Path: {agent.agent_b.config.model_path}")
        print(f"  - Provider: {agent.agent_b.config.provider}")
        print(f"  - Temperature: {agent.agent_b.config.temperature}")
        
        # Verify default paths
        expected_phi4 = "D:/hf_models/Phi-4-mini-instruct"
        expected_medgemma = "D:/hf_models/medgemma-27b-text-it"
        
        if agent.agent_a.config.model_path == expected_phi4:
            print(f"\n✓ Orchestrator path matches expected: {expected_phi4}")
        else:
            print(f"\n✗ Orchestrator path mismatch:")
            print(f"  Expected: {expected_phi4}")
            print(f"  Got: {agent.agent_a.config.model_path}")
        
        if agent.agent_b.config.model_path == expected_medgemma:
            print(f"✓ Consultant path matches expected: {expected_medgemma}")
        else:
            print(f"✗ Consultant path mismatch:")
            print(f"  Expected: {expected_medgemma}")
            print(f"  Got: {agent.agent_b.config.model_path}")
        
        print("\n[PASS] CDDA Agent configuration verified")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] CDDA Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_file():
    """Test 5: Verify orchestrator prompt is updated"""
    print("\n" + "="*80)
    print("TEST 5: Verify Orchestrator Prompt")
    print("="*80)
    
    try:
        prompt_path = Path("config/prompts/agent_a_orchestrator.txt")
        
        if not prompt_path.exists():
            print(f"✗ Prompt file not found: {prompt_path}")
            return False
        
        print(f"✓ Prompt file exists: {prompt_path}")
        
        # Read prompt
        prompt_text = prompt_path.read_text(encoding='utf-8')
        
        # Check for key phrases
        checks = [
            ("precise Orchestrator Agent", "Updated role description"),
            ("MUST output valid JSON ONLY", "JSON constraint"),
            ("No markdown, no conversational text", "Markdown prohibition"),
            ("REQUIRED JSON OUTPUT FORMAT", "Schema section"),
            ("EXAMPLE 1", "Example 1 present"),
            ("EXAMPLE 2", "Example 2 present"),
            ("EXAMPLE 3", "Example 3 present"),
            ("START YOUR RESPONSE WITH {", "Explicit start instruction")
        ]
        
        all_passed = True
        for phrase, description in checks:
            if phrase in prompt_text:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description} - phrase not found: '{phrase}'")
                all_passed = False
        
        if all_passed:
            print("\n[PASS] Orchestrator prompt is properly updated for Phi-4")
        else:
            print("\n[WARN] Some prompt optimizations may be missing")
        
        return all_passed
        
    except Exception as e:
        print(f"\n[FAIL] Prompt verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("PHI-4-MINI + MEDGEMMA-27B INTEGRATION TEST")
    print("="*80)
    print("\nThis script verifies the CDDA system is correctly configured")
    print("to use Phi-4-mini (Orchestrator) and MedGemma-27b (Consultant)")
    print("with 4-bit quantization for VRAM efficiency.")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("LLM Factory", test_llm_factory()))
    results.append(("Model Paths", test_model_paths()))
    results.append(("CDDA Agent Config", test_cdda_agent_config()))
    results.append(("Orchestrator Prompt", test_prompt_file()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n[SUCCESS] All tests passed! System is ready.")
        print("\nNext steps:")
        print("1. Run: python scripts/demo_phase4_complete.py")
        print("2. Monitor VRAM usage during execution")
        print("3. Verify JSON output from Orchestrator")
    else:
        print("\n[WARNING] Some tests failed. Review the output above.")
        print("\nCommon issues:")
        print("- Models not downloaded: Use huggingface-cli to download")
        print("- Missing dependencies: pip install transformers torch accelerate bitsandbytes")
        print("- CUDA not available: Check GPU drivers and PyTorch installation")
    
    print("\n" + "="*80 + "\n")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
