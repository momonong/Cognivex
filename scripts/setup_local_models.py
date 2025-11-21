"""
Setup Local Models for Ollama

This script helps configure Ollama to use locally downloaded models
from HuggingFace (stored in D:\hf_models).

Models:
- GPT-OSS-20B: Agent A (Orchestrator)
- MedGemma-27B: Agent B (Consultant)
"""

import os
import subprocess
from pathlib import Path


def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True
        )
        print(f"[OK] Ollama installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("[ERROR] Ollama not installed")
        print("Install from: https://ollama.ai")
        return False


def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("[OK] Ollama server is running")
            return True
        else:
            print("[WARNING] Ollama server not responding correctly")
            return False
    except Exception as e:
        print(f"[WARNING] Ollama server not running: {e}")
        print("Start with: ollama serve")
        return False


def list_available_models():
    """List models available in Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        print("\n[Available Models in Ollama]")
        print(result.stdout)
        return result.stdout
    except Exception as e:
        print(f"[ERROR] Failed to list models: {e}")
        return ""


def create_modelfile(model_name, model_path, template=None):
    """
    Create Ollama Modelfile for local model
    
    Args:
        model_name: Name for the model in Ollama (e.g., "gpt-oss-20b")
        model_path: Path to the model files
        template: Optional template for the model
    """
    modelfile_content = f"""# Modelfile for {model_name}
FROM {model_path}

# Set parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
    
    if template:
        modelfile_content += f"\nTEMPLATE {template}\n"
    
    modelfile_path = Path(f"Modelfile.{model_name}")
    modelfile_path.write_text(modelfile_content)
    
    print(f"[OK] Created Modelfile: {modelfile_path}")
    return modelfile_path


def import_model_to_ollama(model_name, modelfile_path):
    """Import model to Ollama using Modelfile"""
    try:
        print(f"\n[INFO] Importing {model_name} to Ollama...")
        print(f"[INFO] This may take a few minutes...")
        
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"[OK] Successfully imported {model_name}")
            print(result.stdout)
            return True
        else:
            print(f"[ERROR] Failed to import {model_name}")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False


def setup_gpt_oss_20b():
    """Setup GPT-OSS-20B for Agent A"""
    print("\n" + "="*80)
    print("Setting up GPT-OSS-20B (Agent A - Orchestrator)")
    print("="*80)
    
    model_path = Path("D:/hf_models/gpt-oss-20b")
    
    if not model_path.exists():
        print(f"[ERROR] Model not found at: {model_path}")
        return False
    
    print(f"[OK] Found model at: {model_path}")
    
    # Check for GGUF file
    gguf_files = list(model_path.glob("*.gguf"))
    if not gguf_files:
        print("[ERROR] No GGUF file found in model directory")
        print("Please ensure the model is in GGUF format for Ollama")
        return False
    
    gguf_file = gguf_files[0]
    print(f"[OK] Found GGUF file: {gguf_file.name}")
    
    # Create Modelfile
    modelfile = create_modelfile(
        "gpt-oss-20b",
        str(gguf_file),
        template=None  # Will use default template
    )
    
    # Import to Ollama
    success = import_model_to_ollama("gpt-oss-20b", modelfile)
    
    if success:
        print("\n[OK] GPT-OSS-20B is ready to use!")
        print("Test with: ollama run gpt-oss-20b")
    
    return success


def setup_medgemma_27b():
    """Setup MedGemma-27B for Agent B"""
    print("\n" + "="*80)
    print("Setting up MedGemma-27B (Agent B - Consultant)")
    print("="*80)
    
    model_path = Path("D:/hf_models/medgemma-27b-text-it")
    
    if not model_path.exists():
        print(f"[ERROR] Model not found at: {model_path}")
        return False
    
    print(f"[OK] Found model at: {model_path}")
    
    # Check for GGUF file
    gguf_files = list(model_path.glob("*.gguf"))
    if not gguf_files:
        print("[ERROR] No GGUF file found in model directory")
        print("Please ensure the model is in GGUF format for Ollama")
        return False
    
    gguf_file = gguf_files[0]
    print(f"[OK] Found GGUF file: {gguf_file.name}")
    
    # Create Modelfile
    modelfile = create_modelfile(
        "medgemma-27b",
        str(gguf_file),
        template=None  # Will use default template
    )
    
    # Import to Ollama
    success = import_model_to_ollama("medgemma-27b", modelfile)
    
    if success:
        print("\n[OK] MedGemma-27B is ready to use!")
        print("Test with: ollama run medgemma-27b")
    
    return success


def main():
    """Main setup function"""
    print("\n" + "="*80)
    print("CDDA Phase 4 - Local Model Setup")
    print("="*80)
    
    # Check prerequisites
    if not check_ollama_installed():
        return
    
    check_ollama_running()
    
    # List current models
    list_available_models()
    
    # Setup models
    print("\n[INFO] Starting model setup...")
    print("[INFO] This will import models from D:/hf_models to Ollama")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Setup GPT-OSS-20B
    gpt_success = setup_gpt_oss_20b()
    
    # Setup MedGemma-27B
    med_success = setup_medgemma_27b()
    
    # Summary
    print("\n" + "="*80)
    print("Setup Summary")
    print("="*80)
    print(f"GPT-OSS-20B (Agent A): {'✓ Success' if gpt_success else '✗ Failed'}")
    print(f"MedGemma-27B (Agent B): {'✓ Success' if med_success else '✗ Failed'}")
    
    if gpt_success and med_success:
        print("\n[OK] All models ready!")
        print("\nNext steps:")
        print("1. Test Agent A: python app/agents/agent_a_orchestrator.py")
        print("2. Test Agent B: (coming soon)")
        print("3. Run full system: (coming soon)")
    else:
        print("\n[WARNING] Some models failed to import")
        print("Please check the error messages above")
    
    # List models again
    print("\n")
    list_available_models()


if __name__ == "__main__":
    main()
