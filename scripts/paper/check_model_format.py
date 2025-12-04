"""
Check Model Format

Quick script to check if downloaded models are in the correct format for Ollama.
"""

from pathlib import Path
import os


def check_directory(path):
    """Check what files are in a directory"""
    path = Path(path)
    
    if not path.exists():
        print(f"[ERROR] Directory not found: {path}")
        return None
    
    print(f"\n[Checking] {path}")
    print("-" * 80)
    
    # List all files
    files = list(path.glob("*"))
    
    if not files:
        print("[WARNING] Directory is empty")
        return None
    
    # Check for GGUF files
    gguf_files = list(path.glob("*.gguf"))
    
    # Check for PyTorch files
    pytorch_files = list(path.glob("*.bin")) + list(path.glob("*.pt"))
    
    # Check for SafeTensors files
    safetensors_files = list(path.glob("*.safetensors"))
    
    # Check for config files
    config_files = list(path.glob("config.json")) + list(path.glob("*.yaml"))
    
    print(f"Total files: {len(files)}")
    print(f"\nFile types found:")
    print(f"  GGUF files: {len(gguf_files)}")
    print(f"  PyTorch files: {len(pytorch_files)}")
    print(f"  SafeTensors files: {len(safetensors_files)}")
    print(f"  Config files: {len(config_files)}")
    
    # Show GGUF files if found
    if gguf_files:
        print(f"\n✓ GGUF files found (ready for Ollama):")
        for f in gguf_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    - {f.name} ({size_mb:.1f} MB)")
        return "gguf"
    
    # Show PyTorch files if found
    elif pytorch_files:
        print(f"\n⚠ PyTorch files found (need conversion):")
        for f in pytorch_files[:5]:  # Show first 5
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    - {f.name} ({size_mb:.1f} MB)")
        if len(pytorch_files) > 5:
            print(f"    ... and {len(pytorch_files) - 5} more")
        return "pytorch"
    
    # Show SafeTensors files if found
    elif safetensors_files:
        print(f"\n⚠ SafeTensors files found (need conversion):")
        for f in safetensors_files[:5]:  # Show first 5
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    - {f.name} ({size_mb:.1f} MB)")
        if len(safetensors_files) > 5:
            print(f"    ... and {len(safetensors_files) - 5} more")
        return "safetensors"
    
    else:
        print(f"\n✗ No model files found")
        print(f"Files in directory:")
        for f in files[:10]:
            print(f"    - {f.name}")
        return None


def main():
    print("="*80)
    print("CDDA Phase 4 - Model Format Checker")
    print("="*80)
    
    models_dir = Path("D:/hf_models")
    
    if not models_dir.exists():
        print(f"[ERROR] Models directory not found: {models_dir}")
        return
    
    print(f"\n[OK] Models directory found: {models_dir}")
    
    # Check GPT-OSS-20B
    print("\n" + "="*80)
    print("1. GPT-OSS-20B (Agent A - Orchestrator)")
    print("="*80)
    gpt_format = check_directory(models_dir / "gpt-oss-20b")
    
    # Check MedGemma-27B
    print("\n" + "="*80)
    print("2. MedGemma-27B (Agent B - Consultant)")
    print("="*80)
    med_format = check_directory(models_dir / "medgemma-27b-text-it")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    if gpt_format == "gguf" and med_format == "gguf":
        print("\n✓ Both models are in GGUF format - ready for Ollama!")
        print("\nNext steps:")
        print("1. Run: python scripts/setup_local_models.py")
        print("2. Or manually import with: ollama create <model-name> -f Modelfile")
    
    elif gpt_format == "gguf" or med_format == "gguf":
        print("\n⚠ Some models are ready, some need conversion")
        if gpt_format != "gguf":
            print(f"  - GPT-OSS-20B: {gpt_format} format (needs conversion)")
        if med_format != "gguf":
            print(f"  - MedGemma-27B: {med_format} format (needs conversion)")
        print("\nConversion options:")
        print("1. Use llama.cpp converter")
        print("2. Download GGUF versions from HuggingFace")
        print("\nSee: docs/LOCAL_MODEL_SETUP_GUIDE.md")
    
    else:
        print("\n✗ Models need to be converted to GGUF format")
        print("\nConversion required for:")
        if gpt_format:
            print(f"  - GPT-OSS-20B: {gpt_format} → GGUF")
        if med_format:
            print(f"  - MedGemma-27B: {med_format} → GGUF")
        print("\nOptions:")
        print("1. Convert using llama.cpp:")
        print("   python convert.py D:/hf_models/gpt-oss-20b --outfile model.gguf")
        print("\n2. Download GGUF versions from HuggingFace:")
        print("   - Search for 'gpt-oss-20b GGUF'")
        print("   - Search for 'medgemma-27b GGUF'")
        print("\nSee: docs/LOCAL_MODEL_SETUP_GUIDE.md for detailed instructions")


if __name__ == "__main__":
    main()
