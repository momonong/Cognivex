# Local Model Setup Guide for CDDA Phase 4

## Overview

This guide explains how to configure Ollama to use your locally downloaded models from `D:\hf_models`.

## Your Models

You have downloaded:
- **GPT-OSS-20B**: `D:\hf_models\gpt-oss-20b`
  - For Agent A (Orchestrator)
  - Function calling and decision logic
  
- **MedGemma-27B**: `D:\hf_models\medgemma-27b-text-it`
  - For Agent B (Consultant)
  - Medical reasoning and clinical synthesis

## Prerequisites

### 1. Install Ollama

If not already installed:
```bash
# Download from https://ollama.ai
# Or use winget on Windows
winget install Ollama.Ollama
```

### 2. Start Ollama Server

```bash
ollama serve
```

Keep this running in a separate terminal.

## Model Format Requirements

Ollama requires models in **GGUF format**. Check if your models are in GGUF format:

```bash
# Check GPT-OSS-20B
dir D:\hf_models\gpt-oss-20b\*.gguf

# Check MedGemma-27B
dir D:\hf_models\medgemma-27b-text-it\*.gguf
```

### If Models Are Not in GGUF Format

If your models are in PyTorch/SafeTensors format, you need to convert them:

#### Option 1: Use llama.cpp converter
```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert model
python convert.py D:\hf_models\gpt-oss-20b --outfile D:\hf_models\gpt-oss-20b\model.gguf
```

#### Option 2: Download GGUF versions
Look for GGUF versions on HuggingFace:
- Search for "gpt-oss-20b GGUF"
- Search for "medgemma-27b GGUF"

## Setup Methods

### Method 1: Automated Setup (Recommended)

Run the setup script:

```bash
python scripts/setup_local_models.py
```

This will:
1. Check Ollama installation
2. Verify models exist
3. Create Modelfiles
4. Import models to Ollama
5. Verify setup

### Method 2: Manual Setup

#### Step 1: Create Modelfile for GPT-OSS-20B

Create `Modelfile.gpt-oss-20b`:

```dockerfile
# Modelfile for GPT-OSS-20B (Agent A)
FROM D:/hf_models/gpt-oss-20b/model.gguf

# Set parameters for function calling
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

# System prompt for Agent A
SYSTEM """You are Agent A, the Orchestrator in a diagnostic system following Model Context Protocol (MCP).
Your role is to read diagnostic resources, evaluate signals, and decide which tools to invoke.
Respond with JSON containing 'actions' and 'decision_rationale'."""
```

#### Step 2: Import GPT-OSS-20B to Ollama

```bash
ollama create gpt-oss-20b -f Modelfile.gpt-oss-20b
```

#### Step 3: Create Modelfile for MedGemma-27B

Create `Modelfile.medgemma-27b`:

```dockerfile
# Modelfile for MedGemma-27B (Agent B)
FROM D:/hf_models/medgemma-27b-text-it/model.gguf

# Set parameters for medical reasoning
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192

# System prompt for Agent B
SYSTEM """You are Agent B, the Clinical Consultant specializing in neuroimaging and dementia diagnosis.
Your role is to synthesize clinical narratives from the ContextObject provided by Agent A.
You have NO access to tools or resources. You work ONLY with the context provided to you."""
```

#### Step 4: Import MedGemma-27B to Ollama

```bash
ollama create medgemma-27b -f Modelfile.medgemma-27b
```

## Verification

### 1. List Models

```bash
ollama list
```

You should see:
- `gpt-oss-20b`
- `medgemma-27b`

### 2. Test GPT-OSS-20B

```bash
ollama run gpt-oss-20b "Hello, can you help with function calling?"
```

### 3. Test MedGemma-27B

```bash
ollama run medgemma-27b "Explain the role of hippocampus in Alzheimer's disease"
```

### 4. Test with Python

```python
from app.services.llm_providers import ollama

# Check availability
if ollama.check_availability():
    print("Ollama is running")
    
    # List models
    models = ollama.list_models()
    print(f"Available models: {models}")
    
    # Test GPT-OSS-20B
    if "gpt-oss-20b" in models:
        response = ollama.handle_text(
            prompt="Test message",
            model="gpt-oss-20b"
        )
        print(f"GPT-OSS-20B response: {response}")
```

## Update Agent Configuration

Once models are imported, update the Agent A configuration:

```python
# app/agents/agent_a_orchestrator.py
# Default configuration already uses gpt-oss-20b

config = AgentAConfig(
    model="gpt-oss-20b",  # ✓ Now available in Ollama
    use_llm=True,
    verbose=True
)
```

For Agent B (when implemented):

```python
config = AgentBConfig(
    model="medgemma-27b",  # ✓ Now available in Ollama
    use_llm=True,
    verbose=True
)
```

## Troubleshooting

### Issue: "Model not found"

**Cause**: Model not imported to Ollama

**Solution**:
```bash
# Check if model exists in Ollama
ollama list

# If not, import it
ollama create gpt-oss-20b -f Modelfile.gpt-oss-20b
```

### Issue: "GGUF file not found"

**Cause**: Model not in GGUF format

**Solution**: Convert model to GGUF format (see above)

### Issue: "Out of memory"

**Cause**: Model too large for available RAM/VRAM

**Solution**: Use quantized version
```bash
# Look for quantized GGUF files
# e.g., model-q4_0.gguf (4-bit quantization)
# e.g., model-q8_0.gguf (8-bit quantization)
```

Update Modelfile to use quantized version:
```dockerfile
FROM D:/hf_models/gpt-oss-20b/model-q4_0.gguf
```

### Issue: "Ollama server not running"

**Solution**:
```bash
# Start Ollama server
ollama serve

# Or on Windows, start Ollama from Start Menu
```

### Issue: "Slow inference"

**Cause**: Model running on CPU instead of GPU

**Solution**: Ensure CUDA is installed and Ollama detects GPU
```bash
# Check GPU detection
ollama run gpt-oss-20b --verbose
```

## Model Sizes and Requirements

### GPT-OSS-20B
- **Parameters**: 20 billion
- **GGUF Size**: ~12-15 GB (depending on quantization)
- **RAM Required**: 16-32 GB
- **VRAM Required**: 12-16 GB (for GPU inference)
- **Inference Speed**: 2-5 seconds per response

### MedGemma-27B
- **Parameters**: 27 billion
- **GGUF Size**: ~16-20 GB (depending on quantization)
- **RAM Required**: 32-64 GB
- **VRAM Required**: 16-24 GB (for GPU inference)
- **Inference Speed**: 3-7 seconds per response

## Quantization Options

If you need to reduce memory usage, use quantized versions:

| Quantization | Size Reduction | Quality | Use Case |
|--------------|----------------|---------|----------|
| Q4_0 | ~75% smaller | Good | Development/Testing |
| Q5_0 | ~70% smaller | Better | Balanced |
| Q8_0 | ~50% smaller | Best | Production |
| F16 | Original | Perfect | High-end hardware |

## Alternative: Use Smaller Models for Testing

If the 20B/27B models are too large, you can test with smaller models:

```python
# For testing Agent A
config = AgentAConfig(
    model="llama3.1:8b",  # Much smaller
    use_llm=True
)

# For testing Agent B
config = AgentBConfig(
    model="llama3.1:8b",  # Much smaller
    use_llm=True
)
```

## Next Steps

After setup:

1. **Test Agent A**:
   ```bash
   python app/agents/agent_a_orchestrator.py
   ```

2. **Run Tests**:
   ```bash
   pytest tests/test_agent_a_orchestrator.py -v
   ```

3. **Proceed to Agent B Implementation** (Task 4)

## References

- Ollama Documentation: https://ollama.ai/docs
- GGUF Format: https://github.com/ggerganov/llama.cpp
- Model Quantization: https://huggingface.co/docs/transformers/main/en/quantization
- CDDA Design Doc: `.kiro/specs/cdda-phase4-dual-llm/design.md`
