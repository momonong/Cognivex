# HuggingFace Setup Guide for CDDA Phase 4

## Overview

This guide explains how to use your locally downloaded HuggingFace models (SafeTensors format) directly with the CDDA system, **without needing Ollama or GGUF conversion**.

## Advantages of HuggingFace Provider

✅ **No Conversion Needed**: Use SafeTensors models directly  
✅ **Full Control**: Direct access to model parameters  
✅ **Quantization**: Built-in 8-bit/4-bit quantization support  
✅ **Flexibility**: Easy to switch models or adjust settings  
✅ **Standard**: Uses official HuggingFace transformers library  

## Your Models

You have already downloaded:
- **GPT-OSS-20B**: `D:\hf_models\gpt-oss-20b` (3 SafeTensors files, ~13GB)
- **MedGemma-27B**: `D:\hf_models\medgemma-27b-text-it` (11 SafeTensors files, ~27GB)

These are ready to use! ✓

## Prerequisites

### 1. Install Required Packages

```bash
pip install transformers torch accelerate bitsandbytes
```

**Package Purposes:**
- `transformers`: HuggingFace library for loading models
- `torch`: PyTorch for model inference
- `accelerate`: For efficient model loading
- `bitsandbytes`: For 8-bit/4-bit quantization (saves memory)

### 2. Check GPU (Optional but Recommended)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Note**: Models can run on CPU, but GPU is much faster.

## Quick Start

### Test HuggingFace Provider

```bash
python scripts/test_huggingface_provider.py
```

This will:
1. Check if models exist
2. Load GPT-OSS-20B with 8-bit quantization
3. Test simple text generation
4. Test JSON generation (for Agent A)

### Use with Agent A

```python
from app.agents.agent_a_orchestrator import AgentA, AgentAConfig
from app.core.mcp_server import DiagnosticMCPServer

# Configure Agent A to use HuggingFace
config = AgentAConfig(
    model="gpt-oss-20b",
    model_path="D:/hf_models/gpt-oss-20b",
    provider="huggingface",  # Use HuggingFace instead of Ollama
    use_llm=True,
    load_in_8bit=True,  # Use 8-bit quantization to save memory
    verbose=True
)

# Initialize Agent A
agent_a = AgentA(mcp_server=mcp_server, config=config)

# Run orchestration
context_object = agent_a.orchestrate('sub-0005')
```

## Memory Requirements

### Without Quantization (Full Precision)
- **GPT-OSS-20B**: ~40GB RAM / ~20GB VRAM
- **MedGemma-27B**: ~54GB RAM / ~27GB VRAM

### With 8-bit Quantization (Recommended)
- **GPT-OSS-20B**: ~16GB RAM / ~10GB VRAM
- **MedGemma-27B**: ~20GB RAM / ~14GB VRAM

### With 4-bit Quantization (Maximum Savings)
- **GPT-OSS-20B**: ~8GB RAM / ~5GB VRAM
- **MedGemma-27B**: ~10GB RAM / ~7GB VRAM

## Configuration Options

### Basic Configuration

```python
config = AgentAConfig(
    model="gpt-oss-20b",
    model_path="D:/hf_models/gpt-oss-20b",
    provider="huggingface",
    use_llm=True,
    verbose=True
)
```

### With 8-bit Quantization (Recommended)

```python
config = AgentAConfig(
    model="gpt-oss-20b",
    model_path="D:/hf_models/gpt-oss-20b",
    provider="huggingface",
    load_in_8bit=True,  # Reduces memory by ~60%
    use_llm=True
)
```

### With 4-bit Quantization (Maximum Memory Savings)

```python
config = AgentAConfig(
    model="gpt-oss-20b",
    model_path="D:/hf_models/gpt-oss-20b",
    provider="huggingface",
    load_in_4bit=True,  # Reduces memory by ~75%
    use_llm=True
)
```

### Force CPU (No GPU)

```python
# In huggingface.handle_text()
response = huggingface.handle_text(
    prompt="...",
    model_path="D:/hf_models/gpt-oss-20b",
    device="cpu",  # Force CPU
    load_in_8bit=True
)
```

### Force GPU

```python
response = huggingface.handle_text(
    prompt="...",
    model_path="D:/hf_models/gpt-oss-20b",
    device="cuda",  # Force GPU
    load_in_8bit=True
)
```

## Comparison: HuggingFace vs Ollama

| Feature | HuggingFace | Ollama |
|---------|-------------|--------|
| Model Format | SafeTensors (native) | GGUF (needs conversion) |
| Setup | Install pip packages | Install Ollama + convert models |
| Memory Control | Precise (8-bit/4-bit) | Limited |
| Flexibility | High (direct API) | Medium (server-based) |
| Speed | Fast (direct) | Fast (optimized) |
| Ease of Use | Medium | Easy |

**Recommendation**: Use HuggingFace since you already have SafeTensors models.

## Troubleshooting

### Issue: "Out of Memory"

**Solution 1**: Use 8-bit quantization
```python
config = AgentAConfig(load_in_8bit=True)
```

**Solution 2**: Use 4-bit quantization
```python
config = AgentAConfig(load_in_4bit=True)
```

**Solution 3**: Use CPU
```python
# In handle_text call
device="cpu"
```

### Issue: "CUDA out of memory"

**Solution**: Clear GPU cache
```python
import torch
torch.cuda.empty_cache()

# Or use the provider's clear_cache
from app.services.llm_providers import huggingface
huggingface.clear_cache()
```

### Issue: "Model loading is slow"

**Cause**: First load always takes time (2-5 minutes)

**Solution**: Models are cached after first load. Subsequent loads are instant.

### Issue: "transformers not installed"

**Solution**:
```bash
pip install transformers torch accelerate bitsandbytes
```

### Issue: "bitsandbytes not available"

**Cause**: bitsandbytes not installed (needed for quantization)

**Solution**:
```bash
pip install bitsandbytes
```

Or disable quantization:
```python
config = AgentAConfig(load_in_8bit=False, load_in_4bit=False)
```

## Performance Tips

### 1. Use GPU if Available
GPU inference is 10-50x faster than CPU.

### 2. Use Quantization
8-bit quantization reduces memory by 60% with minimal quality loss.

### 3. Cache Models
Models are automatically cached after first load.

### 4. Batch Processing
If processing multiple subjects, keep model loaded:
```python
# Load once
agent_a = AgentA(mcp_server=mcp_server, config=config)

# Process multiple subjects
for subject_id in subjects:
    context = agent_a.orchestrate(subject_id)
```

### 5. Clear Cache When Done
```python
from app.services.llm_providers import huggingface
huggingface.clear_cache()
```

## Testing

### Test 1: Check Model Files

```bash
python scripts/check_model_format.py
```

Expected output:
```
✓ SafeTensors files found (ready for HuggingFace)
```

### Test 2: Test HuggingFace Provider

```bash
python scripts/test_huggingface_provider.py
```

This will test:
- Model loading
- Text generation
- JSON generation

### Test 3: Test Agent A

```bash
python app/agents/agent_a_orchestrator.py
```

This will run Agent A with HuggingFace provider.

## Next Steps

1. **Test the provider**:
   ```bash
   python scripts/test_huggingface_provider.py
   ```

2. **Test Agent A**:
   ```bash
   python app/agents/agent_a_orchestrator.py
   ```

3. **Implement Agent B** (Task 4):
   - Use MedGemma-27B with HuggingFace provider
   - Similar configuration as Agent A

4. **Run full system** (Task 5):
   - A2A integration with both agents

## Example: Complete Agent A Setup

```python
from app.agents.agent_a_orchestrator import AgentA, AgentAConfig
from app.core.mcp_server import DiagnosticMCPServer
from app.core.ml_processing.cdda_tools import CDDAToolKit
from app.core.knowledge.graph_rag import GraphRAG

# Initialize MCP server
toolkit = CDDAToolKit(
    model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
    data_root="data/MRI_processed"
)
graph_rag = GraphRAG()
mcp_server = DiagnosticMCPServer(toolkit=toolkit, graph_rag=graph_rag)

# Configure Agent A with HuggingFace
config = AgentAConfig(
    model="gpt-oss-20b",
    model_path="D:/hf_models/gpt-oss-20b",
    provider="huggingface",
    use_llm=True,
    load_in_8bit=True,  # Save memory
    verbose=True
)

# Initialize Agent A
agent_a = AgentA(mcp_server=mcp_server, config=config)

# Run analysis
context_object = agent_a.orchestrate('sub-0005')

# Print results
print(f"Subject: {context_object.subject_id}")
print(f"Prediction: {context_object.diagnostic_report.prediction_result}")
print(f"Decision: {context_object.decision_rationale}")
```

## References

- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Model Quantization: https://huggingface.co/docs/transformers/main/en/quantization
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- CDDA Design Doc: `.kiro/specs/cdda-phase4-dual-llm/design.md`
