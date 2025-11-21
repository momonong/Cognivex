# LLM Troubleshooting Guide - CDDA Phase 4

**Last Updated:** November 20, 2025

This guide helps troubleshoot common issues with LLM integration in CDDA Phase 4's dual-LLM architecture.

---

## Table of Contents

1. [Model Availability Issues](#model-availability-issues)
2. [Memory Issues](#memory-issues)
3. [Connection Issues](#connection-issues)
4. [Response Parsing Issues](#response-parsing-issues)
5. [Performance Issues](#performance-issues)
6. [Fallback Mechanisms](#fallback-mechanisms)
7. [Common Error Messages](#common-error-messages)

---

## Model Availability Issues

### Problem: Model not found in Ollama

**Error Message:**
```
[WARNING] Model 'gpt-oss-20b' not found in Ollama
[INFO] Available models: llama3.1:8b, mistral:7b
```

**Solution:**

```bash
# Check available models
ollama list

# Pull the required model
ollama pull gpt-oss-20b

# Alternative: Use a different model
ollama pull llama3.1:8b

# Update configuration to use available model
python -c "
from app.agents.cdda_agent import CDDAAgent
agent = CDDAAgent(
    orchestrator_model='llama3.1:8b',  # Use available model
    use_llm=True
)
"
```

### Problem: HuggingFace model not found

**Error Message:**
```
[WARNING] Model not found at: D:/hf_models/gpt-oss-20b
[INFO] Please ensure the model is downloaded
```

**Solution:**

```python
# Download model from HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your-org/gpt-oss-20b"  # Replace with actual model
save_path = "D:/hf_models/gpt-oss-20b"

# Download model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
```

**Alternative: Use rule-based fallback (no LLM required):**

```python
from app.agents.cdda_agent import CDDAAgent

# Initialize with rule-based fallback
agent = CDDAAgent(
    use_llm=False,  # Disable LLM, use rule-based logic
    verbose=True
)

# Works without any LLM models
result = agent.run_analysis('sub-0005')
```

---

## Memory Issues

### Problem: CUDA out of memory

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solution 1: Enable 8-bit quantization**

```python
from app.agents.cdda_agent import CDDAAgent

agent = CDDAAgent(
    orchestrator_model="gpt-oss-20b",
    consultant_model="medgemma-27b",
    load_in_8bit=True,  # Enable 8-bit quantization
    use_llm=True
)
```

**Solution 2: Use smaller models**

```python
agent = CDDAAgent(
    orchestrator_model="llama3.1:8b",  # Smaller model
    consultant_model="llama3.1:8b",    # Smaller model
    use_llm=True
)
```

**Solution 3: Use CPU instead of GPU**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU

agent = CDDAAgent(use_llm=True)
```

**Solution 4: Use rule-based fallback**

```python
agent = CDDAAgent(use_llm=False)  # No GPU required
```

### Problem: System RAM exhausted

**Error Message:**
```
MemoryError: Unable to allocate array
```

**Solution:**

```bash
# Clear Python cache
pip cache purge
python -c "import gc; gc.collect()"

# Monitor memory usage
free -h  # Linux
# or
Get-Process python | Select-Object WorkingSet  # Windows PowerShell

# Use rule-based fallback (minimal memory)
python -c "
from app.agents.cdda_agent import CDDAAgent
agent = CDDAAgent(use_llm=False)
result = agent.run_analysis('sub-0005')
"
```

---

## Connection Issues

### Problem: Ollama server not running

**Error Message:**
```
ConnectionError: Failed to connect to Ollama server at localhost:11434
```

**Solution:**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama server
ollama serve

# Or restart Ollama
# Windows: Restart Ollama from system tray
# Linux: sudo systemctl restart ollama
# macOS: Restart Ollama app

# Verify connection
python -c "
from app.services.llm_providers import ollama
models = ollama.list_models()
print(f'Available models: {models}')
"
```

### Problem: HuggingFace connection timeout

**Error Message:**
```
TimeoutError: Connection to HuggingFace timed out
```

**Solution:**

```python
# Use local models instead of downloading
from app.agents.cdda_agent import CDDAAgent

agent = CDDAAgent(
    orchestrator_model_path="D:/hf_models/gpt-oss-20b",  # Local path
    consultant_model_path="D:/hf_models/medgemma-27b",   # Local path
    use_llm=True
)

# Or use Ollama instead
agent = CDDAAgent(
    orchestrator_model="llama3.1:8b",
    consultant_model="llama3.1:8b",
    provider="ollama",
    use_llm=True
)
```

---

## Response Parsing Issues

### Problem: LLM returns malformed JSON

**Error Message:**
```
LLMParsingError: Failed to parse LLM response as JSON
```

**Solution:**

The system includes automatic JSON recovery, but if it fails:

```python
# The system will automatically:
# 1. Try to extract JSON from response
# 2. Fix common JSON issues (missing quotes, trailing commas)
# 3. Retry with clarification prompt
# 4. Fall back to rule-based logic after 2 attempts

# To see detailed parsing attempts:
from app.agents.cdda_agent import CDDAAgent

agent = CDDAAgent(
    use_llm=True,
    verbose=True  # Enable verbose logging
)

result = agent.run_analysis('sub-0005')
```

**Manual fallback:**

```python
# If parsing continues to fail, use rule-based fallback
agent = CDDAAgent(use_llm=False)
result = agent.run_analysis('sub-0005')
```

### Problem: LLM returns incomplete response

**Error Message:**
```
ValueError: Response missing required fields
```

**Solution:**

```python
# System automatically adds default values for missing fields
# Check reasoning chain for details:

result = agent.run_analysis('sub-0005')
for step in result.reasoning_chain:
    if 'missing' in step.lower() or 'default' in step.lower():
        print(step)

# If issues persist, use rule-based fallback
agent = CDDAAgent(use_llm=False)
```

---

## Performance Issues

### Problem: LLM responses are slow

**Symptoms:**
- Analysis takes > 30 seconds
- System appears frozen

**Solution 1: Use smaller models**

```python
agent = CDDAAgent(
    orchestrator_model="llama3.1:8b",  # Faster than 20B models
    consultant_model="llama3.1:8b",
    use_llm=True
)
```

**Solution 2: Reduce temperature (faster inference)**

```python
from app.agents.agent_a_orchestrator import AgentAConfig
from app.agents.agent_b_consultant import AgentBConfig

agent_a_config = AgentAConfig(
    model="gpt-oss-20b",
    temperature=0.0,  # Lower temperature = faster
    use_llm=True
)

agent_b_config = AgentBConfig(
    model="medgemma-27b",
    temperature=0.1,  # Lower temperature = faster
    use_llm=True
)
```

**Solution 3: Use GPU acceleration**

```bash
# Ensure CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If not available, install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Solution 4: Use rule-based fallback (instant)**

```python
agent = CDDAAgent(use_llm=False)  # No LLM latency
result = agent.run_analysis('sub-0005')
```

---

## Fallback Mechanisms

### Understanding Fallback Behavior

CDDA Phase 4 includes multi-layer fallback mechanisms:

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: LLM-Based Orchestration (Agent A)              │
│   ↓ (if fails)                                          │
│ Layer 2: Rule-Based Orchestration                       │
│   ↓ (always works)                                      │
│ Layer 3: LLM-Based Synthesis (Agent B)                  │
│   ↓ (if fails)                                          │
│ Layer 4: Template-Based Synthesis                       │
│   ↓ (always works)                                      │
│ Layer 5: GraphRAG Knowledge Retrieval                   │
│   ↓ (if fails)                                          │
│ Layer 6: Local Knowledge Base                           │
│   ↓ (always works)                                      │
└─────────────────────────────────────────────────────────┘
```

### Testing Fallback Mechanisms

```python
from app.agents.cdda_agent import CDDAAgent

# Test Agent A fallback
agent = CDDAAgent(
    use_llm=True,
    orchestrator_model="nonexistent-model",  # Will trigger fallback
    verbose=True
)
result = agent.run_analysis('sub-0005')
# Should see: "LLM orchestration failed. Using rule-based fallback."

# Test Agent B fallback
agent = CDDAAgent(
    use_llm=True,
    consultant_model="nonexistent-model",  # Will trigger fallback
    verbose=True
)
result = agent.run_analysis('sub-0005')
# Should see: "LLM synthesis failed. Using template-based fallback."

# Test complete fallback (no LLMs)
agent = CDDAAgent(use_llm=False, verbose=True)
result = agent.run_analysis('sub-0005')
# Should work without any LLM calls
```

### Checking Fallback Usage

```python
result = agent.run_analysis('sub-0005')

# Check reasoning chain for fallback indicators
for step in result.reasoning_chain:
    if 'fallback' in step.lower():
        print(f"Fallback used: {step}")

# Check for error annotations
if hasattr(result.context_object, 'errors'):
    for error in result.context_object.errors:
        print(f"Error: {error['type']} - {error['message']}")

# Check metadata
if 'use_llm' in result.metadata:
    print(f"LLM mode: {result.metadata['use_llm']}")
```

---

## Common Error Messages

### "LLMRetryExhausted: Maximum retry attempts reached"

**Cause:** LLM failed after 3 retry attempts

**Solution:**
- System automatically falls back to rule-based logic
- Check reasoning chain for details
- Consider using rule-based mode: `use_llm=False`

### "LLMParsingError: Failed to parse LLM response"

**Cause:** LLM returned malformed JSON

**Solution:**
- System automatically attempts JSON recovery
- Falls back to rule-based logic after 2 attempts
- Enable verbose mode to see parsing attempts

### "ValueError: Model path required for HuggingFace provider"

**Cause:** Using HuggingFace provider without model path

**Solution:**
```python
agent = CDDAAgent(
    orchestrator_model="gpt-oss-20b",
    orchestrator_model_path="D:/hf_models/gpt-oss-20b",  # Add path
    provider="huggingface"
)
```

### "ServiceUnavailable: Ollama server not responding"

**Cause:** Ollama server not running

**Solution:**
```bash
# Start Ollama server
ollama serve

# Or use HuggingFace provider
agent = CDDAAgent(
    orchestrator_model_path="D:/hf_models/gpt-oss-20b",
    provider="huggingface"
)
```

---

## Best Practices

### 1. Start with Rule-Based Mode

```python
# Test system without LLMs first
agent = CDDAAgent(use_llm=False, verbose=True)
result = agent.run_analysis('sub-0005')

# Once working, enable LLMs
agent = CDDAAgent(use_llm=True, verbose=True)
result = agent.run_analysis('sub-0005')
```

### 2. Enable Verbose Logging

```python
agent = CDDAAgent(
    use_llm=True,
    verbose=True  # See detailed logs
)
```

### 3. Use 8-bit Quantization

```python
agent = CDDAAgent(
    use_llm=True,
    load_in_8bit=True  # Reduce memory usage
)
```

### 4. Monitor Reasoning Chains

```python
result = agent.run_analysis('sub-0005')

# Check for issues
for step in result.reasoning_chain:
    print(step)

# Save for analysis
agent.save_reasoning_log(result, "output/reasoning.json")
```

### 5. Test Fallback Mechanisms

```python
# Periodically test fallback
agent_fallback = CDDAAgent(use_llm=False)
result_fallback = agent_fallback.run_analysis('sub-0005')

# Compare with LLM mode
agent_llm = CDDAAgent(use_llm=True)
result_llm = agent_llm.run_analysis('sub-0005')

# Both should produce valid results
assert result_fallback.prediction in ['AD', 'NC', 'MCI']
assert result_llm.prediction in ['AD', 'NC', 'MCI']
```

---

## Getting Help

### Check System Status

```python
# Run health check
from app.agents.cdda_agent import CDDAAgent

try:
    agent = CDDAAgent(use_llm=True, verbose=True)
    result = agent.run_analysis('sub-0005')
    print("✓ System working correctly")
except Exception as e:
    print(f"✗ System error: {e}")
    print("Trying fallback mode...")
    agent = CDDAAgent(use_llm=False, verbose=True)
    result = agent.run_analysis('sub-0005')
    print("✓ Fallback mode working")
```

### Export Logs for Debugging

```python
result = agent.run_analysis('sub-0005')

# Save complete reasoning log
agent.save_reasoning_log(result, "output/debug_reasoning.json")

# Save clinical report
with open("output/debug_report.txt", "w") as f:
    f.write(result.clinical_report)

# Print error details
if hasattr(result.context_object, 'errors'):
    for error in result.context_object.errors:
        print(f"Error: {error}")
```

### Contact Information

For additional support:
- Review documentation: `docs/CDDA_Phase4_Complete.md`
- Check examples: `scripts/demo_*.py`
- Run demos: `python scripts/demo_phase4_complete.py`

---

**Last Updated:** November 20, 2025  
**Version:** Phase 4 (MCP + A2A)
