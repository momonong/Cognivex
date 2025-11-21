# Agent A Model Setup Guide

## Overview

Agent A (Orchestrator) is designed to use **GPT-OSS-20B** for function calling and decision logic. This document explains how to set up the model.

## Model Requirements

According to the design specification:
- **Agent A (Orchestrator)**: GPT-OSS-20B
  - Role: MCP Client & Planner
  - Capabilities: Function calling, decision logic, tool orchestration
  
- **Agent B (Consultant)**: MedGemma-27B
  - Role: Medical Specialist
  - Capabilities: Clinical synthesis, medical reasoning

## GPT-OSS-20B Setup

### Option 1: Use GPT-OSS-20B (Recommended)

If GPT-OSS-20B is available in Ollama:

```bash
# Pull the model
ollama pull gpt-oss-20b

# Verify it's available
ollama list
```

### Option 2: Use Alternative Function-Calling Model

If GPT-OSS-20B is not available, you can use alternative models with strong function-calling capabilities:

#### Llama 3.1 (Good function calling support)
```bash
ollama pull llama3.1:8b
```

Update the configuration:
```python
config = AgentAConfig(
    model="llama3.1:8b",
    use_llm=True
)
```

#### Mistral (Good function calling support)
```bash
ollama pull mistral:7b
```

Update the configuration:
```python
config = AgentAConfig(
    model="mistral:7b",
    use_llm=True
)
```

### Option 3: Use Rule-Based Fallback (No LLM Required)

If no suitable LLM is available, Agent A can operate in rule-based mode:

```python
config = AgentAConfig(
    use_llm=False  # Disable LLM, use rule-based logic
)
```

This mode implements the same decision logic but without LLM reasoning:
- IF UQ > threshold → Trigger counterfactual simulation
- IF anomaly detected → Query knowledge graph
- ELSE → Standard report

## Model Comparison

| Model | Size | Function Calling | Speed | Recommended For |
|-------|------|-----------------|-------|-----------------|
| GPT-OSS-20B | 20B | Excellent | Medium | Production (if available) |
| Llama 3.1:8b | 8B | Good | Fast | Development/Testing |
| Mistral:7b | 7B | Good | Fast | Development/Testing |
| Rule-based | N/A | N/A | Very Fast | Fallback/Demo |

## Testing Model Availability

```python
from app.services.llm_providers import ollama

# Check if Ollama is running
if ollama.check_availability():
    print("Ollama is available")
    
    # List available models
    models = ollama.list_models()
    print(f"Available models: {models}")
    
    # Check if GPT-OSS-20B is available
    if "gpt-oss-20b" in models:
        print("GPT-OSS-20B is ready to use")
    else:
        print("GPT-OSS-20B not found. Consider using llama3.1:8b")
else:
    print("Ollama not available. Will use rule-based fallback")
```

## Configuration in Code

### Using GPT-OSS-20B (Default)
```python
from app.agents.agent_a_orchestrator import AgentA, AgentAConfig

config = AgentAConfig(
    model="gpt-oss-20b",  # Default
    use_llm=True,
    verbose=True
)

agent_a = AgentA(mcp_server=mcp_server, config=config)
```

### Using Alternative Model
```python
config = AgentAConfig(
    model="llama3.1:8b",  # Alternative
    use_llm=True,
    verbose=True
)

agent_a = AgentA(mcp_server=mcp_server, config=config)
```

### Using Rule-Based Fallback
```python
config = AgentAConfig(
    use_llm=False,  # Disable LLM
    verbose=True
)

agent_a = AgentA(mcp_server=mcp_server, config=config)
```

## Automatic Fallback

Agent A automatically falls back to rule-based logic if:
1. The specified model is not available
2. Ollama server is not running
3. LLM call fails or times out

This ensures the system continues to function even when LLMs are unavailable.

## Performance Considerations

### GPT-OSS-20B (20B parameters)
- **RAM Required**: ~40GB
- **VRAM Required**: ~20GB (GPU)
- **Inference Time**: ~2-5 seconds per call
- **Best For**: Production with powerful hardware

### Llama 3.1:8b (8B parameters)
- **RAM Required**: ~16GB
- **VRAM Required**: ~8GB (GPU)
- **Inference Time**: ~1-2 seconds per call
- **Best For**: Development and testing

### Rule-Based (No LLM)
- **RAM Required**: Minimal
- **VRAM Required**: None
- **Inference Time**: <100ms
- **Best For**: Demos, testing, resource-constrained environments

## Troubleshooting

### Model Not Found
```
Error: model 'gpt-oss-20b' not found (status code: 404)
```

**Solution**: Pull the model or use an alternative:
```bash
ollama pull gpt-oss-20b
# OR
ollama pull llama3.1:8b
```

### Ollama Not Running
```
Error: Ollama server is not running
```

**Solution**: Start Ollama server:
```bash
ollama serve
```

### Out of Memory
```
Error: Out of memory
```

**Solution**: Use a smaller model or rule-based mode:
```python
config = AgentAConfig(model="llama3.1:8b")  # Smaller model
# OR
config = AgentAConfig(use_llm=False)  # Rule-based
```

## References

- Design Document: `.kiro/specs/cdda-phase4-dual-llm/design.md`
- Requirements: `.kiro/specs/cdda-phase4-dual-llm/requirements.md`
- Ollama Documentation: https://ollama.ai
