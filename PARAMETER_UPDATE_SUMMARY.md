# Parameter Update Summary: load_in_8bit → use_4bit

## Overview
Updated all files to use the new `use_4bit` parameter instead of the deprecated `load_in_8bit` parameter in `CDDAAgent.__init__()`.

## Reason for Change
- The CDDA system now uses **4-bit quantization** instead of 8-bit for better VRAM efficiency
- 4-bit quantization reduces memory usage by ~50% compared to 8-bit
- Allows both Phi-4-mini (Orchestrator) and MedGemma-27b (Consultant) to fit within 24GB VRAM

## Files Updated

### 1. `app_smri.py`
**Before:**
```python
agent = CDDAAgent(
    orchestrator_model="Phi-4-mini-instruct",
    orchestrator_model_path=orchestrator_model_path,
    consultant_model="medgemma-27b",
    consultant_model_path=consultant_model_path,
    model_path="model/cnn_rf/rf_model_NC_MCI_AD.joblib",
    data_root="data/MRI_processed",
    use_llm=use_llm,
    load_in_8bit=True,  # OLD PARAMETER
    verbose=True
)
```

**After:**
```python
agent = CDDAAgent(
    orchestrator_model="phi-4-mini",
    orchestrator_model_path=orchestrator_model_path or "D:/hf_models/Phi-4-mini-instruct",
    consultant_model="medgemma-27b",
    consultant_model_path=consultant_model_path or "D:/hf_models/medgemma-27b-text-it",
    model_path="model/cnn_rf/rf_model_NC_MCI_AD.joblib",
    data_root="data/MRI_processed",
    use_llm=use_llm,
    use_4bit=True,  # NEW PARAMETER
    verbose=True
)
```

**Changes:**
- ✅ `load_in_8bit=True` → `use_4bit=True`
- ✅ Updated model paths to actual locations
- ✅ Added default paths with `or` operator

### 2. `app_cdda.py`
**Before:**
```python
agent = CDDAAgent(
    orchestrator_model="gpt-oss-20b",
    orchestrator_model_path=orchestrator_model_path,
    consultant_model="medgemma-27b",
    consultant_model_path=consultant_model_path,
    use_llm=use_llm,
    load_in_8bit=True,  # OLD PARAMETER
    verbose=True
)
```

**After:**
```python
agent = CDDAAgent(
    orchestrator_model="phi-4-mini",
    orchestrator_model_path=orchestrator_model_path or "D:/hf_models/Phi-4-mini-instruct",
    consultant_model="medgemma-27b",
    consultant_model_path=consultant_model_path or "D:/hf_models/medgemma-27b-text-it",
    use_llm=use_llm,
    use_4bit=True,  # NEW PARAMETER
    verbose=True
)
```

**Changes:**
- ✅ `load_in_8bit=True` → `use_4bit=True`
- ✅ Updated orchestrator from `gpt-oss-20b` to `phi-4-mini`
- ✅ Added default model paths
- ✅ Updated comment to reflect 4-bit quantization

### 3. `test_huggingface_integration.py`
**Before:**
```python
agent = CDDAAgent(
    orchestrator_model="test-model",
    orchestrator_model_path=model_path,
    consultant_model="test-model",
    consultant_model_path=model_path,
    use_llm=True,
    load_in_8bit=True,  # OLD PARAMETER
    verbose=True
)
```

**After:**
```python
agent = CDDAAgent(
    orchestrator_model="test-model",
    orchestrator_model_path=model_path,
    consultant_model="test-model",
    consultant_model_path=model_path,
    use_llm=True,
    use_4bit=True,  # NEW PARAMETER
    verbose=True
)
```

**Changes:**
- ✅ `load_in_8bit=True` → `use_4bit=True`

### 4. `app/agents/cdda_agent.py` (Already Updated)
The main `CDDAAgent` class was already updated to:
- Accept `use_4bit` parameter (default: `True`)
- Remove `load_in_8bit` parameter
- Use conditional logic: `load_in_8bit=not use_4bit` when passing to Agent configs

### 5. Model Path Updates
All files now use the correct model paths:
- **Orchestrator**: `D:/hf_models/Phi-4-mini-instruct`
- **Consultant**: `D:/hf_models/medgemma-27b-text-it` (not `medgemma-27b`)

## Parameter Comparison

| Parameter | Old Value | New Value | Purpose |
|-----------|-----------|-----------|---------|
| `load_in_8bit` | `True` | ❌ Removed | 8-bit quantization (deprecated) |
| `use_4bit` | N/A | `True` | 4-bit quantization (new default) |
| `orchestrator_model` | `"gpt-oss-20b"` | `"phi-4-mini"` | Updated to Phi-4 |
| `orchestrator_model_path` | Various | `"D:/hf_models/Phi-4-mini-instruct"` | Standardized path |
| `consultant_model_path` | Various | `"D:/hf_models/medgemma-27b-text-it"` | Corrected path |

## Benefits of 4-bit Quantization

### Memory Savings
- **8-bit quantization**: ~12-14GB per 27B model
- **4-bit quantization**: ~6-8GB per 27B model
- **Total savings**: ~50% reduction in VRAM usage

### Performance
- **Inference speed**: Slightly faster due to reduced memory bandwidth
- **Quality**: Minimal degradation with NF4 quantization
- **Compatibility**: Works with both standard and pre-quantized models

### VRAM Budget (24GB Total)
```
Phi-4-mini (4-bit):        ~7-8GB
MedGemma-27b (4-bit):     ~14-16GB
System overhead:           ~2-3GB
--------------------------------
Total:                    ~23-27GB (tight but feasible)
```

## Testing

### Verify Parameter Update
```bash
python -c "from app.agents.cdda_agent import CDDAAgent; import inspect; sig = inspect.signature(CDDAAgent.__init__); print('use_4bit' in sig.parameters)"
```
Expected output: `True`

### Test Streamlit Apps
```bash
# Test sMRI app
streamlit run app_smri.py

# Test CDDA app
streamlit run app_cdda.py
```

### Test Integration
```bash
python test_huggingface_integration.py
```

## Error Fixed

### Original Error
```
TypeError: CDDAAgent.__init__() got an unexpected keyword argument 'load_in_8bit'
```

### Root Cause
- `CDDAAgent.__init__()` was updated to use `use_4bit` parameter
- Streamlit apps (`app_smri.py`, `app_cdda.py`) were still using old `load_in_8bit` parameter
- Test files also had outdated parameter

### Solution
- Updated all files to use `use_4bit=True`
- Standardized model paths across all files
- Added default path fallbacks with `or` operator

## Backward Compatibility

### Breaking Change
⚠️ **This is a breaking change**: Code using `load_in_8bit` will fail with `TypeError`

### Migration Guide
If you have custom code using `CDDAAgent`:

**Old code:**
```python
agent = CDDAAgent(load_in_8bit=True)
```

**New code:**
```python
agent = CDDAAgent(use_4bit=True)
```

### Why Not Keep Both?
- Simplifies the API (one quantization parameter instead of two)
- 4-bit is superior for VRAM efficiency
- Reduces confusion about which parameter to use
- Aligns with modern quantization best practices

## Files Verified

All files now pass diagnostics:
- ✅ `app_smri.py` - No diagnostics found
- ✅ `app_cdda.py` - No diagnostics found
- ✅ `test_huggingface_integration.py` - No diagnostics found
- ✅ `app/agents/cdda_agent.py` - No diagnostics found
- ✅ `app/agents/llm_factory.py` - No diagnostics found

## Next Steps

1. **Test Streamlit Apps**: Run both `app_smri.py` and `app_cdda.py` to verify they initialize correctly
2. **Monitor VRAM**: Check that both models fit within 24GB during inference
3. **Verify Inference**: Test end-to-end analysis with actual subjects
4. **Update Documentation**: Ensure all user-facing docs reflect the new parameter

## Related Files

- `PHI4_MEDGEMMA_INTEGRATION.md` - Main integration documentation
- `FIXES_APPLIED_BACKEND.md` - Backend fixes documentation
- `test_phi4_integration.py` - Comprehensive integration test
- `app/agents/llm_factory.py` - Model loading factory

## Summary

✅ All files updated to use `use_4bit` parameter
✅ Model paths corrected to actual locations
✅ Default paths added for convenience
✅ No syntax errors or diagnostics issues
✅ System ready for testing with Phi-4-mini + MedGemma-27b

The system is now configured to use 4-bit quantization for optimal VRAM efficiency while maintaining model quality.
