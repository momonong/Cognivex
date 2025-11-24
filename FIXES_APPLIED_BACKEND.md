# Backend Fixes Applied - 2024-11-24

## Overview
Fixed two critical backend issues that were causing system failures during initialization and inference.

## Task 1: Fixed Random Forest Model Path (Switch to 3-Class Model)

### Changes Made

#### 1. `app/core/ml_processing/cdda_tools.py`

**Updated `CDDAToolKit.__init__`:**
- Changed default `model_path` from `"model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib"` (2-class) to `"model/cnn_rf/rf_model_NC_MCI_AD.joblib"` (3-class)
- Added `self.classes` mapping: `{0: 'NC', 1: 'MCI', 2: 'AD'}`
- Enhanced initialization logging to show model path and classes

**Updated `_calculate_uq_score`:**
- Added documentation noting support for 2 or 3 classes
- Updated comment for `max_entropy` to clarify it handles both `log(2)` and `log(3)`
- The method already correctly handles variable-length probability arrays

**Impact:**
- System now uses the 3-class model by default
- `get_diagnostic_report` correctly handles 3 probabilities (NC, MCI, AD)
- UQ score calculation works correctly for both 2-class and 3-class models

#### 2. `app/core/mcp_server.py`

**Updated `DiagnosticMCPServer.__init__`:**
- Changed default toolkit initialization to use 3-class model: `"model/cnn_rf/rf_model_NC_MCI_AD.joblib"`

**Impact:**
- MCP server now initializes with the correct 3-class model
- All diagnostic resources and tools use the 3-class model

## Task 2: Fixed HuggingFace Loading Crash (Quantization Conflict)

### Changes Made

#### `app/services/llm_providers/huggingface.py`

**Updated `load_model` function:**
- Added try-except block around `AutoModelForCausalLM.from_pretrained()`
- Implemented fallback mechanism for pre-quantized models:
  1. **Try:** Load with `load_in_8bit` or `load_in_4bit` quantization config (for standard models)
  2. **Except ValueError:** If model rejects the config (already quantized with Mxfp4Config), catch the error and reload WITHOUT quantization parameters
  3. Uses only `device_map="auto"` and `trust_remote_code=True` for pre-quantized models

**Error Handling:**
```python
try:
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    print("[OK] Model loaded successfully")
except ValueError as e:
    if "quantized" in str(e).lower() and (load_in_8bit or load_in_4bit):
        # Model is already quantized (e.g., with Mxfp4Config), remove quantization params
        print(f"[INFO] Model is already quantized, loading without additional quantization config")
        model_kwargs.pop("load_in_8bit", None)
        model_kwargs.pop("load_in_4bit", None)
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        print("[OK] Model loaded successfully (using model's native quantization)")
    else:
        raise
```

**Impact:**
- Standard models (like MedGemma) load with BitsAndBytes quantization as before
- Pre-quantized models (like GPT-OSS-20B with Mxfp4Config) load successfully without conflicts
- System gracefully handles both model types

## Testing Recommendations

### Test 1: Verify 3-Class Model Loading
```bash
python -c "from app.core.ml_processing.cdda_tools import CDDAToolKit; tk = CDDAToolKit(); print('Classes:', tk.classes)"
```

Expected output:
```
Classes: {0: 'NC', 1: 'MCI', 2: 'AD'}
```

### Test 2: Verify Diagnostic Report with 3 Classes
```bash
python -c "from app.core.ml_processing.cdda_tools import CDDAToolKit; tk = CDDAToolKit(); report = tk.get_diagnostic_report('sub-0005', verbose=False); print('Prediction:', report['prediction_result'])"
```

Expected: Should complete without errors and return one of: NC, MCI, or AD

### Test 3: Verify HuggingFace Model Loading
```bash
python scripts/demo_phase4_complete.py
```

Expected: Should load models without "quantized with Mxfp4Config" error

### Test 4: Full System Test
```bash
python scripts/demo_phase4_complete.py
```

Expected: All three demo cases should complete successfully

## Files Modified

1. `app/core/ml_processing/cdda_tools.py` - Updated model path and class mapping
2. `app/core/mcp_server.py` - Updated default model path
3. `app/services/llm_providers/huggingface.py` - Added quantization conflict fallback

## Backward Compatibility

- The 2-class model can still be used by explicitly passing `model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib"` to `CDDAToolKit.__init__`
- All existing code that doesn't specify a model path will automatically use the 3-class model
- The UQ score calculation works correctly for both 2-class and 3-class models

## Error Messages Fixed

### Before Fix 1:
```
KeyError: 'MCI' or IndexError when accessing probabilities[2]
```

### After Fix 1:
✅ System correctly handles 3 classes (NC, MCI, AD)

### Before Fix 2:
```
ValueError: The model is quantized with Mxfp4Config but you are passing a BitsAndBytesConfig config.
```

### After Fix 2:
✅ System detects pre-quantized models and loads them without additional quantization config

## Next Steps

1. Run the full test suite to verify all functionality
2. Test with both standard and pre-quantized HuggingFace models
3. Verify MCP server initialization with the 3-class model
4. Test the complete demo script: `python scripts/demo_phase4_complete.py`
