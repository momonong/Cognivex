# Bug Fix: Missing MRI Files Error

## Problem

Agent A orchestration was getting stuck at "PHASE 1" with error: `'subject_id'`

### Root Cause

The application was allowing users to select subjects that don't have complete MRI data files. When Agent A tried to process these subjects, the toolkit would fail with:

```
FileNotFoundError: Missing MRI files for sub-0003. Found: GM=0, FA=0, MD=0
```

This caused the orchestration to fail before creating a ContextObject, resulting in the `'subject_id'` error when trying to access the result.

## Investigation

### Debug Process

1. Created `test_agent_a_debug.py` to isolate the issue
2. Tested each component individually:
   - ✓ Module imports
   - ✓ CDDAToolKit initialization
   - ✗ get_diagnostic_report for sub-0003 (FAILED)
3. Discovered that sub-0003 has no MRI files in `data/MRI_processed/MCI/sub-0003/`

### Data Validation

Checked which subjects have complete data:

```powershell
Get-ChildItem -Path "data\MRI_processed\*\sub-*" -Directory | 
  ForEach-Object { 
    $files = Get-ChildItem -Path $_.FullName -Filter "*.nii.gz"
    Write-Output "$($_.Name): $($files.Count) files" 
  }
```

**Results:**
- ✓ sub-0005: 3 files (GM, FA, MD)
- ✓ sub-0011: 3 files
- ✓ sub-0012: 3 files
- ✗ sub-0003: 0 files
- ✗ sub-0001: 0 files (likely)

## Solution

### 1. Frontend Validation (`app_cdda.py`)

Added file validation when scanning for subjects:

**Before:**
```python
subject_labels = {}
data_folders = glob.glob("data/MRI_processed/*/sub-*")
for folder_path in data_folders:
    parts = folder_path.split(os.sep)
    if len(parts) >= 3:
        subject_id = parts[-1]
        label = parts[-2]
        subject_labels[subject_id] = label
```

**After:**
```python
subject_labels = {}
data_folders = glob.glob("data/MRI_processed/*/sub-*")
for folder_path in data_folders:
    parts = folder_path.split(os.sep)
    if len(parts) >= 3:
        subject_id = parts[-1]
        label = parts[-2]
        
        # Check for complete MRI files (at least 3 .nii.gz files)
        nii_files = list(Path(folder_path).glob("*.nii.gz"))
        if len(nii_files) >= 3:
            subject_labels[subject_id] = label
```

### 2. Enhanced Error Handling (`app/agents/agent_a_orchestrator.py`)

Added better debugging and error messages:

```python
# Check for errors in result
if 'error' in result:
    raise ValueError(f"MCP server returned error: {result['error']}")

# Ensure subject_id is present
if 'subject_id' not in result:
    if self.config.verbose:
        print(f"[WARNING] subject_id not in result, adding it")
    result['subject_id'] = subject_id
```

### 3. Debug Script (`test_agent_a_debug.py`)

Created comprehensive test script to diagnose Agent A issues:

- Tests each component individually
- Provides detailed error messages
- Can be run independently: `python test_agent_a_debug.py`

## Testing

### Test 1: Valid Subject

```python
from app.agents.cdda_agent import CDDAAgent

agent = CDDAAgent(use_llm=False, verbose=True)
result = agent.run_analysis("sub-0005")  # Has complete data

print(f"✓ Analysis complete: {result.prediction}")
```

**Expected Output:**
```
[PHASE 1] Agent A - Orchestration
[AGENT A] Reading resource: diagnosis://sub-0005/report
[OK] DiagnosticReport created: subject_id=sub-0005
[PHASE 2] Agent B - Clinical Synthesis
[OK] Analysis complete
```

### Test 2: Invalid Subject (Now Prevented)

The frontend will no longer show subjects without complete data, so this error is prevented at the UI level.

## Files Modified

1. ✅ `app_cdda.py`
   - Added MRI file validation in subject selection
   - Only shows subjects with ≥3 .nii.gz files

2. ✅ `app/agents/agent_a_orchestrator.py`
   - Enhanced error handling in `_read_diagnostic_report()`
   - Added debug logging
   - Added subject_id validation

3. ✅ `test_agent_a_debug.py` (new)
   - Comprehensive debug script
   - Tests each component individually

4. ✅ `BUGFIX_MISSING_MRI_FILES.md` (this document)

## Required MRI Files

Each subject must have at least 3 MRI files:

1. **GM (Grey Matter)**: `*_GM_to_MNI.nii.gz`
2. **FA (Fractional Anisotropy)**: `*_FA_to_MNI.nii.gz`
3. **MD (Mean Diffusivity)**: `*_MD_to_MNI.nii.gz`

### Example Directory Structure

```
data/MRI_processed/
├── AD/
│   ├── sub-0005/
│   │   ├── sub-0005_GM_to_MNI.nii.gz  ✓
│   │   ├── sub-0005_FA_to_MNI.nii.gz  ✓
│   │   └── sub-0005_MD_to_MNI.nii.gz  ✓
│   └── sub-0003/
│       └── (empty - no files)         ✗
├── MCI/
│   └── sub-0011/
│       ├── sub-0011_GM_to_MNI.nii.gz  ✓
│       ├── sub-0011_FA_to_MNI.nii.gz  ✓
│       └── sub-0011_MD_to_MNI.nii.gz  ✓
└── NC/
    └── sub-0012/
        ├── sub-0012_GM_to_MNI.nii.gz  ✓
        ├── sub-0012_FA_to_MNI.nii.gz  ✓
        └── sub-0012_MD_to_MNI.nii.gz  ✓
```

## Prevention

### For Users

1. **Check data before running**: Use the debug script to validate subjects
2. **Only select valid subjects**: Frontend now filters automatically
3. **Verify file count**: Each subject should have 3+ .nii.gz files

### For Developers

1. **Always validate input**: Check for required files before processing
2. **Fail fast**: Validate at the UI level, not deep in the pipeline
3. **Clear error messages**: Tell users exactly what's missing

## Future Improvements

1. **Data Validation Tool**: Create a script to validate all subjects
2. **Better UI Feedback**: Show file count for each subject
3. **Automatic Data Check**: Run validation on startup
4. **Missing Data Report**: Generate report of incomplete subjects

## References

- **CDDAToolKit**: `app/core/ml_processing/cdda_tools.py`
- **ROI Feature Extractor**: `scripts/cnn_rf/extract_roi_features.py`
- **Agent A Orchestrator**: `app/agents/agent_a_orchestrator.py`

---

## Contact

For questions about this fix, refer to the project documentation or create an issue.
