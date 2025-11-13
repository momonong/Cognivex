# 未使用程式碼審查報告（修正版）

**審查日期**: 2024-11-13  
**狀態**: ⚠️ **重要修正** - 發現之前的審查有誤

---

## 🚨 重要更新

**之前的錯誤判斷**: 認為 `app/core/fmri_processing/pipelines/` 目錄未被使用

**實際情況**: `pipelines/` 目錄**仍在使用中**！

**原因**: `generic_pipeline_steps.py` 依賴 `pipelines/` 中的多個模組

---

## ✅ 實際依賴關係

### generic_pipeline_steps.py 使用的模組

```python
# 這些模組都在使用中，不能刪除！
from app.core.fmri_processing.pipelines.inspector import inspect_torch_model
from app.core.fmri_processing.pipelines.choose_layer import select_visualization_layers
from app.core.fmri_processing.pipelines.attach_hook import (
    prepare_model_with_hooks,
    attach_gradient_hooks,
    remove_hooks,
    _gradient_handles
)
from app.core.fmri_processing.pipelines.act_to_nii import activation_and_gradient_to_nifti
from app.core.fmri_processing.pipelines.spatial_normalizer import normalize_native_heatmap_to_mni_accurate_masked
from app.core.fmri_processing.pipelines.resample import resample_activation_to_atlas
from app.core.fmri_processing.pipelines.brain_map import analyze_brain_activation
from app.core.fmri_processing.pipelines.visualize import visualize_gradcam_2d
```

### pipelines/ 目錄狀態

| 檔案 | 狀態 | 被使用於 |
|------|------|---------|
| inspector.py | ✅ 使用中 | generic_pipeline_steps.py |
| choose_layer.py | ✅ 使用中 | generic_pipeline_steps.py |
| attach_hook.py | ✅ 使用中 | generic_pipeline_steps.py |
| act_to_nii.py | ✅ 使用中 | generic_pipeline_steps.py |
| spatial_normalizer.py | ✅ 使用中 | generic_pipeline_steps.py |
| resample.py | ✅ 使用中 | generic_pipeline_steps.py |
| brain_map.py | ✅ 使用中 | generic_pipeline_steps.py |
| visualize.py | ✅ 使用中 | generic_pipeline_steps.py |
| filter_layer.py | ⚠️ 需確認 | 可能未使用 |
| inference.py | ⚠️ 需確認 | 可能未使用 |
| normalize.py | ⚠️ 需確認 | 可能未使用 |
| validate_layer.py | ⚠️ 需確認 | 可能未使用 |

---

## 修正後的未使用程式碼清單

### ❌ 確定未使用（可安全移除）

#### 1. UI 備份檔案
```
app/ui/structural_mri_components_backup.py
```

#### 2. 佔位符檔案
```
app/core/fmri_processing/fmri_model_loader.py
```

#### 3. 資料處理腳本
```
scripts/copy_smri_data.py
scripts/create_mock_model.py
scripts/reorganize_data.py
scripts/rename_data_folders.py
```

#### 4. 探索性測試檔案
```
tests/analyze_vol_act.py
tests/brain_region.py
tests/capsnet_info.py
tests/check_act_shape.py
tests/check_act.py
tests/check_sub14.py
tests/check_time.py
tests/find_t.py
tests/gpt.py
tests/image_explain.py
tests/model_info.py
tests/nii_check.py
tests/nii_dim_check.py
tests/ollama.py
tests/prototype.py
tests/region_network_map.py
tests/update_map_csv.py
tests/vertex_agent.py
tests/vertex.py
```

#### 5. 模型評估腳本
```
model/evaluate_model_accuracy.py
```

### ⚠️ 需要進一步確認

#### pipelines/ 目錄中可能未使用的檔案

需要檢查這些檔案是否被使用：

```
app/core/fmri_processing/pipelines/filter_layer.py
app/core/fmri_processing/pipelines/inference.py
app/core/fmri_processing/pipelines/normalize.py
app/core/fmri_processing/pipelines/validate_layer.py
```

**檢查方法**:
```bash
# 搜尋每個檔案是否被 import
grep -r "filter_layer" app/ --include="*.py"
grep -r "\.inference" app/ --include="*.py"
grep -r "\.normalize" app/ --include="*.py"
grep -r "validate_layer" app/ --include="*.py"
```

---

## 修正後的清理計畫

### 階段 1: 安全清理（立即執行）

只移除**確定未使用**的檔案：

```bash
# 創建 archive 目錄
mkdir -p archive/{ui_backups,placeholder_files,data_scripts,exploratory_tests}

# 移動 UI 備份
mv app/ui/structural_mri_components_backup.py archive/ui_backups/

# 移動佔位符
mv app/core/fmri_processing/fmri_model_loader.py archive/placeholder_files/

# 移動資料腳本
mv scripts/copy_smri_data.py archive/data_scripts/
mv scripts/create_mock_model.py archive/data_scripts/
mv scripts/reorganize_data.py archive/data_scripts/
mv scripts/rename_data_folders.py archive/data_scripts/

# 移動探索性測試
cd tests
mv analyze_vol_act.py brain_region.py capsnet_info.py check_*.py \
   find_t.py gpt.py image_explain.py model_info.py nii_*.py \
   ollama.py prototype.py region_network_map.py update_map_csv.py \
   vertex*.py ../archive/exploratory_tests/ 2>/dev/null
cd ..

# 移動模型評估腳本
mv model/evaluate_model_accuracy.py archive/data_scripts/
```

### 階段 2: 進一步調查（需要確認）

檢查 pipelines/ 中可能未使用的檔案：

```bash
# 檢查 filter_layer.py
grep -r "filter_layer" app/ --include="*.py"
grep -r "from.*pipelines.*filter" app/ --include="*.py"

# 檢查 inference.py
grep -r "pipelines\.inference" app/ --include="*.py"
grep -r "from.*pipelines.*inference" app/ --include="*.py"

# 檢查 normalize.py
grep -r "pipelines\.normalize" app/ --include="*.py"
grep -r "from.*pipelines.*normalize" app/ --include="*.py"

# 檢查 validate_layer.py
grep -r "validate_layer" app/ --include="*.py"
grep -r "from.*pipelines.*validate" app/ --include="*.py"
```

如果這些檔案確實未被使用，可以移動到 archive：

```bash
mkdir -p archive/pipelines_unused
mv app/core/fmri_processing/pipelines/filter_layer.py archive/pipelines_unused/
mv app/core/fmri_processing/pipelines/inference.py archive/pipelines_unused/
mv app/core/fmri_processing/pipelines/normalize.py archive/pipelines_unused/
mv app/core/fmri_processing/pipelines/validate_layer.py archive/pipelines_unused/
```

---

## 更新的清理腳本

我需要更新清理腳本，**不要移動 pipelines/ 目錄**。

---

## 經驗教訓

### 為什麼會出錯？

1. **搜尋方法不夠全面**: 
   - 只搜尋了 `from app.core.fmri_processing.pipelines`
   - 沒有搜尋 `from app.core.fmri_processing.pipelines.xxx`

2. **沒有檢查間接依賴**:
   - 只檢查了直接 import
   - 沒有檢查 `generic_pipeline_steps.py` 的內容

3. **過於依賴自動化**:
   - 應該手動檢查關鍵檔案的內容

### 改進方法

1. **多種搜尋模式**:
```bash
# 搜尋目錄級別的 import
grep -r "from.*pipelines" app/ --include="*.py"
# 搜尋模組級別的 import
grep -r "pipelines\." app/ --include="*.py"
# 搜尋特定檔案名
grep -r "inspector|choose_layer|attach_hook" app/ --include="*.py"
```

2. **檢查關鍵檔案**:
   - 手動檢查 `generic_pipeline_steps.py`
   - 檢查所有 `*_pipeline*.py` 檔案

3. **測試驅動**:
   - 在移動檔案前先測試
   - 使用 Python 的 import 檢查

---

## 正確的檢查方法

### 方法 1: Python Import 測試

```python
# test_imports.py
import sys

try:
    from app.core.fmri_processing.pipelines.inspector import inspect_torch_model
    print("✅ inspector.py is used")
except ImportError as e:
    print(f"❌ inspector.py import failed: {e}")

try:
    from app.core.fmri_processing.pipelines.choose_layer import select_visualization_layers
    print("✅ choose_layer.py is used")
except ImportError as e:
    print(f"❌ choose_layer.py import failed: {e}")

# ... 繼續測試其他模組
```

### 方法 2: 依賴分析工具

```bash
# 使用 pipdeptree 或類似工具
pip install pipdeptree
pipdeptree -p app

# 或使用 modulefinder
python -m modulefinder app.py
```

### 方法 3: 逐步測試

```bash
# 1. 重命名目錄（不刪除）
mv app/core/fmri_processing/pipelines app/core/fmri_processing/pipelines_backup

# 2. 測試應用
python -m streamlit run app.py

# 3. 如果出錯，立即恢復
mv app/core/fmri_processing/pipelines_backup app/core/fmri_processing/pipelines
```

---

## 總結

### ✅ 可以安全移除的檔案（約 25 個）

- UI 備份檔案（1 個）
- 佔位符檔案（1 個）
- 資料處理腳本（4 個）
- 探索性測試（20+ 個）
- 模型評估腳本（1 個）

### ❌ 不能移除的目錄

- **app/core/fmri_processing/pipelines/** - 仍在使用中！

### ⚠️ 需要進一步確認

- pipelines/ 中的 4 個檔案可能未使用

---

**下一步**: 執行修正後的清理腳本

