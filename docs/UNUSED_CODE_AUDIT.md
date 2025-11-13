# 未使用程式碼審查報告

**審查日期**: 2024-11-12  
**目的**: 識別並記錄專案中未被使用的程式碼檔案

---

## 執行摘要

本次審查發現以下類別的未使用程式碼：
1. **舊版 Pipeline 系統** - 已被 generic_pipeline_steps.py 取代
2. **備份檔案** - UI 組件的備份版本
3. **獨立腳本** - 一次性使用的資料處理腳本
4. **測試/除錯檔案** - 開發過程中的臨時測試檔案

---

## 第一層：app/core/fmri_processing/pipelines/ 目錄

### 狀態：❌ **整個目錄未被使用**

**原因**: 已被 `generic_pipeline_steps.py` 取代

**檔案清單**:
```
app/core/fmri_processing/pipelines/
├── __init__.py
├── act_to_nii.py
├── attach_hook.py
├── brain_map.py
├── choose_layer.py
├── filter_layer.py
├── inference.py
├── inspector.py
├── normalize.py
├── resample.py
├── spatial_normalizer.py
├── validate_layer.py
└── visualize.py
```

**建議**: 
- ✅ **可以安全刪除整個目錄**
- 功能已整合到 `generic_pipeline_steps.py`
- 如需保留歷史記錄，可移至 `archive/` 目錄

---

## 第二層：app/core/fmri_processing/fmri_model_loader.py

### 狀態：❌ **未被使用**

**檢查結果**:
- 無任何檔案 import 此模組
- 功能可能已被 `model_config.py` 中的 adapter 系統取代

**建議**:
- ⚠️ **需要確認** - 檢查是否有遺留功能
- 如確認無用，可刪除

---

## 第三層：app/ui/structural_mri_components_backup.py

### 狀態：❌ **備份檔案，未被使用**

**檢查結果**:
- 無任何檔案 import 此模組
- 明顯是 `structural_mri_components.py` 的備份

**建議**:
- ✅ **可以安全刪除**
- 如需保留，移至 `archive/` 或使用 git 歷史記錄

---

## 第四層：scripts/ 目錄中的獨立腳本

### 狀態：⚠️ **獨立腳本，不被其他程式碼 import**

這些是獨立執行的腳本，不被其他程式碼 import，但可能仍有用途：


**資料處理腳本**:
```
scripts/
├── copy_smri_data.py          - 複製 sMRI 資料
├── create_mock_model.py       - 建立模擬模型（測試用）
├── rename_data_folders.py     - 重新命名資料夾
└── reorganize_data.py         - 重組資料結構
```

**建議**:
- 🔍 **需要確認** - 這些是一次性使用的腳本嗎？
- 如果已完成資料處理，可移至 `scripts/archive/` 或 `scripts/data_migration/`
- 如果未來可能需要，保留但加上文件說明

---

## 第五層：tests/ 目錄中的除錯/探索性檔案

### 狀態：⚠️ **開發過程中的臨時測試檔案**

**可能未使用的測試檔案**:
```
tests/
├── analyze_vol_act.py         - 分析體積活化
├── brain_region.py            - 腦區測試
├── capsnet_info.py            - CapsNet 資訊
├── check_act_shape.py         - 檢查活化形狀
├── check_act.py               - 檢查活化
├── check_sub14.py             - 檢查特定受試者
├── check_time.py              - 時間檢查
├── find_t.py                  - 尋找 t 值
├── gpt.py                     - GPT 測試
├── image_explain.py           - 圖像解釋測試
├── model_info.py              - 模型資訊
├── nii_check.py               - NIfTI 檢查
├── nii_dim_check.py           - NIfTI 維度檢查
├── ollama.py                  - Ollama 測試
├── prototype.py               - 原型測試
├── region_network_map.py      - 區域網絡映射
├── update_map_csv.py          - 更新映射 CSV
├── vertex_agent.py            - Vertex agent 測試
└── vertex.py                  - Vertex 測試
```

**正式測試檔案（應保留）**:
```
tests/
├── test_activation_extractor.py
├── test_app_integration.py
├── test_complete_pipeline.py
├── test_dynamic_filtering.py
├── test_layer_selection_flexibility.py
├── test_ml_model_loader.py
├── test_nilearn_activation.py
├── test_roi_feature_extractor.py
├── test_structural_agents.py
└── test_structural_workflow_integration.py
```

**建議**:
- 🔍 **需要確認** - 這些除錯檔案是否還需要？
- 建議：
  - 刪除明顯過時的檔案（如 `check_sub14.py`）
  - 將有用的測試整合到正式測試檔案中
  - 移動探索性程式碼到 `tests/exploratory/` 或刪除

---

## 第六層：model/ 目錄中的評估腳本

### 狀態：⚠️ **獨立腳本**

```
model/
└── evaluate_model_accuracy.py
```

**建議**:
- 🔍 **需要確認** - 這是否為常用的評估腳本？
- 如果是，應移至 `scripts/evaluation/`
- 如果不常用，可刪除或移至 archive

---

## 清理建議優先順序

### 🔴 高優先級（可立即刪除）

1. **app/ui/structural_mri_components_backup.py**
   - 明確的備份檔案
   - 無任何引用

2. **app/core/fmri_processing/pipelines/** (整個目錄)
   - 已被新系統取代
   - 無任何引用
   - 功能已遷移

### 🟡 中優先級（需要確認後刪除）

3. **app/core/fmri_processing/fmri_model_loader.py**
   - 檢查是否有遺留功能
   - 確認後可刪除

4. **tests/ 中的除錯檔案**
   - 逐一檢查是否還需要
   - 大部分可能可以刪除

### 🟢 低優先級（整理歸檔）

5. **scripts/ 中的資料處理腳本**
   - 移至 `scripts/archive/` 或 `scripts/data_migration/`
   - 加上 README 說明用途和執行時間

6. **model/evaluate_model_accuracy.py**
   - 移至適當位置或刪除

---

## 建議的目錄結構調整

### 創建 archive 目錄

```bash
mkdir -p archive/old_pipelines
mkdir -p archive/data_scripts
mkdir -p archive/exploratory_tests
```

### 移動檔案而非刪除（保險做法）

```bash
# 移動舊 pipeline
mv app/core/fmri_processing/pipelines archive/old_pipelines/

# 移動備份檔案
mv app/ui/structural_mri_components_backup.py archive/

# 移動資料處理腳本
mv scripts/copy_smri_data.py archive/data_scripts/
mv scripts/reorganize_data.py archive/data_scripts/
mv scripts/rename_data_folders.py archive/data_scripts/

# 移動探索性測試
mv tests/check_*.py archive/exploratory_tests/
mv tests/nii_*.py archive/exploratory_tests/
```

---

## 執行清理的步驟

### 步驟 1: 備份（安全第一）

```bash
# 創建完整備份
git add -A
git commit -m "Backup before cleanup"
git tag backup-before-cleanup-2024-11-12
```

### 步驟 2: 創建 archive 目錄

```bash
mkdir -p archive/{old_pipelines,data_scripts,exploratory_tests,ui_backups}
```

### 步驟 3: 移動檔案到 archive

```bash
# UI 備份
mv app/ui/structural_mri_components_backup.py archive/ui_backups/

# 舊 pipeline 系統
mv app/core/fmri_processing/pipelines archive/old_pipelines/

# 資料處理腳本
mv scripts/copy_smri_data.py archive/data_scripts/
mv scripts/create_mock_model.py archive/data_scripts/
mv scripts/reorganize_data.py archive/data_scripts/
mv scripts/rename_data_folders.py archive/data_scripts/
```

### 步驟 4: 測試系統

```bash
# 執行所有測試確保沒有破壞
python -m pytest tests/test_*.py

# 測試主要功能
streamlit run app.py
```

### 步驟 5: 提交變更

```bash
git add -A
git commit -m "Clean up unused code - moved to archive/"
```

---

## 檢查清單

使用此清單逐一確認：

- [ ] 已備份當前狀態（git tag）
- [ ] 已創建 archive 目錄結構
- [ ] 已移動 `structural_mri_components_backup.py`
- [ ] 已移動 `pipelines/` 目錄
- [ ] 已移動資料處理腳本
- [ ] 已檢查 `fmri_model_loader.py` 是否可刪除
- [ ] 已整理 tests/ 目錄
- [ ] 已執行測試確認系統正常
- [ ] 已提交變更到 git
- [ ] 已更新 README（如需要）

---

## 預期效果

清理後的專案結構將：
- ✅ 更清晰易懂
- ✅ 減少混淆（沒有重複/過時的程式碼）
- ✅ 更容易維護
- ✅ 減少專案大小
- ✅ 保留歷史記錄（在 archive/ 和 git history）

---

## 附錄：詳細檢查命令

### 檢查檔案是否被 import

```bash
# 檢查特定檔案是否被使用
grep -r "from.*pipelines" app/ --include="*.py"
grep -r "import.*pipelines" app/ --include="*.py"

# 檢查 fmri_model_loader
grep -r "fmri_model_loader" app/ --include="*.py"

# 檢查備份檔案
grep -r "structural_mri_components_backup" app/ --include="*.py"
```

### 檢查 __pycache__ 時間戳

```bash
# 查看最近編譯的檔案（表示最近被使用）
find . -name "*.pyc" -type f -mtime -7  # 最近 7 天
```

---

**報告結束**

如有疑問或需要進一步確認，請參考此報告並逐項檢查。
