# 快速清理指南

**5 分鐘快速開始** 🚀

---

## TL;DR

```bash
# 1. 備份
git add -A && git commit -m "Backup before cleanup" && git tag backup-$(date +%Y%m%d)

# 2. 清理（選擇你的系統）
# Linux/Mac:
./scripts/cleanup_unused_code.sh

# Windows:
scripts\cleanup_unused_code.bat

# 3. 測試
python -m pytest tests/test_*.py
streamlit run app.py

# 4. 提交
git add -A && git commit -m "Clean up unused code"
```

---

## 將被移除的檔案

### ❌ 確定移除（30+ 個檔案）

```
app/core/fmri_processing/pipelines/          (整個目錄)
app/ui/structural_mri_components_backup.py
app/core/fmri_processing/fmri_model_loader.py
scripts/copy_smri_data.py
scripts/create_mock_model.py
scripts/reorganize_data.py
scripts/rename_data_folders.py
tests/check_*.py                              (多個檔案)
tests/nii_*.py                                (多個檔案)
tests/analyze_vol_act.py
tests/brain_region.py
tests/capsnet_info.py
tests/find_t.py
tests/gpt.py
tests/image_explain.py
tests/model_info.py
tests/ollama.py
tests/prototype.py
tests/region_network_map.py
tests/update_map_csv.py
tests/vertex*.py
model/evaluate_model_accuracy.py
```

### ✅ 保留的檔案

```
app/agents/*                    (所有 agent 檔案)
app/core/fmri_processing/generic_pipeline_steps.py
app/core/fmri_processing/model_config.py
app/core/ml_processing/*        (所有 ML 處理檔案)
app/ui/structural_mri_components.py
tests/test_*.py                 (所有正式測試)
```

---

## 安全保證

✅ **所有檔案移至 `archive/` 而非刪除**  
✅ **Git 備份標籤已創建**  
✅ **可隨時恢復**  
✅ **不影響系統功能**

---

## 如果出問題

### 恢復單一檔案
```bash
cp archive/old_pipelines/pipelines/inference.py app/core/fmri_processing/pipelines/
```

### 完全恢復
```bash
git checkout backup-YYYYMMDD
```

---

## 詳細文件

- 📋 **CLEANUP_SUMMARY.md** - 完整總結報告
- 📝 **UNUSED_CODE_AUDIT.md** - 詳細審查報告
- ✅ **CLEANUP_CHECKLIST.md** - 逐項檢查清單

---

**準備好了嗎？執行清理腳本！** 🎯
