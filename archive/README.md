# Archive 目錄

此目錄包含已從主要程式碼庫中移除但保留以供參考的檔案。

## 重要說明

**pipelines/ 和 fmri_model_loader.py 已恢復**: 經過測試發現這些檔案仍在使用中，
已恢復到原位置 `app/core/fmri_processing/`。

## 目錄結構

- **data_scripts/**: 一次性資料處理腳本
  - 用於資料遷移和重組
  - 通常只執行一次
  - 包含模型評估腳本

- **exploratory_tests/**: 探索性和除錯測試檔案
  - 開發過程中的臨時測試
  - 不是正式測試套件的一部分
  - 約 20 個檔案

- **ui_backups/**: UI 組件的備份檔案
  - 開發過程中的備份版本

## 已移除的檔案清單

### 資料處理腳本 (5 個)
- copy_smri_data.py
- create_mock_model.py
- reorganize_data.py
- rename_data_folders.py
- evaluate_model_accuracy.py

### 探索性測試 (20 個)
- analyze_vol_act.py
- brain_region.py
- capsnet_info.py
- check_act.py
- check_act_shape.py
- check_sub14.py
- check_time.py
- find_t.py
- gpt.py
- image_explain.py
- model_info.py
- nii_check.py
- nii_dim_check.py
- ollama.py
- prototype.py
- region_network_map.py
- update_map_csv.py
- vertex.py
- vertex_agent.py

### UI 備份 (1 個)
- structural_mri_components_backup.py

## 恢復檔案

如需恢復任何檔案，可以從此目錄複製回原位置，或使用 git 歷史記錄。

## 清理日期

檔案移至此目錄的日期: 2024-11-13

## Git 備份標籤

在清理前創建的備份標籤: backup-before-cleanup-*
