# 程式碼清理檢查清單

**日期**: 2024-11-12  
**目的**: 逐步確認並清理未使用的程式碼

---

## 使用說明

1. 逐項檢查每個檔案/目錄
2. 在確認後打勾 ✓
3. 記錄任何發現或決定
4. 完成後執行清理腳本

---

## 第一層：確認未使用的目錄

### app/core/fmri_processing/pipelines/

- [ ] **檢查**: 搜尋是否有任何 import
  ```bash
  grep -r "from.*pipelines" app/ --include="*.py"
  grep -r "import.*pipelines" app/ --include="*.py"
  ```

- [ ] **確認**: 功能已被 `generic_pipeline_steps.py` 取代

- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 保留

- [ ] **備註**: 
  ```
  
  
  ```

---

## 第二層：確認未使用的檔案

### app/core/fmri_processing/fmri_model_loader.py

- [ ] **檢查**: 搜尋是否有任何 import
  ```bash
  grep -r "fmri_model_loader" app/ --include="*.py"
  ```

- [ ] **檢視內容**: 確認是否只是佔位符
  ```bash
  cat app/core/fmri_processing/fmri_model_loader.py
  ```

- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 保留

- [ ] **備註**: 
  ```
  
  
  ```

---

### app/ui/structural_mri_components_backup.py

- [ ] **檢查**: 搜尋是否有任何 import
  ```bash
  grep -r "structural_mri_components_backup" app/ --include="*.py"
  ```

- [ ] **確認**: 是否為 `structural_mri_components.py` 的備份

- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 保留

- [ ] **備註**: 
  ```
  
  
  ```

---

## 第三層：資料處理腳本

### scripts/copy_smri_data.py

- [ ] **檢視用途**: 
  ```bash
  head -20 scripts/copy_smri_data.py
  ```

- [ ] **確認**: 是否為一次性腳本

- [ ] **檢查最後修改時間**:
  ```bash
  ls -l scripts/copy_smri_data.py
  ```

- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 保留

- [ ] **備註**: 
  ```
  
  
  ```

---

### scripts/create_mock_model.py

- [ ] **檢視用途**: 
  ```bash
  head -20 scripts/create_mock_model.py
  ```

- [ ] **確認**: 是否為測試用腳本

- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 保留

- [ ] **備註**: 
  ```
  
  
  ```

---

### scripts/reorganize_data.py

- [ ] **檢視用途**: 
  ```bash
  head -20 scripts/reorganize_data.py
  ```

- [ ] **確認**: 是否已完成資料重組

- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 保留

- [ ] **備註**: 
  ```
  
  
  ```

---

### scripts/rename_data_folders.py

- [ ] **檢視用途**: 
  ```bash
  head -20 scripts/rename_data_folders.py
  ```

- [ ] **確認**: 是否已完成資料夾重命名

- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 保留

- [ ] **備註**: 
  ```
  
  
  ```

---

## 第四層：測試/除錯檔案

### tests/ 目錄中的探索性檔案

對每個檔案進行檢查：

#### tests/analyze_vol_act.py
- [ ] **檢視**: `head -10 tests/analyze_vol_act.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/brain_region.py
- [ ] **檢視**: `head -10 tests/brain_region.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/capsnet_info.py
- [ ] **檢視**: `head -10 tests/capsnet_info.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/check_act_shape.py
- [ ] **檢視**: `head -10 tests/check_act_shape.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/check_act.py
- [ ] **檢視**: `head -10 tests/check_act.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/check_sub14.py
- [ ] **檢視**: `head -10 tests/check_sub14.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/check_time.py
- [ ] **檢視**: `head -10 tests/check_time.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/find_t.py
- [ ] **檢視**: `head -10 tests/find_t.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/gpt.py
- [ ] **檢視**: `head -10 tests/gpt.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/image_explain.py
- [ ] **檢視**: `head -10 tests/image_explain.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/model_info.py
- [ ] **檢視**: `head -10 tests/model_info.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/nii_check.py
- [ ] **檢視**: `head -10 tests/nii_check.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/nii_dim_check.py
- [ ] **檢視**: `head -10 tests/nii_dim_check.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/ollama.py
- [ ] **檢視**: `head -10 tests/ollama.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/prototype.py
- [ ] **檢視**: `head -10 tests/prototype.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/region_network_map.py
- [ ] **檢視**: `head -10 tests/region_network_map.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/update_map_csv.py
- [ ] **檢視**: `head -10 tests/update_map_csv.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/vertex_agent.py
- [ ] **檢視**: `head -10 tests/vertex_agent.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

#### tests/vertex.py
- [ ] **檢視**: `head -10 tests/vertex.py`
- [ ] **決定**: ☐ 刪除  ☐ 移至 archive  ☐ 整合到正式測試  ☐ 保留

---

## 第五層：其他檔案

### model/evaluate_model_accuracy.py

- [ ] **檢視用途**: 
  ```bash
  head -20 model/evaluate_model_accuracy.py
  ```

- [ ] **確認**: 是否為常用腳本

- [ ] **決定**: ☐ 刪除  ☐ 移至 scripts/  ☐ 移至 archive  ☐ 保留

- [ ] **備註**: 
  ```
  
  
  ```

---

## 執行清理

### 準備工作

- [ ] **備份當前狀態**:
  ```bash
  git add -A
  git commit -m "Backup before cleanup"
  git tag backup-before-cleanup-$(date +%Y%m%d)
  ```

- [ ] **確認所有檢查項目已完成**

- [ ] **確認決定已記錄**

### 執行清理腳本

選擇適合你的作業系統的腳本：

#### Linux/Mac:
```bash
chmod +x scripts/cleanup_unused_code.sh
./scripts/cleanup_unused_code.sh
```

#### Windows:
```cmd
scripts\cleanup_unused_code.bat
```

### 驗證

- [ ] **執行測試**:
  ```bash
  python -m pytest tests/test_*.py
  ```

- [ ] **測試 Streamlit 應用**:
  ```bash
  streamlit run app.py
  ```

- [ ] **檢查是否有 import 錯誤**

- [ ] **檢查功能是否正常**

### 提交變更

- [ ] **檢視變更**:
  ```bash
  git status
  git diff
  ```

- [ ] **提交**:
  ```bash
  git add -A
  git commit -m "Clean up unused code - moved to archive/"
  ```

- [ ] **推送** (如果需要):
  ```bash
  git push origin main
  ```

---

## 恢復計畫

如果清理後發現問題：

### 選項 1: 從 archive 恢復特定檔案

```bash
# 恢復特定檔案
cp archive/old_pipelines/pipelines/inference.py app/core/fmri_processing/pipelines/
```

### 選項 2: 使用 Git 標籤恢復

```bash
# 查看備份標籤
git tag -l 'backup-before-cleanup-*'

# 恢復到備份點
git checkout <tag-name>

# 或創建新分支
git checkout -b restore-from-backup <tag-name>
```

### 選項 3: 使用 Git 恢復特定檔案

```bash
# 恢復特定檔案到之前的版本
git checkout HEAD~1 -- path/to/file.py
```

---

## 完成確認

- [ ] 所有檢查項目已完成
- [ ] 清理腳本已執行
- [ ] 測試已通過
- [ ] 變更已提交
- [ ] 文件已更新（如需要）

---

**簽名**: ________________  
**日期**: ________________

