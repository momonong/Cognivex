# 程式碼清理總結報告

**審查日期**: 2024-11-12  
**審查者**: AI Assistant  
**專案**: Cognivex - Explainable AI Framework for MRI Analysis

---

## 📊 執行摘要

本次審查識別出 **4 個主要類別** 的未使用程式碼，總計約 **30+ 個檔案**。

### 統計數據

| 類別 | 檔案數量 | 建議動作 | 優先級 |
|------|---------|---------|--------|
| 舊版 Pipeline 系統 | 13 個檔案 | 移至 archive | 🔴 高 |
| 備份檔案 | 1 個檔案 | 移至 archive | 🔴 高 |
| 佔位符檔案 | 1 個檔案 | 移至 archive | 🟡 中 |
| 資料處理腳本 | 4-5 個檔案 | 移至 archive | 🟢 低 |
| 探索性測試 | 20+ 個檔案 | 移至 archive | 🟢 低 |

---

## 🎯 主要發現

### 1. 舊版 Pipeline 系統（高優先級）

**位置**: `app/core/fmri_processing/pipelines/`

**狀態**: ❌ 完全未使用

**原因**: 
- 已被 `generic_pipeline_steps.py` 完全取代
- 無任何檔案 import 此目錄
- 功能已整合到新系統

**影響**:
- 佔用空間：~13 個檔案
- 造成混淆：開發者可能不知道使用哪個系統
- 維護負擔：需要同時維護兩套系統

**建議**: ✅ **立即移至 archive/**

---

### 2. UI 備份檔案（高優先級）

**位置**: `app/ui/structural_mri_components_backup.py`

**狀態**: ❌ 未使用

**原因**:
- 明確的備份檔案（檔名包含 `_backup`）
- 無任何 import
- 正式版本為 `structural_mri_components.py`

**建議**: ✅ **立即移至 archive/**

---

### 3. 佔位符檔案（中優先級）

**位置**: `app/core/fmri_processing/fmri_model_loader.py`

**狀態**: ⚠️ 只有佔位符實作

**內容**:
```python
def get_model_and_input_shape(...):
    # TODO: 實作實際的模型載入邏輯
    return None, None
```

**原因**:
- 只有空函數和 TODO 註解
- 無實際功能
- 功能已在 `model_config.py` 中實作

**建議**: ⚠️ **確認後移至 archive/**

---

### 4. 資料處理腳本（低優先級）

**位置**: `scripts/`

**檔案**:
- `copy_smri_data.py` - 複製 sMRI 資料
- `create_mock_model.py` - 建立模擬模型
- `reorganize_data.py` - 重組資料結構
- `rename_data_folders.py` - 重新命名資料夾

**狀態**: ⚠️ 獨立腳本，不被其他程式碼 import

**特性**:
- 一次性使用的資料遷移腳本
- 可能已完成其任務
- 未來可能不再需要

**建議**: 🔍 **確認後移至 archive/data_scripts/**

---

### 5. 探索性測試檔案（低優先級）

**位置**: `tests/`

**類型**:
- 除錯檔案：`check_*.py`, `nii_*.py`
- 探索性測試：`prototype.py`, `gpt.py`
- 特定測試：`check_sub14.py`
- LLM 測試：`vertex.py`, `ollama.py`

**狀態**: ⚠️ 開發過程中的臨時檔案

**特性**:
- 不是正式測試套件的一部分
- 檔名不符合 `test_*.py` 慣例
- 可能是一次性除錯用途

**建議**: 🔍 **逐一確認後移至 archive/exploratory_tests/**

---

## 📁 建議的清理結構

```
專案根目錄/
├── app/                          # 保持乾淨
│   ├── agents/                   # ✅ 所有檔案都在使用
│   ├── core/
│   │   ├── fmri_processing/
│   │   │   ├── generic_pipeline_steps.py  # ✅ 使用中
│   │   │   ├── model_config.py            # ✅ 使用中
│   │   │   └── pipelines/                 # ❌ 移除
│   │   ├── ml_processing/        # ✅ 所有檔案都在使用
│   │   └── ...
│   └── ui/
│       ├── structural_mri_components.py   # ✅ 使用中
│       └── *_backup.py                    # ❌ 移除
│
├── scripts/                      # 整理
│   ├── (保留常用腳本)
│   └── (移除一次性腳本)
│
├── tests/                        # 整理
│   ├── test_*.py                 # ✅ 保留正式測試
│   └── (其他)                    # ❌ 移除探索性檔案
│
└── archive/                      # 新增
    ├── old_pipelines/
    ├── data_scripts/
    ├── exploratory_tests/
    ├── ui_backups/
    └── placeholder_files/
```

---

## 🚀 執行計畫

### 階段 1: 準備（5 分鐘）

1. ✅ 閱讀審查報告
2. ✅ 確認 Git 狀態乾淨
3. ✅ 創建備份標籤

```bash
git add -A
git commit -m "Backup before cleanup"
git tag backup-before-cleanup-$(date +%Y%m%d)
```

### 階段 2: 執行清理（10 分鐘）

**選項 A: 自動清理（推薦）**

Linux/Mac:
```bash
chmod +x scripts/cleanup_unused_code.sh
./scripts/cleanup_unused_code.sh
```

Windows:
```cmd
scripts\cleanup_unused_code.bat
```

**選項 B: 手動清理**

使用 `docs/CLEANUP_CHECKLIST.md` 逐項檢查

### 階段 3: 驗證（15 分鐘）

1. 執行測試套件
```bash
python -m pytest tests/test_*.py -v
```

2. 測試 Streamlit 應用
```bash
streamlit run app.py
```

3. 檢查功能
   - [ ] fMRI 分析正常
   - [ ] sMRI 分析正常
   - [ ] 報告生成正常
   - [ ] 視覺化正常

### 階段 4: 提交（5 分鐘）

```bash
git add -A
git commit -m "Clean up unused code - moved to archive/

- Moved old pipelines system to archive/old_pipelines/
- Moved backup files to archive/ui_backups/
- Moved data scripts to archive/data_scripts/
- Moved exploratory tests to archive/exploratory_tests/
- Moved placeholder files to archive/placeholder_files/

All functionality preserved in archive/ for reference.
Backup tag: backup-before-cleanup-YYYYMMDD"

git push origin main  # 如果需要
```

---

## 📈 預期效果

### 程式碼品質改善

| 指標 | 清理前 | 清理後 | 改善 |
|------|--------|--------|------|
| 總檔案數 | ~150 | ~120 | -20% |
| 混淆程度 | 高 | 低 | ✅ |
| 維護負擔 | 高 | 低 | ✅ |
| 新手友善度 | 中 | 高 | ✅ |

### 具體改善

1. **更清晰的結構**
   - 移除重複/過時的程式碼
   - 只保留實際使用的檔案
   - 更容易理解系統架構

2. **減少混淆**
   - 不再有兩套 pipeline 系統
   - 沒有備份檔案造成困惑
   - 測試目錄更整潔

3. **更容易維護**
   - 減少需要維護的程式碼
   - 更容易找到相關檔案
   - 降低出錯機率

4. **保留歷史**
   - 所有檔案移至 archive/
   - Git 歷史完整保留
   - 隨時可以恢復

---

## 🔄 恢復策略

### 如果需要恢復某個檔案

**方法 1: 從 archive 複製**
```bash
cp archive/old_pipelines/pipelines/inference.py app/core/fmri_processing/pipelines/
```

**方法 2: 使用 Git 標籤**
```bash
git checkout backup-before-cleanup-YYYYMMDD -- path/to/file.py
```

**方法 3: 完全恢復**
```bash
git checkout backup-before-cleanup-YYYYMMDD
```

---

## ✅ 檢查清單

### 清理前

- [ ] 已閱讀完整審查報告
- [ ] 已理解每個檔案的狀態
- [ ] 已創建 Git 備份標籤
- [ ] 已確認 Git 狀態乾淨

### 清理中

- [ ] 已執行清理腳本或手動清理
- [ ] 已檢查 archive/ 目錄結構
- [ ] 已確認檔案已正確移動

### 清理後

- [ ] 已執行測試套件（全部通過）
- [ ] 已測試 Streamlit 應用（正常運作）
- [ ] 已檢查主要功能（fMRI/sMRI 分析）
- [ ] 已提交變更到 Git
- [ ] 已更新相關文件（如需要）

---

## 📚 相關文件

1. **UNUSED_CODE_AUDIT.md** - 詳細的審查報告
2. **CLEANUP_CHECKLIST.md** - 逐項檢查清單
3. **cleanup_unused_code.sh** - Linux/Mac 清理腳本
4. **cleanup_unused_code.bat** - Windows 清理腳本

---

## 💡 建議

### 短期（本次清理）

1. ✅ 執行自動清理腳本
2. ✅ 驗證系統功能
3. ✅ 提交變更

### 中期（未來 1-2 週）

1. 監控是否有遺漏的依賴
2. 確認沒有功能受影響
3. 如果一切正常，可考慮刪除 archive/（保留在 Git 歷史）

### 長期（未來 1-2 個月）

1. 建立程式碼審查流程
2. 定期清理未使用的程式碼
3. 使用 linter 檢測未使用的 import
4. 建立程式碼品質指標

---

## 🎓 經驗教訓

### 避免未來累積未使用程式碼

1. **使用版本控制**
   - 不要創建 `*_backup.py` 檔案
   - 使用 Git 分支和標籤

2. **及時清理**
   - 重構後立即刪除舊程式碼
   - 不要「以防萬一」保留

3. **明確命名**
   - 測試檔案使用 `test_*.py`
   - 腳本放在適當目錄

4. **文件記錄**
   - 記錄為什麼保留某個檔案
   - 記錄何時可以刪除

---

**報告完成日期**: 2024-11-12  
**下次審查建議**: 2025-02-12（3 個月後）

