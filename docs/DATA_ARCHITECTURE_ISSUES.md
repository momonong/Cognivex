# 資料架構問題分析報告

**建立日期**: 2024-11-21  
**嚴重程度**: 🔴 高 - 系統無法正確讀取資料

---

## 問題總結

你的程式碼中的資料讀取邏輯與實際的資料目錄結構**不匹配**，導致系統無法正確找到和載入 MRI 檔案。

---

## 實際資料結構

### 1. fMRI 資料 (`data/fMRI/`)
```
data/fMRI/
├── AD/
│   └── sub-07/                    ← 注意：沒有連字號
│       └── dswausub-027_S_6648_task-rest_bold.nii.gz
└── CN/                            ← 注意：是 CN 不是 NC
    └── sub-01/
        └── *.nii.gz
```

**關鍵特徵**:
- 受試者 ID: `sub-07`, `sub-01` (沒有連字號 `-`)
- 標籤: `CN` (Control Normal) 不是 `NC`
- 檔案命名: `dswausub-027_S_6648_task-rest_bold.nii.gz`

### 2. sMRI 資料 (`data/sMRI/`)
```
data/sMRI/
├── AD/
│   └── sub-0005/                  ← 注意：有連字號
│       ├── sub_0005_T1.nii.gz    ← 注意：檔名用底線
│       ├── sub_0005_DWI.nii.gz
│       └── sub_0005_T2_FLAIR.nii.gz
├── MCI/
│   └── sub-XXXX/
└── NC/                            ← 注意：是 NC 不是 CN
    └── sub-0001/
```

**關鍵特徵**:
- 受試者 ID: `sub-0005` (有連字號 `-`)
- 檔案命名: `sub_0005_T1.nii.gz` (底線 `_`)
- 標籤: `NC` (Normal Control)
- 有三種類別: AD, MCI, NC

### 3. MRI_processed 資料 (`data/MRI_processed/`)
```
data/MRI_processed/
├── AD/
│   └── sub-0005/                  ← 注意：有連字號
│       ├── sub-0005_GM_to_MNI.nii.gz    ← 注意：檔名也用連字號
│       ├── sub-0005_FA_to_MNI.nii.gz
│       └── sub-0005_MD_to_MNI.nii.gz
├── MCI/
└── NC/
```

**關鍵特徵**:
- 受試者 ID: `sub-0005` (有連字號 `-`)
- 檔案命名: `sub-0005_GM_to_MNI.nii.gz` (連字號 `-`)
- 三種模態: GM (灰質), FA (各向異性分數), MD (平均擴散率)

---

## 程式碼問題分析

### 問題 1: app.py - fMRI 資料讀取錯誤

**程式碼位置**: `app.py` 第 133-145 行

```python
# 功能性 MRI: 使用 data/fMRI（子資料夾結構）
fmri_folders = glob.glob("data/fMRI/*/sub-*")
for folder_path in fmri_folders:
    parts = folder_path.split(os.sep)
    if len(parts) >= 3:
        subject_id = parts[-1]
        label = parts[-2]
        # 處理 CN -> NC 的標籤轉換
        if label == "CN":
            label = "NC"
        subject_labels[subject_id] = label
```

**問題**:
1. ✅ `glob.glob("data/fMRI/*/sub-*")` - 這個可以找到 `sub-07`, `sub-01` 等
2. ❌ 標籤轉換 `CN -> NC` - **方向錯誤**！實際資料是 `CN`，但你要顯示為 `NC`
3. ❌ 後續搜尋檔案時使用錯誤的路徑

**實際執行結果**:
- 找到: `data/fMRI/CN/sub-01` → 標籤變成 `NC`
- 但實際路徑是 `data/fMRI/CN/...` 不是 `data/fMRI/NC/...`

### 問題 2: app.py - fMRI 檔案搜尋錯誤

**程式碼位置**: `app.py` 第 234-240 行

```python
# 功能性 MRI: 從 data/fMRI 搜尋檔案
nii_search_pattern = f"data/fMRI/*/{selected_subject}/*.nii.gz"
nii_file_list = glob.glob(nii_search_pattern)
if not nii_file_list:
    raise FileNotFoundError(
        f"找不到受試者 '{selected_subject}' 的 .nii.gz 檔案。\n"
        f"搜尋路徑: {nii_search_pattern}"
    )
```

**問題**:
- `selected_subject` = `sub-01` (來自 UI 選擇)
- 搜尋模式: `data/fMRI/*/sub-01/*.nii.gz`
- 實際路徑: `data/fMRI/CN/sub-01/dswausub-027_S_6648_task-rest_bold.nii.gz`
- ✅ 這個應該可以找到

**但是**，如果 `ground_truth_label` 已經被轉換成 `NC`，後續邏輯可能會出錯。

### 問題 3: app.py - sMRI 資料讀取錯誤

**程式碼位置**: `app.py` 第 119-131 行

```python
# 結構性 MRI: 使用 data/sMRI（子資料夾結構）
smri_folders = glob.glob("data/sMRI/*/sub-*")
for folder_path in smri_folders:
    parts = folder_path.split(os.sep)
    if len(parts) >= 3:
        subject_id = parts[-1]  # sub-0005
        label = parts[-2]  # AD or NC
        # 統一格式為 sub_XXXX
        subject_id_normalized = subject_id.replace("-", "_")
        subject_labels[subject_id_normalized] = label
```

**問題**:
1. ✅ 找到 `sub-0005`
2. ❌ **轉換成 `sub_0005`** - 這是錯的！
3. ❌ UI 顯示 `sub_0005`，但實際資料夾是 `sub-0005`

**後果**:
- UI 選擇: `sub_0005`
- 搜尋路徑: `data/sMRI/AD/sub_0005/*_T1.nii.gz` ← **找不到！**
- 實際路徑: `data/sMRI/AD/sub-0005/sub_0005_T1.nii.gz`

### 問題 4: app.py - sMRI 檔案搜尋錯誤

**程式碼位置**: `app.py` 第 217-228 行

```python
# 結構性 MRI: 從 data/sMRI 搜尋 T1 檔案
label = ground_truth_label
# 將 sub_XXXX 轉換為 sub-XXXX（資料夾格式）
subject_folder = selected_subject.replace("_", "-")
nii_search_pattern = f"data/sMRI/{label}/{subject_folder}/*_T1.nii.gz"
nii_file_list = glob.glob(nii_search_pattern)
```

**問題**:
- `selected_subject` = `sub_0005` (來自 UI)
- `subject_folder` = `sub-0005` (轉換回來)
- 搜尋模式: `data/sMRI/AD/sub-0005/*_T1.nii.gz`
- 實際檔案: `data/sMRI/AD/sub-0005/sub_0005_T1.nii.gz`
- ✅ 這個可以找到

**但是**，為什麼要來回轉換？這增加了複雜度和出錯機會。

### 問題 5: app_cdda.py - MRI_processed 資料讀取錯誤

**程式碼位置**: `app_cdda.py` 第 329-335 行

```python
# 掃描可用的受試者
subject_labels = {}
# 修正：使用 sub-* 而不是 sub_*（匹配實際的目錄命名格式）
data_folders = glob.glob("data/MRI_processed/*/sub-*")
for folder_path in data_folders:
    parts = folder_path.split(os.sep)
    if len(parts) >= 3:
        subject_id = parts[-1]  # sub-0001
        label = parts[-2]  # AD, MCI, or NC
        subject_labels[subject_id] = label
```

**問題**:
1. ✅ 找到 `sub-0005`
2. ✅ 保持 `sub-0005` 格式
3. ✅ 標籤正確 (AD, MCI, NC)

**這部分是對的！**

---


## 根本問題總結

### 🔴 嚴重問題

1. **命名不一致**:
   - fMRI: `sub-07` (無連字號，兩位數)
   - sMRI: `sub-0005` (有連字號，四位數)
   - MRI_processed: `sub-0005` (有連字號，四位數)

2. **標籤不一致**:
   - fMRI: 使用 `CN` (Control Normal)
   - sMRI: 使用 `NC` (Normal Control)
   - MRI_processed: 使用 `NC`

3. **檔案命名不一致**:
   - sMRI 資料夾: `sub-0005` (連字號)
   - sMRI 檔案: `sub_0005_T1.nii.gz` (底線)
   - MRI_processed 資料夾: `sub-0005` (連字號)
   - MRI_processed 檔案: `sub-0005_GM_to_MNI.nii.gz` (連字號)

4. **不必要的轉換**:
   - `app.py` 將 `sub-0005` 轉換成 `sub_0005` 顯示在 UI
   - 然後又轉換回 `sub-0005` 去搜尋檔案
   - 這增加了複雜度和出錯風險

### ⚠️ 中等問題

5. **fMRI 和 sMRI 受試者 ID 不重疊**:
   - fMRI: `sub-01` 到 `sub-32`
   - sMRI: `sub-0001` 到 `sub-0142`
   - 這意味著它們是**不同的資料集**

6. **MRI_processed 和 sMRI 的關係不明確**:
   - 兩者有相同的受試者 (如 `sub-0005`)
   - 但檔案內容不同 (原始 vs 處理後)
   - 程式碼沒有明確說明何時使用哪個

---

## 建議的修正方案

### 方案 A: 最小修改 (推薦)

**目標**: 保持資料不變，只修改程式碼

#### 1. 修正 app.py - sMRI 部分

```python
# 不要轉換 subject_id，保持原始格式
smri_folders = glob.glob("data/sMRI/*/sub-*")
for folder_path in smri_folders:
    parts = folder_path.split(os.sep)
    if len(parts) >= 3:
        subject_id = parts[-1]  # 保持 sub-0005 格式
        label = parts[-2]
        subject_labels[subject_id] = label  # 不轉換
```

#### 2. 修正 app.py - fMRI 標籤處理

```python
# 保持原始標籤，不轉換
fmri_folders = glob.glob("data/fMRI/*/sub-*")
for folder_path in fmri_folders:
    parts = folder_path.split(os.sep)
    if len(parts) >= 3:
        subject_id = parts[-1]
        label = parts[-2]  # 保持 CN，不轉換成 NC
        subject_labels[subject_id] = label
```

#### 3. 修正檔案搜尋邏輯

```python
# sMRI 檔案搜尋
if st.session_state.analysis_mode == "structural":
    label = ground_truth_label
    # 不需要轉換，直接使用
    nii_search_pattern = f"data/sMRI/{label}/{selected_subject}/*_T1.nii.gz"
    nii_file_list = glob.glob(nii_search_pattern)
    
    if not nii_file_list:
        raise FileNotFoundError(
            f"找不到受試者 '{selected_subject}' 的 T1 檔案。\n"
            f"搜尋路徑: {nii_search_pattern}\n"
            f"請確認檔案存在於 data/sMRI/{label}/{selected_subject}/ 目錄下"
        )
```

### 方案 B: 標準化資料 (長期方案)

**目標**: 重新組織資料，統一命名規範

#### 建議的標準格式

```
data/
├── raw/                          # 原始資料
│   ├── fmri/
│   │   ├── AD/
│   │   │   └── sub-0007/        # 統一四位數，有連字號
│   │   │       └── sub-0007_task-rest_bold.nii.gz
│   │   └── NC/                  # 統一使用 NC
│   │       └── sub-0001/
│   │           └── sub-0001_task-rest_bold.nii.gz
│   │
│   └── smri/
│       ├── AD/
│       │   └── sub-0005/        # 統一四位數，有連字號
│       │       ├── sub-0005_T1.nii.gz      # 統一連字號
│       │       ├── sub-0005_DWI.nii.gz
│       │       └── sub-0005_T2-FLAIR.nii.gz
│       ├── MCI/
│       └── NC/
│
└── processed/                    # 處理後資料
    ├── AD/
    │   └── sub-0005/
    │       ├── sub-0005_GM_to_MNI.nii.gz
    │       ├── sub-0005_FA_to_MNI.nii.gz
    │       └── sub-0005_MD_to_MNI.nii.gz
    ├── MCI/
    └── NC/
```

**標準化規則**:
1. 所有受試者 ID: `sub-XXXX` (四位數，有連字號)
2. 所有檔案名稱: 使用連字號 `-` 而非底線 `_`
3. 統一標籤: `AD`, `MCI`, `NC` (不使用 `CN`)
4. 目錄結構: `data/{raw|processed}/{modality}/{label}/{subject_id}/`

---

## 立即行動建議

### 🚨 緊急修正 (今天完成)

1. **檢查實際使用的資料路徑**:
   ```bash
   # 檢查 sMRI 檔案是否存在
   dir data\sMRI\AD\sub-0005\sub_0005_T1.nii.gz
   
   # 檢查 fMRI 檔案是否存在
   dir data\fMRI\CN\sub-01\*.nii.gz
   ```

2. **修正 app.py 的 subject_id 轉換邏輯**:
   - 移除 `subject_id.replace("-", "_")` 
   - 移除 `selected_subject.replace("_", "-")`
   - 保持原始格式

3. **統一標籤命名**:
   - 決定使用 `CN` 還是 `NC`
   - 在整個系統中保持一致

### 📋 短期修正 (本週完成)

4. **建立資料驗證腳本**:
   ```python
   # scripts/validate_data_structure.py
   import glob
   import os
   
   def validate_data():
       # 檢查所有資料路徑
       # 驗證檔案存在
       # 報告不一致的地方
   ```

5. **更新文件**:
   - 記錄實際的資料結構
   - 說明命名規範
   - 提供資料準備指南

### 🔧 長期改進 (下個月)

6. **重構資料載入邏輯**:
   - 建立統一的 `DataLoader` 類別
   - 處理所有命名變體
   - 提供清晰的錯誤訊息

7. **考慮資料標準化**:
   - 評估重新組織資料的成本
   - 建立資料遷移腳本
   - 逐步遷移到標準格式

---

## 測試檢查清單

### ✅ 必須通過的測試

- [ ] fMRI 分析可以找到 `sub-01` 的檔案
- [ ] fMRI 分析可以找到 `sub-07` 的檔案
- [ ] sMRI 分析可以找到 `sub-0005` 的檔案
- [ ] sMRI 分析可以找到 `sub-0001` 的檔案
- [ ] CDDA 分析可以找到 `sub-0005` 的處理後檔案
- [ ] 標籤顯示正確 (AD, MCI, NC 或 CN)
- [ ] Ground truth 與實際資料夾匹配

### 🔍 需要驗證的邊界情況

- [ ] 受試者 ID 有連字號和沒連字號的情況
- [ ] 標籤 CN vs NC 的轉換
- [ ] 檔案名稱使用連字號 vs 底線
- [ ] 不存在的受試者 ID 的錯誤處理
- [ ] 檔案不存在時的錯誤訊息是否清晰

---

## 結論

你的系統有**嚴重的資料路徑不匹配問題**，主要原因是：

1. **資料本身的命名不一致** (fMRI vs sMRI)
2. **程式碼嘗試統一格式但方法不當** (來回轉換)
3. **標籤命名混亂** (CN vs NC)

**建議優先採用方案 A (最小修改)**，因為：
- 不需要移動或重命名大量資料
- 修改範圍小，風險低
- 可以快速驗證和部署

**長期應該考慮方案 B (標準化)**，因為：
- 提高系統可維護性
- 減少未來的錯誤
- 更容易擴展新功能

---

**下一步**: 我可以幫你生成修正後的程式碼，或者建立資料驗證腳本。你想先做哪一個？
