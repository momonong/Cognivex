# MCI 資料遷移指南

本指南說明如何將 MCI (Mild Cognitive Impairment) 資料遷移到 `data/sMRI/MCI/` 目錄。

---

## 目標資料結構

```
data/sMRI/
├── AD/              # Alzheimer's Disease (23 個受試者)
│   ├── sub-0005/
│   │   ├── sub_0005_T1.nii.gz
│   │   ├── sub_0005_T2_FLAIR.nii.gz
│   │   └── sub_0005_DWI.nii.gz
│   └── ...
├── NC/              # Normal Control (42 個受試者)
│   ├── sub-0001/
│   │   └── sub_0001_T1.nii.gz
│   └── ...
└── MCI/             # Mild Cognitive Impairment (待新增)
    ├── sub-XXXX/
    │   └── sub_XXXX_T1.nii.gz
    └── ...
```

---

## 步驟 1: 尋找 MCI 資料

執行尋找腳本來定位你的 MCI 資料：

```bash
python scripts/find_mci_data.py
```

這個腳本會：
- 檢查 CSV 檔案中是否有 MCI 標籤
- 搜尋常見的資料目錄
- 顯示當前的資料結構
- 提供下一步建議

---

## 步驟 2: 準備 MCI 資料

### 選項 A: 如果你已經有 MCI 的 NIfTI 檔案

假設你的 MCI 資料在某個目錄中，例如：

```
/path/to/mci/data/
├── subject001_T1.nii.gz
├── subject002_T1.nii.gz
└── ...
```

或者：

```
/path/to/mci/data/
├── sub-0100/
│   └── T1.nii.gz
├── sub-0101/
│   └── T1.nii.gz
└── ...
```

### 選項 B: 如果 MCI 資料在 CSV 中但沒有對應的 NIfTI 檔案

你需要：
1. 從 ADNI 網站下載對應受試者的 T1 MRI 資料
2. 或者從原始資料備份中找到這些檔案

### 選項 C: 從 ADNI 下載

1. 登入 ADNI 網站: https://adni.loni.usc.edu/
2. 進入 "Download" 區域
3. 選擇 "Image Collections"
4. 篩選條件：
   - Diagnosis: MCI
   - Modality: MRI
   - Sequence: T1
5. 下載選定的檔案

---

## 步驟 3: 執行遷移（模擬）

先執行模擬模式，確認操作正確：

```bash
python scripts/migrate_mci_data.py --source /path/to/mci/data --dry-run
```

這會顯示將要執行的操作，但不會實際複製檔案。

### 範例輸出

```
============================================================
MCI 資料遷移腳本
============================================================
來源目錄: /path/to/mci/data
目標目錄: data/sMRI/MCI
檔案模式: *T1*.nii.gz
模式: 模擬執行 (DRY RUN)
============================================================

在 /path/to/mci/data 中找到 15 個 T1 檔案

[DRY RUN] 將創建目標目錄: data/sMRI/MCI

開始處理 15 個檔案...
============================================================

處理: sub-0100
  來源: /path/to/mci/data/sub-0100/T1.nii.gz
  目標: data/sMRI/MCI/sub-0100/sub_0100_T1.nii.gz
  [DRY RUN] 將複製檔案

...

============================================================
處理完成！
  成功: 15 個檔案
  跳過: 0 個檔案
  總計: 15 個檔案

[DRY RUN] 這是模擬執行，沒有實際複製檔案
移除 --dry-run 參數以實際執行
```

---

## 步驟 4: 執行實際遷移

確認模擬結果正確後，執行實際遷移：

```bash
python scripts/migrate_mci_data.py --source /path/to/mci/data
```

---

## 步驟 5: 驗證資料

遷移完成後，驗證資料結構：

```bash
python scripts/migrate_mci_data.py --verify
```

### 範例輸出

```
驗證資料結構...
  目標目錄: data/sMRI/MCI
  受試者數量: 15

驗證結果:
  有效: 15 個受試者
  無效: 0 個受試者
```

---

## 步驟 6: 更新特徵提取

遷移完成後，你需要重新提取 MCI 的 ROI 特徵。

### 6.1 檢查現有的特徵提取腳本

```bash
# 查看是否有特徵提取腳本
ls scripts/ml/
```

### 6.2 提取 MCI 特徵

如果有現成的腳本：

```bash
python scripts/ml/extract_features.py --data-dir data/sMRI/MCI --label MCI
```

如果沒有，需要創建特徵提取腳本（參考 `app/core/ml_processing/feature_extractor.py`）。

---

## 進階選項

### 自訂檔案模式

如果你的 T1 檔案名稱不同，可以指定模式：

```bash
# 例如: T1w.nii.gz
python scripts/migrate_mci_data.py --source /path/to/mci/data --pattern "*T1w*.nii.gz"

# 例如: anat.nii.gz
python scripts/migrate_mci_data.py --source /path/to/mci/data --pattern "*anat*.nii.gz"
```

### 自訂目標目錄

```bash
python scripts/migrate_mci_data.py --source /path/to/mci/data --target data/sMRI/MCI_test
```

---

## 常見問題

### Q1: 找不到 T1 檔案

**問題**: 腳本顯示 "沒有找到任何 T1 檔案"

**解決方法**:
1. 檢查來源目錄路徑是否正確
2. 檢查檔案名稱模式是否匹配
3. 使用 `--pattern` 參數指定正確的模式

```bash
# 列出來源目錄中的檔案
ls /path/to/mci/data/**/*.nii.gz

# 根據實際檔案名稱調整模式
python scripts/migrate_mci_data.py --source /path/to/mci/data --pattern "實際的模式"
```

### Q2: 受試者 ID 提取錯誤

**問題**: 腳本無法正確提取受試者 ID

**解決方法**:
腳本支援以下格式：
- `sub-0001`
- `sub_0001`
- `0001`
- `ADNI_001_S_0001`

如果你的格式不同，需要修改 `extract_subject_id()` 函數。

### Q3: 檔案已存在

**問題**: 腳本顯示 "檔案已存在"

**解決方法**:
1. 檢查是否已經遷移過
2. 如果需要覆蓋，先刪除目標目錄：
   ```bash
   rm -rf data/sMRI/MCI/sub-XXXX
   ```
3. 重新執行遷移

### Q4: 權限錯誤

**問題**: "Permission denied" 錯誤

**解決方法**:
```bash
# Windows
# 以管理員身份執行 PowerShell

# Linux/Mac
sudo python scripts/migrate_mci_data.py --source /path/to/mci/data
```

---

## 手動遷移（備用方案）

如果腳本無法使用，可以手動遷移：

### Windows (PowerShell)

```powershell
# 創建 MCI 目錄
New-Item -ItemType Directory -Path "data\sMRI\MCI" -Force

# 對每個 MCI 受試者
$subjects = @("sub-0100", "sub-0101", "sub-0102")  # 替換為實際的受試者 ID

foreach ($sub in $subjects) {
    # 創建受試者目錄
    New-Item -ItemType Directory -Path "data\sMRI\MCI\$sub" -Force
    
    # 複製 T1 檔案
    Copy-Item -Path "來源路徑\$sub\T1.nii.gz" -Destination "data\sMRI\MCI\$sub\${sub}_T1.nii.gz"
}
```

### Linux/Mac (Bash)

```bash
# 創建 MCI 目錄
mkdir -p data/sMRI/MCI

# 對每個 MCI 受試者
subjects=("sub-0100" "sub-0101" "sub-0102")  # 替換為實際的受試者 ID

for sub in "${subjects[@]}"; do
    # 創建受試者目錄
    mkdir -p "data/sMRI/MCI/$sub"
    
    # 複製 T1 檔案
    cp "來源路徑/$sub/T1.nii.gz" "data/sMRI/MCI/$sub/${sub//-/_}_T1.nii.gz"
done
```

---

## 下一步

遷移完成後：

1. ✅ 驗證資料結構
2. ✅ 提取 ROI 特徵
3. ✅ 更新訓練資料集
4. ✅ 重新訓練模型（包含 MCI 類別）
5. ✅ 更新系統以支援三分類（AD/MCI/NC）

參考文件：
- `docs/SYSTEM_TECHNICAL_DOCUMENTATION.md` - 系統架構
- `app/core/ml_processing/feature_extractor.py` - 特徵提取
- `app/core/ml_processing/model_loader.py` - 模型載入

---

## 需要幫助？

如果遇到問題：

1. 執行 `python scripts/find_mci_data.py` 診斷
2. 檢查錯誤訊息
3. 查看此文件的常見問題部分
4. 聯繫系統維護者

---

**最後更新**: 2024-11-13
