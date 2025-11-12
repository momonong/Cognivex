# ✅ 最終設置完成

## 📊 新的資料結構

```
data/
├── fMRI/          (功能性 MRI 資料)
│   ├── AD/        (21 個受試者)
│   │   └── sub-XX/
│   │       └── *.nii.gz
│   └── CN/        (11 個受試者)
│       └── sub-XX/
│           └── *.nii.gz
│
└── sMRI/          (結構性 MRI 資料)
    ├── AD/        (23 個受試者)
    │   └── sub-XXXX/
    │       ├── sub_XXXX_T1.nii.gz
    │       ├── sub_XXXX_T2_FLAIR.nii.gz
    │       └── sub_XXXX_DWI.nii.gz
    └── NC/        (42 個受試者)
        └── sub-XXXX/
            ├── sub_XXXX_T1.nii.gz
            ├── sub_XXXX_T2_FLAIR.nii.gz
            └── sub_XXXX_DWI.nii.gz
```

## ✅ 完成的工作

### 1. 資料夾重新命名
- ✅ `data/raw` → `data/fMRI` (功能性 MRI)
- ✅ `data/cardinal_tien` → `data/sMRI` (結構性 MRI)

### 2. app.py 更新
- ✅ 更新受試者列表生成邏輯
  - 結構性 MRI: 從 `data/sMRI/` 讀取
  - 功能性 MRI: 從 `data/fMRI/` 讀取
- ✅ 更新檔案搜尋邏輯
  - 結構性 MRI: 搜尋 `*_T1.nii.gz`
  - 功能性 MRI: 搜尋所有 `.nii.gz`
- ✅ 處理標籤轉換 (CN → NC)
- ✅ 處理受試者 ID 格式 (sub_XXXX ↔ sub-XXXX)

### 3. 驗證完成
- ✅ 功能性 MRI: 32 個受試者 (21 AD + 11 NC)
- ✅ 結構性 MRI: 65 個受試者 (23 AD + 42 NC)
- ✅ 檔案搜尋測試通過

## 🚀 使用方式

### 啟動應用
```bash
streamlit run app.py
```

### 功能性 MRI 分析
1. 選擇 **"Functional MRI (fMRI)"** 模式
2. 選擇模型 (ShuffleNet/CapsNet/MCADNNet)
3. 選擇受試者 (例如: sub-07, sub-08)
4. 點擊 "Start Analysis"

### 結構性 MRI 分析
1. 選擇 **"Structural MRI (T1)"** 模式
2. 選擇受試者 (例如: sub_0005, sub_0001)
3. 點擊 "Start Analysis"
4. 查看結果:
   - 預測結果 (AD/NC)
   - 特徵重要性
   - 腦區視覺化 (中文名稱)
   - 功能系統分析
   - 中英文報告

## 📊 可用受試者

### 功能性 MRI (32 個)
- **AD**: 21 個受試者
- **NC**: 11 個受試者
- 格式: `sub-XX` (例如: sub-07, sub-08)

### 結構性 MRI (65 個)
- **AD**: 23 個受試者
  - sub_0005, sub_0011, sub_0012, sub_0014, sub_0020, ...
- **NC**: 42 個受試者
  - sub_0001, sub_0002, sub_0007, sub_0008, sub_0010, ...
- 格式: `sub_XXXX` (例如: sub_0005, sub_0001)

## 🎯 系統特色

### 功能性 MRI
- ✅ 深度學習模型 (ShuffleNet/CapsNet/MCADNNet)
- ✅ 活化圖視覺化
- ✅ 互動式 3D 檢視器
- ✅ 雙語報告

### 結構性 MRI
- ✅ Random Forest 分類器
- ✅ ROI 特徵提取 (32 個特徵)
- ✅ 中文腦區名稱 (100+ ROI)
- ✅ 功能系統分類 (5 大系統)
- ✅ Dashboard 風格結果
- ✅ 雙語報告

## 💡 注意事項

### 受試者 ID 格式
- **功能性 MRI**: `sub-XX` (短橫線)
- **結構性 MRI**: `sub_XXXX` (底線，4 位數字)
- 系統會自動處理格式轉換

### 標籤處理
- 功能性 MRI 的 `CN` 標籤會自動轉換為 `NC`
- 兩種模式都使用 AD/NC 標籤

### 檔案類型
- **功能性 MRI**: 4D fMRI 數據
- **結構性 MRI**: T1 加權影像 (只使用 T1 檔案)

## 🐛 問題排查

### 找不到受試者
- 確認選擇了正確的分析模式
- 檢查資料在正確的目錄下:
  - 功能性: `data/fMRI/AD/` 或 `data/fMRI/CN/`
  - 結構性: `data/sMRI/AD/` 或 `data/sMRI/NC/`

### 找不到檔案
- 功能性 MRI: 確認受試者資料夾中有 `.nii.gz` 檔案
- 結構性 MRI: 確認有 `*_T1.nii.gz` 檔案

### 分析失敗
- 查看錯誤訊息
- 確認模型檔案存在
- 檢查記憶體是否足夠

## 📝 相關檔案

- `scripts/rename_data_folders.py` - 資料夾重新命名腳本
- `verify_new_structure.py` - 驗證資料結構
- `check_data_structure.py` - 全面檢查資料
- `app.py` - 主應用程式

## 🎉 總結

✅ **系統完全準備就緒！**

- 資料結構: 清晰直觀 (fMRI/sMRI)
- 功能性 MRI: 32 個受試者可用
- 結構性 MRI: 65 個受試者可用
- 程式碼: 已更新並測試通過

**現在可以開始使用了！**

```bash
streamlit run app.py
```

---

*最後更新: 2024年*
*資料來源: Cardinal Tien Hospital + ADNI*
