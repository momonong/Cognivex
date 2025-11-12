# ✅ 資料設置完成

## 📊 資料結構

### 當前結構（扁平式）
```
data/raw/
├── AD/
│   ├── sub_0005_DWI.nii.gz
│   ├── sub_0005_T1.nii.gz
│   ├── sub_0005_T2_FLAIR.nii.gz
│   ├── sub_0011_T1.nii.gz
│   └── ... (23 個受試者，69 個檔案)
└── NC/
    ├── sub_0001_DWI.nii.gz
    ├── sub_0001_T1.nii.gz
    ├── sub_0001_T2_FLAIR.nii.gz
    ├── sub_0002_T1.nii.gz
    └── ... (42 個受試者，126 個檔案)
```

### 統計
- **AD 受試者**: 23 個
- **NC 受試者**: 42 個
- **總檔案數**: 195 個
- **總大小**: 538.76 MB (0.53 GB)

## 🔧 已完成的調整

### 1. 資料複製
✅ 從外接硬碟複製到專案資料夾
- 來源: `E:\fMRI\Model\sMRI_data_MultiModal_Aligned_MNI`
- 目標: `data/raw/`

### 2. app.py 調整
✅ 更新受試者列表生成邏輯
- 從扁平結構提取受試者 ID
- 支援 `sub_XXXX` 格式

✅ 更新檔案搜尋邏輯
- 結構性 MRI: 搜尋 `*_T1.nii.gz` 檔案
- 功能性 MRI: 搜尋所有 `.nii.gz` 檔案
- 支援多種檔名格式

## 🚀 使用方式

### 啟動應用
```bash
streamlit run app.py
```

### 選擇分析模式
1. 在側邊欄選擇 **"Structural MRI (T1)"**
2. 從下拉選單選擇受試者（例如：`sub_0005`）
3. 點擊 **"Start Analysis"**

### 可用的受試者

#### AD 組（23 個）
- sub_0005, sub_0011, sub_0012, sub_0014, sub_0020
- sub_0024, sub_0038, sub_0044, sub_0046, sub_0047
- sub_0056, sub_0058, sub_0065, sub_0073, sub_0074
- sub_0075, sub_0082, sub_0099, sub_0101, sub_0102
- sub_0125, sub_0139, sub_0140

#### NC 組（42 個）
- sub_0001, sub_0002, sub_0007, sub_0008, sub_0010
- sub_0015, sub_0018, sub_0021, sub_0023, sub_0027
- sub_0028, sub_0030, sub_0031, sub_0034, sub_0035
- sub_0037, sub_0040, sub_0042, sub_0043, sub_0045
- sub_0048, sub_0052, sub_0054, sub_0064, sub_0067
- sub_0072, sub_0076, sub_0079, sub_0081, sub_0083
- sub_0085, sub_0086, sub_0087, sub_0088, sub_0089
- sub_0090, sub_0105, sub_0110, sub_0111, sub_0115
- sub_0116, sub_0119

## 📝 檔案類型

每個受試者有 3 個檔案：
- **T1**: 結構性 MRI（用於結構性分析）
- **T2_FLAIR**: T2 加權影像
- **DWI**: 擴散加權影像

## ⚠️ 注意事項

### 警告訊息（可忽略）
應用啟動時會顯示以下警告，這些是可選依賴，不影響結構性 MRI 功能：
```
[WARNING] google-generativeai not installed. Gemini provider will not be available.
[WARNING] langchain_aws not installed. Bedrock provider will not be available.
[WARNING] ollama not installed. Ollama provider will not be available.
Error importing constants: No module named 'model.shufflenet.model'
```

### 檔案搜尋邏輯
- 結構性 MRI 只使用 T1 檔案
- 系統會自動根據受試者 ID 和標籤（AD/NC）找到正確的檔案
- 支援多種檔名格式（`sub_XXXX` 和 `sub-XXXX`）

## 🎯 下一步

1. ✅ 資料已準備好
2. ✅ app.py 已調整
3. ✅ 應用已啟動
4. 🔄 開始測試分析功能

### 測試建議
1. 選擇一個 AD 受試者（例如：sub_0005）
2. 選擇 Structural MRI (T1) 模式
3. 點擊 Start Analysis
4. 查看結果：
   - 預測結果（AD/NC）
   - 特徵重要性
   - 腦區視覺化
   - 中英文報告

## 📞 問題排查

### 找不到受試者
- 確認資料在 `data/raw/AD/` 和 `data/raw/NC/` 目錄下
- 檢查檔名格式是否為 `sub_XXXX_T1.nii.gz`

### 找不到 T1 檔案
- 確認每個受試者都有對應的 `*_T1.nii.gz` 檔案
- 檢查檔案權限

### 分析失敗
- 查看錯誤訊息
- 確認模型檔案存在於 `model/ml/final/`
- 檢查記憶體是否足夠

---

**狀態**: ✅ 完成  
**最後更新**: 2024年
