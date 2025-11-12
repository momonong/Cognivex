# 🚀 快速啟動指南

## 系統需求

- Python 3.11+
- 8GB+ RAM
- 網路連接（首次執行時下載 Atlas）

## 安裝步驟

### 1. 安裝依賴

```bash
# 核心依賴
pip install streamlit scikit-learn nilearn antspyx matplotlib pandas numpy

# 可選依賴（功能性 MRI）
pip install opencv-python torch langgraph
```

### 2. 檢查模型檔案

確保以下檔案存在：
```
model/ml/final/
├── final_model.pkl
└── final_scaler.pkl
```

### 3. 準備數據

數據應放在以下結構：
```
data/raw/
├── AD/
│   └── sub-001/
│       └── *.nii.gz
└── NC/
    └── sub-002/
        └── *.nii.gz
```

## 啟動應用

```bash
streamlit run app.py
```

應用會在瀏覽器中自動打開（通常是 http://localhost:8501）

## 使用結構性 MRI 分析

### 步驟 1: 選擇分析模式
在側邊欄選擇 **"Structural MRI (T1)"**

### 步驟 2: 選擇受試者
從下拉選單選擇受試者（例如：sub-001）

### 步驟 3: 開始分析
點擊 **"Start Analysis"** 按鈕

### 步驟 4: 查看結果
- **預測結果**: AD 或 NC 分類
- **信心度**: 模型預測的信心程度
- **特徵重要性**: 哪些腦區最重要
- **腦區視覺化**: 重要腦區的 3D 顯示
- **功能系統分析**: 按功能系統分組的結果
- **中英文報告**: 詳細的分析報告

## 使用功能性 MRI 分析

### 步驟 1: 選擇分析模式
在側邊欄選擇 **"Functional MRI (fMRI)"**

### 步驟 2: 選擇模型
選擇深度學習模型：
- **ShuffleNet**: 高準確度（推薦）
- **CapsNet**: 複雜 3D 模式
- **MCADNNet**: 傳統 CNN

### 步驟 3: 選擇受試者並開始分析

### 步驟 4: 查看結果
- 活化圖
- 預測結果
- 互動式 3D 檢視器
- 中英文報告

## 測試系統

### 快速測試
```bash
# 測試結構性 MRI 組件
python test_structural_only.py

# 測試 workflow
python test_workflow_mock.py
```

### 預期輸出
```
✅ UI 組件導入成功
✅ Structural MRI agents 導入成功
✅ 核心 ML 模組測試通過
✅ 中文名稱系統正常
✅ 功能分類系統正常
✅ 配置測試通過
✅ 模型檔案檢查通過
```

## 常見問題

### Q: 第一次執行很慢？
A: 系統需要下載 AAL atlas（約 50MB），只需下載一次。

### Q: 看到 WARNING 訊息？
A: 可選依賴的警告不影響結構性 MRI 功能。

### Q: 模型載入失敗？
A: 確保模型檔案存在於 `model/ml/final/` 目錄。

### Q: 找不到受試者？
A: 檢查數據是否放在正確的目錄結構中。

### Q: 分析失敗？
A: 查看錯誤訊息，確保：
- NIfTI 檔案格式正確
- 檔案路徑可訪問
- 有足夠的記憶體

## 功能特色

### 🧠 結構性 MRI
- Random Forest 分類器
- ROI 特徵提取
- 中文腦區名稱
- 功能系統分類
- Dashboard 風格結果

### 🎯 功能性 MRI
- 深度學習模型
- 活化圖視覺化
- 互動式 3D 檢視器
- XAI 可解釋性

### 📊 報告生成
- 雙語報告（中英文）
- 詳細的分析結果
- 視覺化圖表
- 臨床建議

## 效能指標

- **結構性 MRI**: 5-10 秒/受試者
- **功能性 MRI**: 30-60 秒/受試者
- **記憶體使用**: 2-4 GB
- **準確度**: 80%+（取決於模型）

## 支援的格式

- **輸入**: NIfTI (.nii, .nii.gz)
- **輸出**: PNG (視覺化), Markdown (報告)

## 下一步

1. 📖 閱讀 [完整文檔](docs/INTEGRATION_COMPLETE.md)
2. 🧪 執行測試腳本
3. 🎨 自訂視覺化設定
4. 📊 分析您的數據

## 需要幫助？

- 查看 `docs/` 目錄中的詳細文檔
- 執行測試腳本診斷問題
- 檢查錯誤日誌

---

**祝您使用愉快！** 🎉
