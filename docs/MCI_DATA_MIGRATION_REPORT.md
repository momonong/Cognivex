# MCI 資料遷移報告

**執行日期**: 2024-11-13  
**狀態**: ✅ 完成

---

## 執行摘要

成功將 **71 個 MCI 受試者**的資料從 E 槽遷移到 `data/sMRI/MCI/` 目錄，總計 **213 個檔案**。

---

## 資料來源

**來源路徑**: `E:\fMRI\Model\sMRI_data_MultiModal_Aligned_MNI\MCI`

**來源結構**: 所有檔案直接放在 MCI 目錄下，沒有子目錄
```
E:\...\MCI\
├── sub_0003_T1.nii.gz
├── sub_0003_T2_FLAIR.nii.gz
├── sub_0003_DWI.nii.gz
├── sub_0009_T1.nii.gz
└── ...
```

---

## 目標結構

**目標路徑**: `data/sMRI/MCI/`

**目標結構**: 按照 AD 和 NC 的格式組織
```
data/sMRI/MCI/
├── sub-0003/
│   ├── sub_0003_T1.nii.gz
│   ├── sub_0003_T2_FLAIR.nii.gz
│   └── sub_0003_DWI.nii.gz
├── sub-0009/
│   ├── sub_0009_T1.nii.gz
│   ├── sub_0009_T2_FLAIR.nii.gz
│   └── sub_0009_DWI.nii.gz
└── ...
```

---

## 遷移統計

### 受試者數量

| 類別 | 受試者數量 | 檔案數量 |
|------|-----------|---------|
| AD | 23 | 69 |
| NC | 42 | 126 |
| **MCI** | **71** | **213** |
| **總計** | **136** | **408** |

### 檔案類型分布

每個受試者包含 3 種檔案：
- **T1.nii.gz**: T1-weighted MRI (71 個)
- **T2_FLAIR.nii.gz**: T2 FLAIR MRI (71 個)
- **DWI.nii.gz**: Diffusion Weighted Imaging (71 個)

---

## MCI 受試者列表

共 71 個受試者：

```
sub-0003, sub-0009, sub-0013, sub-0016, sub-0017, sub-0019, sub-0022, sub-0025,
sub-0026, sub-0032, sub-0033, sub-0036, sub-0039, sub-0041, sub-0049, sub-0050,
sub-0051, sub-0053, sub-0055, sub-0059, sub-0060, sub-0061, sub-0062, sub-0063,
sub-0066, sub-0068, sub-0069, sub-0070, sub-0077, sub-0078, sub-0080, sub-0084,
sub-0091, sub-0092, sub-0093, sub-0094, sub-0095, sub-0096, sub-0097, sub-0098,
sub-0100, sub-0103, sub-0104, sub-0106, sub-0107, sub-0108, sub-0109, sub-0112,
sub-0113, sub-0114, sub-0117, sub-0118, sub-0121, sub-0122, sub-0123, sub-0124,
sub-0126, sub-0127, sub-0128, sub-0129, sub-0130, sub-0131, sub-0132, sub-0133,
sub-0134, sub-0135, sub-0136, sub-0137, sub-0138, sub-0141, sub-0142
```

---

## 驗證結果

✅ **結構驗證**: 所有受試者目錄結構正確  
✅ **檔案完整性**: 每個受試者都有 3 個檔案（T1, T2_FLAIR, DWI）  
✅ **命名規範**: 檔案命名符合現有格式  
✅ **目錄組織**: 與 AD/NC 的結構一致

### 樣本驗證

**受試者**: sub-0003  
**檔案**:
- sub_0003_T1.nii.gz
- sub_0003_T2_FLAIR.nii.gz
- sub_0003_DWI.nii.gz

---

## 下一步工作

### 1. 特徵提取

需要為 MCI 資料提取 ROI 特徵：

```bash
# 使用現有的特徵提取器
python scripts/ml/extract_features.py --data-dir data/sMRI/MCI --label MCI
```

或者使用 Python API：

```python
from app.core.ml_processing import ROIFeatureExtractor

extractor = ROIFeatureExtractor()

# 對每個 MCI 受試者提取特徵
for subject_dir in Path("data/sMRI/MCI").iterdir():
    t1_file = subject_dir / f"{subject_dir.name.replace('-', '_')}_T1.nii.gz"
    features = extractor.extract_features(str(t1_file), roi_list)
    # 儲存特徵...
```

### 2. 更新訓練資料集

將 MCI 特徵加入訓練資料：

```python
import pandas as pd

# 讀取現有資料
df_existing = pd.read_csv("data/processed/all_aal_roi_features.csv")

# 加入 MCI 資料
df_mci = pd.read_csv("data/processed/mci_aal_roi_features.csv")
df_combined = pd.concat([df_existing, df_mci], ignore_index=True)

# 儲存
df_combined.to_csv("data/processed/all_aal_roi_features_with_mci.csv", index=False)
```

### 3. 模型訓練

#### 選項 A: 二分類（AD vs. NC，MCI 作為測試集）

```python
# 使用 MCI 作為獨立測試集
train_data = df[df['label'].isin(['AD', 'NC'])]
test_data = df[df['label'] == 'MCI']
```

#### 選項 B: 三分類（AD vs. MCI vs. NC）

```python
# 訓練三分類模型
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, max_depth=15)
model.fit(X_train, y_train)  # y_train 包含 'AD', 'MCI', 'NC'
```

#### 選項 C: 階層分類

```python
# 第一層: NC vs. 認知障礙 (AD + MCI)
# 第二層: AD vs. MCI
```

### 4. 更新系統支援三分類

需要修改的檔案：
- `app/core/ml_processing/model_loader.py` - 支援三分類模型
- `app/agents/structural_mri_inference.py` - 處理 MCI 預測
- `app/ui/structural_mri_components.py` - 顯示 MCI 結果
- `app.py` - UI 更新

---

## 資料集統計

### 類別平衡

```
NC:  42 (30.9%)
MCI: 71 (52.2%)  ← 最多
AD:  23 (16.9%)
```

**觀察**:
- MCI 是最大的類別（52.2%）
- AD 是最小的類別（16.9%）
- 類別不平衡，訓練時需要考慮

**建議**:
- 使用 class_weight='balanced' 參數
- 或使用 SMOTE 等過採樣技術
- 或使用分層抽樣

### 資料分割建議

```python
from sklearn.model_selection import train_test_split

# 分層抽樣，保持類別比例
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # 保持類別比例
    random_state=42
)
```

---

## 檔案清單

### 遷移腳本

- `scripts/migrate_mci_data.py` - 通用遷移腳本
- `scripts/find_mci_data.py` - 資料尋找工具
- `scripts/MCI_DATA_MIGRATION_GUIDE.md` - 遷移指南

### 文件

- `docs/MCI_DATA_MIGRATION_REPORT.md` - 本報告
- `docs/SYSTEM_TECHNICAL_DOCUMENTATION.md` - 系統技術文件

---

## 參考資料

### MCI 相關資訊

**Mild Cognitive Impairment (MCI)**:
- 介於正常老化和失智症之間的過渡階段
- 認知功能下降但日常生活功能正常
- 每年約 10-15% 的 MCI 患者會進展為 AD
- 是早期介入的重要目標

### 臨床意義

三分類模型的價值：
1. **早期檢測**: 識別 MCI 可以早期介入
2. **疾病進展**: 追蹤從 NC → MCI → AD 的進展
3. **治療決策**: 不同階段需要不同的治療策略
4. **預後評估**: MCI 患者的預後評估

---

## 總結

✅ **遷移完成**: 71 個 MCI 受試者，213 個檔案  
✅ **結構正確**: 與 AD/NC 格式一致  
✅ **資料完整**: 每個受試者都有 T1, T2_FLAIR, DWI  
✅ **準備就緒**: 可以開始特徵提取和模型訓練

**下一步**: 提取 MCI 的 ROI 特徵，然後訓練改進的模型

---

**報告完成日期**: 2024-11-13  
**執行者**: AI Assistant
