# Quick Start: Final Model

## 🚀 從零開始到最終模型

### Step 1: 提取所有 AAL ROI 特徵

```bash
python scripts/ml/extract_all_roi_features.py
```

**這會做什麼？**
- 從所有 MRI 影像提取 116 個 AAL 腦區特徵
- 儲存到 `data/processed/all_aal_roi_features.csv`
- 約需 5-10 分鐘

---

### Step 2: 訓練最終模型（混合方案）

```bash
python scripts/ml/train_final_model.py
```

**這會做什麼？**
- 使用 32 個 ROIs（24 文獻 + 8 數據驅動）
- 5-fold 交叉驗證
- 儲存最終模型到 `model/ml/final/`

**預期輸出：**
```
CV Accuracy: 75.4% ± 5.8%
ROC-AUC: 80.1% ± 6.7%

Top 10 Most Important ROIs:
  Cingulum_Post_R    0.0861
  Lingual_R          0.0635
  Cingulum_Mid_L     0.0614
  ...
```

---

### Step 3: 使用模型預測

```python
import joblib
import numpy as np
from nilearn import image as nimg, datasets
from nilearn.maskers import NiftiLabelsMasker

# 載入模型
model = joblib.load('model/ml/final/final_model.pkl')
scaler = joblib.load('model/ml/final/final_scaler.pkl')

# 載入特徵名稱
with open('model/ml/final/final_feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f]

# 載入 AAL atlas
aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
aal_img = nimg.load_img(aal_atlas.maps)
masker = NiftiLabelsMasker(labels_img=aal_img, standardize=False, strategy='mean')

# 提取特徵
mri_img = nimg.load_img('path/to/scan_T1.nii.gz')
all_features = masker.fit_transform(mri_img).flatten()

# 選擇 32 個 ROIs
aal_labels = [label.decode('utf-8') if isinstance(label, bytes) else label 
              for label in aal_atlas.labels[1:]]
selected_features = [all_features[aal_labels.index(roi)] for roi in feature_names]
features = np.array(selected_features).reshape(1, -1)

# 預測
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]
probabilities = model.predict_proba(features_scaled)[0]

print(f"Prediction: {'AD' if prediction == 1 else 'NC'}")
print(f"Confidence: NC={probabilities[0]:.2%}, AD={probabilities[1]:.2%}")
```

---

## 📊 模型效能

| 指標 | 數值 |
|------|------|
| CV 準確率 | 75.4% ± 5.8% |
| ROC-AUC | 80.1% ± 6.7% |
| F1 分數 | 52.9% ± 27.8% |
| 過擬合差距 | 0.246 |

---

## 🧠 使用的 32 個 ROIs

### 原始 24 個（文獻選擇）
- Hippocampus, Amygdala, ParaHippocampal
- Temporal lobe (Sup, Mid, Inf)
- Parietal lobe (Sup, Inf)
- Cingulate (Ant, Post)
- Frontal (Sup, Mid)

### 新增 8 個（數據驅動）
- **Cingulum_Mid** (L/R) - Default Mode Network
- **Fusiform** (L/R) - 物體識別
- **Lingual** (L/R) - 視覺處理
- **SupraMarginal** (L/R) - 語言處理

---

## 🎯 為什麼選擇混合方案？

### ✅ 優點

1. **平衡效能與可解釋性**
   - 準確率 75.4%（可接受）
   - 所有 ROIs 都有臨床意義

2. **結合領域知識與數據驅動**
   - 24 個文獻驗證的 ROIs
   - 8 個統計發現的 ROIs

3. **臨床可解釋**
   - 可以向醫生解釋每個腦區的作用
   - 符合 AD 病理學

4. **穩定性好**
   - 標準差 5.8%（可接受）
   - 過擬合風險中等

### ⚠️ 限制

1. **樣本量小** (n=65)
   - 需要更多數據驗證
   - 高變異性

2. **需要外部驗證**
   - 目前只在單一數據集測試
   - 需要獨立測試集

---

## 📁 輸出檔案

### 模型檔案
```
model/ml/final/
├── final_model.pkl           # 訓練好的模型
├── final_scaler.pkl          # 特徵標準化器
├── final_feature_names.txt   # 32 個 ROI 名稱
└── final_roi_list.csv       # ROI 列表（含來源標籤）
```

### 分析結果
```
output/ml/final_model/
├── final_model_report.txt         # 詳細報告
├── final_model_analysis.png       # 視覺化分析
└── final_feature_importance.csv   # 特徵重要性
```

---

## 🔍 驗證模型品質

### 1. 查看特徵重要性

```bash
python scripts/ml/analyze_feature_importance.py
```

確認模型是否學到正確的腦區。

### 2. 批次預測測試

```bash
python scripts/ml/batch_predict.py
```

在所有樣本上測試模型效能。

---

## 💡 下一步

### 如果效能滿意：
✅ 使用最終模型進行預測  
✅ 收集更多數據進行驗證  
✅ 考慮部署到臨床環境

### 如果想進一步改進：
- 收集更多樣本（目標：200+）
- 加入多模態數據（T2, DWI, fMRI）
- 加入臨床特徵（年齡、性別、MMSE）
- 嘗試深度學習方法

---

## 📚 相關文件

- `docs/FINAL_MODEL_SUMMARY.md` - 最終模型完整總結
- `docs/MODEL.md` - 模型方法論詳細說明
- `docs/FEATURE_SELECTION_ANALYSIS.md` - 特徵選擇分析
- `output/ml/final_model/final_model_report.txt` - 訓練報告

---

## ❓ 常見問題

### Q: 為什麼不用 Top 30？準確率更高（81.5%）

A: Top 30 包含很多非 AD 相關的腦區（運動皮質、小腦等），臨床上難以解釋。混合方案在效能和可解釋性之間取得平衡。

### Q: 可以用這個模型做臨床診斷嗎？

A: **不可以單獨使用**。這個模型應該作為輔助工具，配合臨床評估、認知測試和其他生物標記一起使用。

### Q: 75.4% 的準確率夠好嗎？

A: 對於 n=65 的小樣本來說是合理的。文獻中類似樣本量的研究準確率在 70-85% 之間。需要更多數據來提高穩定性。

### Q: 如何提高模型效能？

A: 
1. 收集更多數據（最重要）
2. 加入多模態影像
3. 加入臨床特徵
4. 使用集成學習

---

## 🎉 恭喜！

你已經完成了從特徵提取到最終模型的完整流程！

**模型特點：**
- ✅ 32 個臨床可解釋的 ROIs
- ✅ 75.4% 準確率，80.1% ROC-AUC
- ✅ 結合領域知識與數據驅動
- ✅ 適合小樣本（n=65）

**準備好使用了！** 🚀
