# 模型診斷報告

## 📋 執行摘要

經過三個診斷腳本的分析，我們發現了以下關鍵問題：

### ✅ 正常的部分

1. **Scaling Pipeline**: ✓ 正確
   - StandardScaler 已正確整合在 Pipeline 中
   - 訓練時的 mean 和 scale 已保存
   - 推理時自動應用 scaling

2. **同一 ROI 內的共線性**: ✓ 正常
   - GM vs FA: 平均相關性 = 0.11
   - GM vs MD: 平均相關性 = -0.05
   - FA vs MD: 平均相關性 = -0.17
   - **沒有同一 ROI 內的高共線性** (|r| < 0.9)

3. **AD 相關腦區**: ✓ 存在
   - Hippocampus, Amygdala, Olfactory 等都有特徵
   - 但**不在 Top 10 SHAP 特徵中**

### ⚠️ 發現的問題

## 問題 1: 鏡像特徵模式 (Mirror Effect)

**觀察**:
```
Top SHAP Features (所有受試者都相似):
1. Supp_Motor_Area_L_GM:  -0.0742 ← towards NC
2. Supp_Motor_Area_L_FA:  +0.0742 → towards AD
3. Frontal_Sup_Medial_L_GM: -0.0427 ← towards NC
4. Frontal_Sup_Medial_L_FA: +0.0427 → towards AD
5. Olfactory_R_FA: -0.0269 ← towards NC
6. Olfactory_R_MD: +0.0269 → towards AD
```

**問題**:
- 同一區域的不同模態（GM vs FA, FA vs MD）有**完全相同的絕對值**但**相反的符號**
- 這表示模型在這些特徵對之間**平分權重**
- 雖然不是同一 ROI 內的共線性，但可能是**跨 ROI 的共線性**

**原因**:
- 111 對高相關特徵（|r| >= 0.9），主要是：
  - 左右半球的相同區域（如 Frontal_Sup_2_L_GM vs Frontal_Sup_2_R_GM, r=0.96)
  - 相鄰的額葉區域（如 Frontal_Sup_2_L_GM vs Frontal_Mid_2_L_GM, r=0.97)

## 問題 2: AD 生物標記不在 Top 特徵

**觀察**:
- Hippocampus: **不在 Top 10**
- Amygdala: **不在 Top 10**
- ParaHippocampal: **不在 Top 10**
- Posterior Cingulate: **不在 Top 10**
- 只有 Olfactory (2/10) 在 Top 10 中

**問題**:
- 模型主要依賴**運動區**（Supp_Motor_Area, Precentral, Rolandic_Oper）
- 這些區域在 AD 病理學中**不是主要受損區域**

**可能原因**:
1. **特徵選擇偏差**: SelectFromModel 選擇了 30/498 (6%) 特徵
   - 可能選擇了高方差但生物學意義較低的特徵
   - Hippocampus 可能因為方差較小而被排除

2. **類別不平衡**: AD=21, MCI=66, NC=36
   - AD 樣本太少，模型可能學習到的是 MCI vs NC 的差異
   - 而不是 AD 的特異性標記

3. **跨 ROI 共線性**: 
   - 額葉區域之間高度相關
   - 模型可能選擇了這些相關特徵而忽略了獨立的 AD 標記

## 問題 3: SHAP 值的一致性

**觀察**:
```
sub-0005 (AD): Supp_Motor_Area_L_GM = -0.0742
sub-0010 (NC): Supp_Motor_Area_L_GM = -0.0737
sub-0015 (NC): Supp_Motor_Area_L_GM = -0.0735
```

**問題**:
- 不同診斷的受試者，SHAP 值**幾乎相同**
- 這表示模型可能依賴**全局模式**而非**個別化特徵**

**可能原因**:
- 特徵選擇太激進（只保留 6% 特徵）
- 保留的特徵可能是**穩定但不具區分性**的特徵

## 🔧 建議的修復方案

### 方案 1: 重新訓練 - 生物學導向的特徵選擇

```python
# 1. 手動選擇 AD 相關腦區
ad_relevant_rois = [
    'Hippocampus', 'Amygdala', 'Olfactory', 
    'ParaHippocampal', 'Cingulate_Post', 'Entorhinal',
    'Temporal_Mid', 'Temporal_Inf'
]

# 2. 只使用這些 ROI 的特徵
selected_features = []
for roi in ad_relevant_rois:
    for modality in ['_GM', '_FA', '_MD']:
        for side in ['_L', '_R']:
            feature = roi + side + modality
            if feature in all_features:
                selected_features.append(feature)

# 3. 訓練模型
X_selected = X[selected_features]
```

**優點**:
- 強制模型使用生物學相關的特徵
- 減少共線性問題
- 提高可解釋性

**缺點**:
- 可能降低準確率
- 需要領域知識

### 方案 2: 處理類別不平衡

```python
from imblearn.over_sampling import SMOTE

# 對 AD 類別進行過採樣
smote = SMOTE(sampling_strategy={0: 60, 1: 36})  # AD=60, NC=36
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**優點**:
- 增加 AD 樣本
- 模型可以更好地學習 AD 特徵

**缺點**:
- 合成樣本可能不真實
- 可能過擬合

### 方案 3: 減少跨 ROI 共線性

```python
# 1. 對於高相關的特徵對，只保留一個
high_corr_pairs = [
    ('Frontal_Sup_2_L_GM', 'Frontal_Mid_2_L_GM'),
    ('Frontal_Sup_2_L_GM', 'Frontal_Sup_2_R_GM'),
    ...
]

# 2. 移除其中一個
features_to_drop = [pair[1] for pair in high_corr_pairs]
X_reduced = X.drop(columns=features_to_drop)
```

**優點**:
- 減少冗餘特徵
- 提高模型穩定性

**缺點**:
- 可能丟失有用信息

### 方案 4: 只使用 GM 特徵

```python
# 只使用灰質特徵
gm_features = [col for col in X.columns if col.endswith('_GM')]
X_gm = X[gm_features]
```

**優點**:
- GM 是 AD 最直接的標記（萎縮）
- 減少特徵數量
- 避免模態間的鏡像效應

**缺點**:
- 丟失 FA 和 MD 的信息

### 方案 5: 使用 L1 正則化

```python
from sklearn.linear_model import LogisticRegression

# 使用 L1 正則化自動選擇特徵
model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
```

**優點**:
- 自動特徵選擇
- 傾向選擇獨立的特徵

**缺點**:
- 可能不如 Random Forest 準確

## 📊 診斷數據

### 共線性統計

- **高相關對** (|r| >= 0.9): 111 對
- **同一 ROI 內**: 0 對 ✓
- **跨 ROI**: 111 對 ⚠️

### 特徵選擇統計

- **原始特徵**: 498
- **選擇後**: 30 (6.0%)
- **AD 生物標記在 Top 10**: 2/10 (20%) ⚠️

### 類別分布

- **AD**: 21 (17%)
- **MCI**: 66 (54%)
- **NC**: 36 (29%)

## 🎯 推薦行動

### 立即行動（優先級高）

1. **重新訓練 - 只使用 GM 特徵**
   ```bash
   python scripts/cnn_rf/train_gm_only.py
   ```

2. **重新訓練 - 生物學導向特徵選擇**
   ```bash
   python scripts/cnn_rf/train_bio_features.py
   ```

3. **處理類別不平衡**
   ```bash
   python scripts/cnn_rf/train_balanced.py
   ```

### 中期行動

1. 收集更多 AD 樣本
2. 使用 SMOTE 過採樣
3. 嘗試其他模型（XGBoost, LightGBM）

### 長期行動

1. 整合臨床數據（MMSE, CDR）
2. 使用深度學習（3D CNN）
3. 多中心數據驗證

## 📚 參考資料

- [Collinearity Analysis](../output/cnn_rf/collinearity_analysis/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [AD Biomarkers Review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6380394/)
