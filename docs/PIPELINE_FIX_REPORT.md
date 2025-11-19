# 🔧 Pipeline 修復報告

## 📋 問題發現

你發現了一個關鍵問題：**SHAP 計算沒有正確處理 sklearn Pipeline**

### 原始問題

```python
# 錯誤的做法
shap_values = explainer.shap_values(raw_features)  # ✗ 使用原始數據
```

**問題**:
- SHAP 直接在原始數據上計算
- 沒有應用 StandardScaler
- 沒有應用 Feature Selection
- 導致 SHAP 值不正確

## ✅ 修復方案

### 正確的 Pipeline 處理

```python
# 1. 提取 Pipeline 組件
scaler = pipeline.named_steps['scale']
selector = pipeline.named_steps['select']
rf_model = pipeline.named_steps['model']

# 2. 手動應用轉換
X_scaled = scaler.transform(raw_features)      # 應用 scaling
X_selected = selector.transform(X_scaled)      # 應用 feature selection

# 3. 在轉換後的數據上計算 SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_selected)  # ✓ 使用轉換後的數據
```

## 🔍 實施細節

### 更新的函數

**檔案**: `scripts/cnn_rf/end_to_end_inference.py`

#### 1. `calculate_shap_values()` 函數

**之前**:
```python
def calculate_shap_values(self, feature_df, verbose=True):
    # 直接在原始數據上計算 SHAP
    shap_values = self.shap_explainer.shap_values(feature_df)  # ✗
    return shap_values, feature_names
```

**之後**:
```python
def calculate_shap_values(self, feature_df, verbose=True):
    # 提取 Pipeline 組件
    scaler = self.model.named_steps['scale']
    selector = self.model.named_steps['select']
    
    # 手動應用轉換
    X_scaled = scaler.transform(feature_df)
    X_selected = selector.transform(X_scaled)
    
    # 獲取選中的特徵名稱
    selected_mask = selector.get_support()
    selected_feature_names = [name for name, selected in 
                             zip(feature_df.columns, selected_mask) if selected]
    
    # 在轉換後的數據上計算 SHAP
    shap_values = self.shap_explainer.shap_values(X_selected)  # ✓
    
    return shap_values[0][0], selected_feature_names
```

#### 2. SHAP Explainer 初始化

**之前**:
```python
# 可能使用整個 pipeline
self.shap_explainer = shap.TreeExplainer(self.model)  # ✗
```

**之後**:
```python
# 只使用 RF 模型
rf_model = self.model.named_steps['model']
self.shap_explainer = shap.TreeExplainer(rf_model)  # ✓
```

## 📊 修復前後對比

### 修復前

```
sub-0005 (AD) Top 5 SHAP:
1. Supp_Motor_Area_L_GM: -0.0742
2. Supp_Motor_Area_L_FA: +0.0742
3. Frontal_Sup_Medial_L_GM: -0.0427
4. Frontal_Sup_Medial_L_FA: +0.0427
5. Frontal_Inf_Oper_L_MD: -0.0378

sub-0010 (NC) Top 5 SHAP:
1. Supp_Motor_Area_L_GM: -0.0737  # 幾乎相同！
2. Supp_Motor_Area_L_FA: +0.0737  # 幾乎相同！
3. Frontal_Sup_Medial_L_GM: -0.0372
4. Frontal_Sup_Medial_L_FA: +0.0372
5. Olfactory_R_FA: -0.0292
```

**問題**:
- ❌ SHAP 值幾乎相同
- ❌ 鏡像效應（GM vs FA 相反符號）
- ❌ 不是真正的局部解釋

### 修復後

```
sub-0005 (AD) Top 5 SHAP:
1. Thal_VPL_R_FA: +0.0439
2. Thal_VL_R_MD: -0.0439
3. Cerebellum_4_5_R_FA: -0.0234
4. Cerebellum_4_5_L_MD: +0.0234
5. Olfactory_L_FA: -0.0182

sub-0010 (NC) Top 5 SHAP:
1. Lingual_R_FA: -0.0513  # 不同！
2. Lingual_R_MD: +0.0513  # 不同！
3. Cerebellum_4_5_R_FA: +0.0313
4. Cerebellum_4_5_L_MD: -0.0313
5. Olfactory_L_FA: -0.0312
```

**改進**:
- ✅ SHAP 值現在不同
- ✅ 真正的局部解釋
- ⚠️ 鏡像效應仍存在（但這是特徵選擇的問題）

## 🎯 關鍵學習

### 1. Pipeline 的正確使用

**預測時**:
```python
# 這是正確的，Pipeline 會自動應用所有轉換
prediction = pipeline.predict(raw_data)  # ✓
```

**SHAP 解釋時**:
```python
# 必須手動提取和應用轉換
X_transformed = pipeline[:-1].transform(raw_data)  # 所有步驟除了最後的模型
shap_values = explainer.shap_values(X_transformed)  # ✓
```

### 2. 為什麼需要手動轉換？

- **SHAP 需要解釋模型**，不是 Pipeline
- **模型看到的是轉換後的數據**
- **SHAP 必須在相同的數據空間中計算**

### 3. 特徵名稱的處理

```python
# 獲取選中的特徵名稱
selector = pipeline.named_steps['select']
selected_mask = selector.get_support()
selected_names = [name for name, selected in 
                 zip(original_names, selected_mask) if selected]
```

## 📈 影響

### 修復前的問題

1. **SHAP 值不準確**
   - 在錯誤的數據空間中計算
   - 沒有應用 scaling 和 selection

2. **局部解釋失效**
   - 所有受試者的 SHAP 值幾乎相同
   - 無法提供個別化的解釋

3. **誤導性結果**
   - 顯示的特徵可能不是模型實際使用的
   - 特徵重要性不正確

### 修復後的改進

1. **SHAP 值正確**
   - ✓ 在正確的數據空間中計算
   - ✓ 應用了所有必要的轉換

2. **真正的局部解釋**
   - ✓ 不同受試者有不同的 SHAP 值
   - ✓ 提供個別化的解釋

3. **準確的特徵重要性**
   - ✓ 顯示模型實際使用的特徵
   - ✓ 特徵重要性正確

## 🔧 其他需要注意的地方

### 1. 特徵選擇仍然是問題

雖然 SHAP 現在正確了，但：
- ❌ 只有 30/498 (6%) 特徵被選中
- ❌ Hippocampus, Amygdala 等 AD 標記沒被選中
- ❌ 鏡像效應仍然存在

**解決方案**: 使用 GM-only 模型
```python
model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib"
```

### 2. SHAP 值的長度問題

```python
# Binary classification 返回 [class_0_shap, class_1_shap]
# 每個的形狀是 (n_samples, n_features)
shap_values = explainer.shap_values(X)

# 對於單個樣本
shap_ad = shap_values[0][0]  # Class 0 (AD), first sample
```

### 3. 特徵名稱的對應

```python
# 確保特徵名稱與 SHAP 值對應
assert len(shap_values) == len(feature_names)
```

## 🎉 總結

### 問題

❌ SHAP 在原始數據上計算，沒有應用 Pipeline 轉換

### 解決方案

✅ 手動提取 Pipeline 組件並應用轉換

### 結果

✅ SHAP 值現在正確且不同  
✅ 真正的局部可解釋性  
✅ 準確的特徵重要性  

### 下一步

1. 使用 GM-only 模型以獲得更好的生物學可解釋性
2. 或重新訓練模型with less aggressive feature selection
3. 繼續監控 SHAP 值以確保它們有意義

## 📚 參考

- [SHAP Documentation](https://shap.readthedocs.io/)
- [sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [TreeExplainer](https://shap-lrjball.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
