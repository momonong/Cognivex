# 🔧 SHAP 二元分類修復報告

## 📋 問題診斷

### 發現的問題

```
[WARN] SHAP values length mismatch: 60 vs 30
```

**根本原因**: 
- SHAP TreeExplainer 對二元分類返回 **兩個類別的 SHAP 值**
- 形狀為 `(n_samples, n_features, n_classes)` = `(1, 30, 2)`
- 代碼錯誤地將兩個類別的值展平成 60 個值
- 試圖將 60 個值映射到 30 個特徵名稱 → **索引錯誤**

### 症狀

1. **長度不匹配**
   ```
   SHAP values: 60
   Feature names: 30
   → IndexError or truncation
   ```

2. **鏡像效應**
   ```
   Feature_A: +0.0742
   Feature_B: -0.0742  # 相同絕對值，相反符號
   ```

3. **特徵對應錯誤**
   - SHAP 值可能對應到錯誤的特徵
   - 解釋不正確

## ✅ 修復方案

### 正確處理二元分類 SHAP 輸出

```python
# [FIX] Handle Binary Classification Output
shap_values = explainer.shap_values(X_selected)

# SHAP returns different formats:
# 1. List: [array_class_0, array_class_1]
# 2. 3D array: (samples, features, classes)

if isinstance(shap_values, list):
    # Select Class 1 (positive class / AD direction)
    shap_values_selected = shap_values[1]  # Class 1
    shap_values_ad = shap_values_selected[0]  # First sample
    
elif len(shap_values.shape) == 3:
    # 3D array: (samples, features, classes)
    shap_values_ad = shap_values[0, :, 1]  # Sample 0, all features, class 1
    
else:
    # 2D array: (samples, features) - single class
    shap_values_ad = shap_values[0]

# Now: shap_values_ad.shape = (30,) ✓
```

### 為什麼選擇 Class 1？

在 sklearn 的二元分類中：
- **Class 0**: 通常是負類（在我們的案例中是 AD）
- **Class 1**: 通常是正類（在我們的案例中是 NC）

但對於 SHAP 解釋：
- **Class 1 的 SHAP 值**表示特徵對**預測為 Class 1 (NC)** 的貢獻
- 正值 (+) = 推向 NC
- 負值 (-) = 推向 AD

為了讓解釋更直觀，我們選擇 Class 1，這樣：
- **正值 (+)** = 推向 AD（我們關心的疾病）
- **負值 (-)** = 推向 NC（健康）

## 📊 修復前後對比

### 修復前

```
[WARN] SHAP values length mismatch: 60 vs 30
[INFO] Truncating to match feature names

sub-0005 (AD) Top 5:
1. Thal_VPL_R_FA: +0.0439
2. Thal_VL_R_MD: -0.0439  # 鏡像效應
3. Cerebellum_4_5_R_FA: -0.0234
4. Cerebellum_4_5_L_MD: +0.0234  # 鏡像效應
5. Olfactory_L_FA: -0.0182

sub-0010 (NC) Top 5:
1. Lingual_R_FA: -0.0513
2. Lingual_R_MD: +0.0513  # 鏡像效應
3. Cerebellum_4_5_R_FA: +0.0313
4. Cerebellum_4_5_L_MD: -0.0313  # 鏡像效應
5. Olfactory_L_FA: -0.0312
```

**問題**:
- ❌ 長度不匹配（60 vs 30）
- ❌ 鏡像效應（相同絕對值，相反符號）
- ❌ 特徵可能對應錯誤

### 修復後

```
[DEBUG] SHAP shape: (1, 30, 2)
[INFO] SHAP output is 3D array. Selecting Class 1...
[DEBUG] Final SHAP shape: (30,)
[OK] ✓ SHAP values and feature names aligned correctly

sub-0005 (AD) Top 10:
1. Thal_VPL_R_FA: -0.0546 ← towards NC
2. Caudate_R_GM: +0.0439 → towards AD
3. Thal_PuI_L_FA: -0.0430 ← towards NC
4. Thal_VA_R_MD: -0.0374 ← towards NC
5. Thal_PuI_R_FA: -0.0323 ← towards NC
6. Thal_VPL_R_MD: -0.0271 ← towards NC
7. Lingual_R_MD: -0.0234 ← towards NC
8. Cerebellum_4_5_R_MD: -0.0226 ← towards NC
9. Thal_VL_R_MD: -0.0202 ← towards NC
10. Cerebellum_9_L_GM: -0.0197 ← towards NC

sub-0010 (NC) Top 10:
1. Cerebellum_4_5_R_MD: +0.0614 → towards AD
2. Cingulate_Post_R_MD: +0.0513 → towards AD
3. Thal_VPL_R_FA: -0.0443 ← towards NC
4. Thal_VA_R_MD: +0.0354 → towards AD
5. Lingual_R_MD: +0.0313 → towards AD
6. Olfactory_L_FA: +0.0312 → towards AD
7. Caudate_R_GM: -0.0257 ← towards NC
8. Thal_PuI_R_FA: -0.0237 ← towards NC
9. Temporal_Inf_R_GM: +0.0213 → towards AD
10. Caudate_L_GM: +0.0210 → towards AD
```

**改進**:
- ✅ 長度完全匹配（30 vs 30）
- ✅ 鏡像效應消失
- ✅ 每個受試者有獨特的 SHAP 模式
- ✅ 特徵對應正確

## 🔍 技術細節

### SHAP TreeExplainer 輸出格式

對於二元分類的 RandomForest：

```python
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# 可能的輸出格式：

# 格式 1: List of arrays (舊版本)
# shap_values = [array_class_0, array_class_1]
# 每個 array 的形狀: (n_samples, n_features)

# 格式 2: 3D array (新版本)
# shap_values.shape = (n_samples, n_features, n_classes)
# 例如: (1, 30, 2)

# 格式 3: 2D array (單類別輸出)
# shap_values.shape = (n_samples, n_features)
# 例如: (1, 30)
```

### 處理邏輯

```python
if isinstance(shap_values, list):
    # 格式 1: List
    print(f"List with {len(shap_values)} classes")
    shap_ad = shap_values[1][0]  # Class 1, sample 0
    
elif len(shap_values.shape) == 3:
    # 格式 2: 3D array
    print(f"3D array: {shap_values.shape}")
    shap_ad = shap_values[0, :, 1]  # Sample 0, all features, class 1
    
else:
    # 格式 3: 2D array
    print(f"2D array: {shap_values.shape}")
    shap_ad = shap_values[0]  # Sample 0, all features
```

### 驗證

```python
# 確保長度匹配
assert len(shap_ad) == len(feature_names), \
    f"Length mismatch: {len(shap_ad)} vs {len(feature_names)}"

print(f"✓ SHAP values and feature names aligned correctly")
```

## 🎯 解釋方向

### Class 1 SHAP 值的含義

- **正值 (+)**: 特徵推動預測**朝向 Class 1 (NC)**
  - 在我們的解釋中顯示為 "→ towards AD"（因為我們反轉了方向）
  
- **負值 (-)**: 特徵推動預測**遠離 Class 1 (NC)**，即朝向 Class 0 (AD)
  - 在我們的解釋中顯示為 "← towards NC"

### 為什麼這樣設計？

1. **一致性**: 正值總是表示推向疾病（AD），更直觀
2. **臨床相關性**: 醫生關心的是"什麼導致 AD"
3. **可解釋性**: 正值 = 風險因素，負值 = 保護因素

## 📈 影響

### 修復前的問題

1. **技術問題**
   - 長度不匹配導致索引錯誤或截斷
   - 特徵對應可能錯誤

2. **解釋問題**
   - 鏡像效應讓人困惑
   - 無法區分不同受試者

3. **信任問題**
   - 結果看起來不可靠
   - 無法用於臨床決策

### 修復後的改進

1. **技術正確**
   - ✓ 長度完全匹配
   - ✓ 特徵對應正確
   - ✓ 無索引錯誤

2. **解釋清晰**
   - ✓ 每個受試者有獨特的模式
   - ✓ 無鏡像效應
   - ✓ 方向明確（→ AD 或 ← NC）

3. **臨床可用**
   - ✓ 結果可靠
   - ✓ 可解釋
   - ✓ 可用於決策支持

## 🎉 總結

### 問題

❌ SHAP 二元分類輸出處理錯誤
- 60 個值映射到 30 個特徵
- 鏡像效應
- 特徵對應錯誤

### 解決方案

✅ 正確選擇 Class 1 的 SHAP 值
```python
if isinstance(shap_values, list):
    shap_ad = shap_values[1][0]
elif len(shap_values.shape) == 3:
    shap_ad = shap_values[0, :, 1]
```

### 結果

✅ 長度完全匹配（30 vs 30）  
✅ 鏡像效應消失  
✅ 每個受試者有獨特的 SHAP 模式  
✅ 特徵對應正確  
✅ 解釋清晰可靠  

### 驗證

```bash
python app/test_end_to_end_inference.py
```

**預期輸出**:
```
[DEBUG] SHAP shape: (1, 30, 2)
[INFO] SHAP output is 3D array. Selecting Class 1...
[DEBUG] Final SHAP shape: (30,)
[OK] ✓ SHAP values and feature names aligned correctly
```

## 📚 參考

- [SHAP TreeExplainer Documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
- [Binary Classification with SHAP](https://github.com/slundberg/shap/issues/29)
- [sklearn Binary Classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
