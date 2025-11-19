# SHAP 局部可解釋性整合

## 📋 概述

Cognivex 系統現在整合了 **SHAP (SHapley Additive exPlanations)** 來提供**局部可解釋性**（Local Explainability），為每個受試者計算個別的特徵貢獻，而不是使用全局特徵重要性。

## 🎯 問題解決

### 之前的問題

所有受試者顯示相同的特徵重要性：

```
Top 5 Important Features:
  1. Feature 21: 0.0901 (9.01%)
  2. Feature 10: 0.0687 (6.87%)
  3. Feature 9: 0.0585 (5.85%)
  4. Feature 19: 0.0551 (5.51%)
  5. Feature 27: 0.0525 (5.25%)
```

**原因**: 這是**全局特徵重要性**（Global Feature Importance），對所有受試者都一樣。

### 現在的解決方案

每個受試者有**個別化的特徵貢獻**：

**受試者 sub-0005 (AD)**:
```
Top 5 Features for This Subject (SHAP):
  1. Supp_Motor_Area_L_GM
     SHAP: -0.0742 ← towards NC
  2. Supp_Motor_Area_L_FA
     SHAP: +0.0742 → towards AD
  3. Frontal_Sup_Medial_L_GM
     SHAP: -0.0427 ← towards NC
  4. Frontal_Sup_Medial_L_FA
     SHAP: +0.0427 → towards AD
  5. Frontal_Inf_Oper_L_MD
     SHAP: -0.0378 ← towards NC
```

**受試者 sub-0010 (NC)**:
```
Top 5 Features for This Subject (SHAP):
  1. Supp_Motor_Area_L_GM
     SHAP: -0.0737 ← towards NC
  2. Supp_Motor_Area_L_FA
     SHAP: +0.0737 → towards AD
  3. Frontal_Sup_Medial_L_GM
     SHAP: -0.0372 ← towards NC
  4. Frontal_Sup_Medial_L_FA
     SHAP: +0.0372 → towards AD
  5. Olfactory_R_FA
     SHAP: -0.0292 ← towards NC
```

## 🔍 SHAP 值解釋

### SHAP 值的意義

- **正值 (+)**: 該特徵推動預測**朝向 AD**
- **負值 (-)**: 該特徵推動預測**朝向 NC**
- **絕對值大小**: 表示該特徵對此次預測的**影響程度**

### 範例解釋

對於 sub-0005 (AD 患者):

```
Supp_Motor_Area_L_FA: +0.0742 → towards AD
```

**解釋**: 
- 左側輔助運動區的 FA（分數各向異性）值
- SHAP 值為 +0.0742
- 這個特徵**強烈推動**模型預測為 AD
- 這是該受試者最重要的 AD 指標

```
Supp_Motor_Area_L_GM: -0.0742 ← towards NC
```

**解釋**:
- 左側輔助運動區的 GM（灰質）值
- SHAP 值為 -0.0742
- 這個特徵**推動**模型預測為 NC
- 但整體預測仍為 AD（因為其他特徵的綜合影響）

## 🛠️ 技術實作

### 1. SHAP 整合

**檔案**: `scripts/cnn_rf/end_to_end_inference.py`

```python
# 初始化 SHAP explainer
import shap

if hasattr(self.model, 'named_steps'):
    rf_model = self.model.named_steps['model']
else:
    rf_model = self.model

self.shap_explainer = shap.TreeExplainer(rf_model)
```

### 2. 計算 SHAP 值

```python
def calculate_shap_values(self, feature_df, verbose=True):
    """Calculate SHAP values for local explainability"""
    
    # Get feature names
    feature_names = list(feature_df.columns)
    
    # Calculate SHAP values
    shap_values = self.shap_explainer.shap_values(feature_df)
    
    # For binary classification, use AD class (class 0)
    if isinstance(shap_values, list):
        shap_values_ad = shap_values[0][0]
    else:
        shap_values_ad = shap_values[0]
    
    return shap_values_ad, feature_names
```

### 3. 提取 Top 特徵

```python
def get_top_shap_features(self, shap_values, feature_names, top_n=10):
    """Get top features by SHAP value"""
    
    # Get absolute SHAP values for ranking
    abs_shap = np.abs(shap_values)
    
    # Get top indices
    top_indices = np.argsort(abs_shap)[-top_n:][::-1]
    
    # Create feature info list
    top_features = []
    for i in range(len(top_indices)):
        idx = top_indices[i]
        feature_info = {
            'name': feature_names[idx],
            'shap_value': float(shap_values[idx]),
            'abs_shap_value': float(abs_shap[idx]),
            'direction': 'towards AD' if shap_values[idx] > 0 else 'towards NC',
            'impact': 'High' if abs_shap[idx] > np.mean(abs_shap) else 'Medium'
        }
        top_features.append(feature_info)
    
    return top_features
```

## 📊 輸出格式

### 結果結構

```python
{
    'subject_id': 'sub-0005',
    'predicted_label': 'AD',
    'confidence': 0.85,
    'shap_features': [
        {
            'name': 'Supp_Motor_Area_L_GM',
            'shap_value': -0.0742,
            'abs_shap_value': 0.0742,
            'direction': 'towards NC',
            'impact': 'High'
        },
        {
            'name': 'Supp_Motor_Area_L_FA',
            'shap_value': 0.0742,
            'abs_shap_value': 0.0742,
            'direction': 'towards AD',
            'impact': 'High'
        },
        ...
    ]
}
```

## 🎯 使用方法

### 方法 1: 通過 Agent

```python
from app.agents.cnn_rf_inference import run_cnn_rf_inference

state = {
    'subject_id': 'sub-0005',
    'model_name': 'NC_vs_AD',
    'data_root': 'data/MRI_processed'
}

result = run_cnn_rf_inference(state)

# 獲取 SHAP 特徵
shap_features = result['shap_features']
for feat in shap_features[:5]:
    print(f"{feat['name']}: {feat['shap_value']:+.4f} ({feat['direction']})")
```

### 方法 2: 直接使用 API

```python
from scripts.cnn_rf.end_to_end_inference import EndToEndPredictor

predictor = EndToEndPredictor()
results = predictor.predict_subject('sub-0005')

# SHAP 特徵已包含在結果中
shap_features = results['shap_features']
```

## 🧪 測試

```bash
python app/test_end_to_end_inference.py
```

**預期輸出**:
```
[3/3] Analyzing local feature importance (SHAP)...
   ✓ SHAP analysis complete

   🎯 Top 5 Features for This Subject (SHAP):
      1. Supp_Motor_Area_L_GM
         SHAP: -0.0742 ← towards NC
      2. Supp_Motor_Area_L_FA
         SHAP: +0.0742 → towards AD
      3. Frontal_Sup_Medial_L_GM
         SHAP: -0.0427 ← towards NC
      4. Frontal_Sup_Medial_L_FA
         SHAP: +0.0427 → towards AD
      5. Frontal_Inf_Oper_L_MD
         SHAP: -0.0378 ← towards NC
```

## 📈 優勢

### 1. 個別化解釋

每個受試者有**獨特的特徵貢獻**，不是所有人都一樣。

### 2. 臨床可解釋性

醫生可以看到：
- 哪些腦區對**這個特定患者**的診斷最重要
- 每個特徵是推向 AD 還是 NC
- 特徵的影響程度

### 3. 特徵名稱清晰

顯示完整的 ROI 名稱（如 `Supp_Motor_Area_L_GM`）而不是 "Feature 21"。

### 4. 方向性指示

- `→ towards AD`: 推向阿茲海默症
- `← towards NC`: 推向正常對照

## 🔧 依賴項

```bash
pip install shap
```

**版本**: SHAP 0.50.0 或更高

## 📚 相關文件

- [端到端遷移報告](END_TO_END_MIGRATION.md)
- [CNN-RF 整合文檔](CNN_RF_INTEGRATION.md)
- [SHAP 官方文檔](https://shap.readthedocs.io/)

## 🎉 總結

SHAP 整合為 Cognivex 系統帶來：

✅ **局部可解釋性** - 每個受試者的個別化特徵貢獻  
✅ **臨床可解釋性** - 清晰的腦區名稱和方向指示  
✅ **透明度** - 醫生可以理解模型的決策過程  
✅ **信任度** - 提供可驗證的診斷依據  

系統現在不僅能預測，還能**解釋為什麼**這樣預測！
