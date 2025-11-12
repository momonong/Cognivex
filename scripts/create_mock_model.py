"""
創建 Mock 模型用於測試
如果真實模型檔案損壞，可以使用這個腳本創建一個測試用的模型
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

print("="*70)
print("🔧 創建 Mock ML 模型")
print("="*70)

# 創建目標目錄
model_dir = Path("model/ml/final")
model_dir.mkdir(parents=True, exist_ok=True)

# 1. 創建 Random Forest 模型
print("\n[1] 創建 Random Forest 模型...")
n_features = 32  # AAL ROI features
n_samples = 100  # Mock training samples

# 創建 mock 訓練數據
X_train = np.random.randn(n_samples, n_features)
y_train = np.random.choice([0, 1], size=n_samples)  # 0=NC, 1=AD

# 訓練模型
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# 保存模型
model_path = model_dir / "final_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ 模型已保存: {model_path}")
print(f"   - n_estimators: {model.n_estimators}")
print(f"   - n_features: {model.n_features_in_}")
print(f"   - classes: {model.classes_}")

# 2. 創建 Scaler
print("\n[2] 創建 Feature Scaler...")
scaler = StandardScaler()
scaler.fit(X_train)

scaler_path = model_dir / "final_scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Scaler 已保存: {scaler_path}")
print(f"   - mean shape: {scaler.mean_.shape}")
print(f"   - scale shape: {scaler.scale_.shape}")

# 3. 創建 ROI 列表
print("\n[3] 創建 ROI 列表...")
roi_names = [
    "Precentral_L", "Precentral_R",
    "Frontal_Sup_L", "Frontal_Sup_R",
    "Frontal_Mid_L", "Frontal_Mid_R",
    "Hippocampus_L", "Hippocampus_R",
    "Cingulum_Post_L", "Cingulum_Post_R",
    "Temporal_Mid_L", "Temporal_Mid_R",
    "Fusiform_L", "Fusiform_R",
    "Occipital_Mid_L", "Occipital_Mid_R",
    "Parietal_Sup_L", "Parietal_Sup_R",
    "Parietal_Inf_L", "Parietal_Inf_R",
    "Precuneus_L", "Precuneus_R",
    "Caudate_L", "Caudate_R",
    "Putamen_L", "Putamen_R",
    "Thalamus_L", "Thalamus_R",
    "Insula_L", "Insula_R",
    "Amygdala_L", "Amygdala_R"
]

roi_list_path = model_dir / "final_roi_list.csv"
with open(roi_list_path, 'w') as f:
    f.write("ROI_Name\n")
    for roi in roi_names:
        f.write(f"{roi}\n")

print(f"✅ ROI 列表已保存: {roi_list_path}")
print(f"   - 總共 {len(roi_names)} 個 ROI")

# 4. 創建特徵名稱
print("\n[4] 創建特徵名稱...")
feature_names_path = model_dir / "final_feature_names.txt"
with open(feature_names_path, 'w') as f:
    for roi in roi_names:
        f.write(f"{roi}\n")

print(f"✅ 特徵名稱已保存: {feature_names_path}")

# 5. 測試載入
print("\n[5] 測試載入...")
try:
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    print(f"✅ 模型載入成功")
    print(f"   Type: {type(loaded_model)}")
    
    with open(scaler_path, 'rb') as f:
        loaded_scaler = pickle.load(f)
    print(f"✅ Scaler 載入成功")
    
    # 測試預測
    X_test = np.random.randn(1, n_features)
    X_scaled = loaded_scaler.transform(X_test)
    prediction = loaded_model.predict(X_scaled)
    proba = loaded_model.predict_proba(X_scaled)
    
    print(f"\n✅ 測試預測成功")
    print(f"   Prediction: {'AD' if prediction[0] == 1 else 'NC'}")
    print(f"   Probability: NC={proba[0][0]:.2%}, AD={proba[0][1]:.2%}")
    
except Exception as e:
    print(f"❌ 測試失敗: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ Mock 模型創建完成！")
print("="*70)

print("\n⚠️  注意:")
print("   這是一個 MOCK 模型，僅用於測試系統功能")
print("   預測結果是隨機的，不具有臨床意義")
print("   請使用真實訓練的模型進行實際分析")

print("\n下一步:")
print("   1. 重新啟動 Streamlit: streamlit run app.py")
print("   2. 選擇 Structural MRI (T1) 模式")
print("   3. 測試分析功能")

print("\n" + "="*70)
