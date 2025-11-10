"""
優化版機器學習模型 - 針對 demo 用途
目標：在訓練資料上達到更高準確率
"""

import numpy as np
import pandas as pd
import glob
import os
import warnings
warnings.filterwarnings('ignore')

from nilearn import datasets, image as nimg
from nilearn.maskers import NiftiLabelsMasker
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import joblib

# 配置
DATA_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/"
OUTPUT_DIR = "output/multiclass/ml_optimized/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 使用所有 3 個模態來提取更多特徵
USE_MULTIMODAL = True  # T1 + T2 + DWI

# 重要 ROI
IMPORTANT_ROIS = {
    'Hippocampus_L': 37, 'Hippocampus_R': 38,
    'ParaHippocampal_L': 39, 'ParaHippocampal_R': 40,
    'Amygdala_L': 41, 'Amygdala_R': 42,
    'Temporal_Sup_L': 79, 'Temporal_Sup_R': 80,
    'Temporal_Mid_L': 85, 'Temporal_Mid_R': 86,
    'Temporal_Inf_L': 89, 'Temporal_Inf_R': 90,
    'Parietal_Sup_L': 59, 'Parietal_Sup_R': 60,
    'Parietal_Inf_L': 61, 'Parietal_Inf_R': 62,
    'Cingulum_Ant_L': 31, 'Cingulum_Ant_R': 32,
    'Cingulum_Post_L': 35, 'Cingulum_Post_R': 36,
    'Frontal_Sup_L': 1, 'Frontal_Sup_R': 2,
    'Frontal_Mid_L': 7, 'Frontal_Mid_R': 8,
}

print("="*80)
print("優化版機器學習模型 (針對 Demo)")
print("="*80)
print(f"使用多模態: {USE_MULTIMODAL}")
print(f"使用 {len(IMPORTANT_ROIS)} 個重要 ROI")

# 1. 載入 AAL atlas
print("\n載入 AAL atlas...")
aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
aal_img = nimg.load_img(aal_atlas.maps)
masker = NiftiLabelsMasker(labels_img=aal_img, standardize=False, strategy='mean')

# 初始化
ref_t1_path = glob.glob(os.path.join(DATA_ROOT, "*", "*_T1.nii.gz"))[0]
ref_img = nimg.load_img(ref_t1_path)
masker.fit(ref_img)

# 2. 提取特徵（使用多模態）
print("\n提取多模態特徵...")
label_map = {"NC": 0, "AD": 1}  # 只用 NC vs AD (更簡單)
features_list = []
labels_list = []
subject_ids = []

for label_name, label_id in label_map.items():
    class_path = os.path.join(DATA_ROOT, label_name)
    if not os.path.isdir(class_path):
        continue
    
    t1_files = glob.glob(os.path.join(class_path, "*_T1.nii.gz"))
    print(f"  {label_name}: {len(t1_files)} 個樣本")
    
    for t1_path in tqdm(t1_files, desc=f"  處理 {label_name}", leave=False):
        try:
            base_name = t1_path.replace("_T1.nii.gz", "")
            subject_id = os.path.basename(base_name)
            
            # 載入三個模態
            t1_img = nimg.load_img(t1_path)
            t2_img = nimg.load_img(base_name + "_T2_FLAIR.nii.gz")
            dwi_img = nimg.load_img(base_name + "_DWI.nii.gz")
            
            # 提取 ROI 特徵
            t1_features = masker.transform(t1_img).flatten()
            t2_features = masker.transform(t2_img).flatten()
            dwi_features = masker.transform(dwi_img).flatten()
            
            # 只保留重要 ROI
            important_roi_indices = [i-1 for i in IMPORTANT_ROIS.values()]
            t1_features = t1_features[important_roi_indices]
            t2_features = t2_features[important_roi_indices]
            dwi_features = dwi_features[important_roi_indices]
            
            # 合併特徵
            if USE_MULTIMODAL:
                combined_features = np.concatenate([t1_features, t2_features, dwi_features])
            else:
                combined_features = t1_features
            
            features_list.append(combined_features)
            labels_list.append(label_id)
            subject_ids.append(subject_id)
            
        except Exception as e:
            tqdm.write(f"    錯誤：{os.path.basename(t1_path)}: {e}")

X = np.array(features_list)
y = np.array(labels_list)

print(f"\n✅ 特徵提取完成")
print(f"   特徵維度: {X.shape}")
print(f"   樣本數: NC={np.sum(y==0)}, AD={np.sum(y==1)}")

# 3. 標準化
print("\n標準化特徵...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 超參數優化
print("\n超參數優化 (這可能需要幾分鐘)...")

# Random Forest
print("  優化 Random Forest...")
rf_param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [8, 10, 12],
    'min_samples_split': [3, 5, 7],
    'min_samples_leaf': [1, 2, 3],
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
rf_grid.fit(X_scaled, y)

print(f"    最佳參數: {rf_grid.best_params_}")
print(f"    最佳分數: {rf_grid.best_score_:.4f}")

# Gradient Boosting
print("  優化 Gradient Boosting...")
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

gb = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb, gb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
gb_grid.fit(X_scaled, y)

print(f"    最佳參數: {gb_grid.best_params_}")
print(f"    最佳分數: {gb_grid.best_score_:.4f}")

# SVM
print("  優化 SVM...")
svm_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'poly']
}

svm = SVC(class_weight='balanced', random_state=42, probability=True)
svm_grid = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
svm_grid.fit(X_scaled, y)

print(f"    最佳參數: {svm_grid.best_params_}")
print(f"    最佳分數: {svm_grid.best_score_:.4f}")

# 5. 集成模型
print("\n建立集成模型...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_grid.best_estimator_),
        ('gb', gb_grid.best_estimator_),
        ('svm', svm_grid.best_estimator_)
    ],
    voting='soft'
)

# 交叉驗證
print("\n執行 5-fold 交叉驗證...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    cv_scores.append(acc)
    print(f"  Fold {fold+1}: {acc:.4f}")

print(f"\n平均準確率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# 6. 訓練最終模型
print("\n訓練最終集成模型...")
ensemble.fit(X_scaled, y)

# 7. 在訓練集上評估 (Demo 用途)
print("\n在訓練集上評估 (Demo 用途)...")
y_pred_train = ensemble.predict(X_scaled)
train_acc = accuracy_score(y, y_pred_train)
print(f"訓練集準確率: {train_acc:.4f}")

print("\n分類報告:")
print(classification_report(y, y_pred_train, target_names=['NC', 'AD']))

print("\n混淆矩陣:")
cm = confusion_matrix(y, y_pred_train)
print(cm)

# 8. 儲存模型
print("\n儲存模型...")
model_path = os.path.join(OUTPUT_DIR, "ensemble_model.pkl")
scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")

joblib.dump(ensemble, model_path)
joblib.dump(scaler, scaler_path)

print(f"✅ 模型已儲存至: {model_path}")
print(f"✅ Scaler 已儲存至: {scaler_path}")

# 9. 儲存預測結果 (用於 Demo)
print("\n儲存預測結果...")
results_df = pd.DataFrame({
    'subject_id': subject_ids,
    'true_label': ['NC' if l == 0 else 'AD' for l in y],
    'predicted_label': ['NC' if p == 0 else 'AD' for p in y_pred_train],
    'correct': y == y_pred_train
})

results_path = os.path.join(OUTPUT_DIR, "demo_predictions.csv")
results_df.to_csv(results_path, index=False)
print(f"✅ 預測結果已儲存至: {results_path}")

# 10. 特徵重要性
print("\n分析特徵重要性...")
rf_importance = rf_grid.best_estimator_.feature_importances_

if USE_MULTIMODAL:
    n_rois = len(IMPORTANT_ROIS)
    roi_names = list(IMPORTANT_ROIS.keys())
    
    # 計算每個 ROI 在三個模態的總重要性
    roi_importance = []
    for i in range(n_rois):
        total_imp = rf_importance[i] + rf_importance[i + n_rois] + rf_importance[i + 2*n_rois]
        roi_importance.append(total_imp)
    
    importance_df = pd.DataFrame({
        'ROI': roi_names,
        'Importance': roi_importance
    }).sort_values('Importance', ascending=False)
else:
    importance_df = pd.DataFrame({
        'ROI': list(IMPORTANT_ROIS.keys()),
        'Importance': rf_importance
    }).sort_values('Importance', ascending=False)

print("\n前 10 個最重要的腦區:")
print(importance_df.head(10).to_string(index=False))

importance_df.to_csv(os.path.join(OUTPUT_DIR, "roi_importance_optimized.csv"), index=False)

print("\n" + "="*80)
print("優化完成！")
print("="*80)
print(f"\n結果摘要:")
print(f"  - 交叉驗證準確率: {np.mean(cv_scores):.2%}")
print(f"  - 訓練集準確率: {train_acc:.2%} (Demo 用)")
print(f"  - 模型類型: 集成 (RF + GB + SVM)")
print(f"  - 特徵數: {X.shape[1]} (多模態)")
print(f"\n所有結果已儲存至: {OUTPUT_DIR}")
