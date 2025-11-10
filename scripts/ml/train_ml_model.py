"""
機器學習模型訓練腳本
使用 Random Forest 分析重要腦區 (NC vs AD)

輸出：
- model/ml/rf_model.pkl
- model/ml/scaler.pkl
- output/ml/training_results.csv
- output/ml/roi_importance.csv
"""

import numpy as np
import pandas as pd
import glob
import os
import warnings
warnings.filterwarnings('ignore')

from nilearn import datasets, image as nimg
from nilearn.maskers import NiftiLabelsMasker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import joblib

# ====================================================================
# 配置
# ====================================================================
DATA_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/"
MODEL_DIR = "model/ml/"
OUTPUT_DIR = "output/ml/"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 重要 ROI (根據文獻)
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
print("機器學習模型訓練 (NC vs AD)")
print("="*80)
print(f"資料來源: {DATA_ROOT}")
print(f"模型輸出: {MODEL_DIR}")
print(f"結果輸出: {OUTPUT_DIR}")
print(f"使用 {len(IMPORTANT_ROIS)} 個重要 ROI")

# ====================================================================
# 1. 載入 AAL Atlas
# ====================================================================
print("\n載入 AAL atlas...")
aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
aal_img = nimg.load_img(aal_atlas.maps)
masker = NiftiLabelsMasker(labels_img=aal_img, standardize=False, strategy='mean')

# 初始化 masker
ref_t1_path = glob.glob(os.path.join(DATA_ROOT, "*", "*_T1.nii.gz"))[0]
ref_img = nimg.load_img(ref_t1_path)
masker.fit(ref_img)
print("✅ AAL atlas 載入完成")

# ====================================================================
# 2. 提取特徵
# ====================================================================
print("\n提取 ROI 特徵...")
label_map = {"NC": 0, "AD": 1}
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
            t1_img = nimg.load_img(t1_path)
            roi_features = masker.transform(t1_img).flatten()
            important_roi_indices = [i-1 for i in IMPORTANT_ROIS.values()]
            roi_features = roi_features[important_roi_indices]
            
            features_list.append(roi_features)
            labels_list.append(label_id)
            subject_ids.append(os.path.basename(t1_path).replace("_T1.nii.gz", ""))
        except Exception as e:
            tqdm.write(f"    錯誤：{os.path.basename(t1_path)}: {e}")

X = np.array(features_list)
y = np.array(labels_list)

print(f"\n✅ 特徵提取完成")
print(f"   特徵維度: {X.shape}")
print(f"   樣本數: NC={np.sum(y==0)}, AD={np.sum(y==1)}")

# ====================================================================
# 3. 標準化
# ====================================================================
print("\n標準化特徵...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   特徵範圍: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

# ====================================================================
# 4. 訓練模型
# ====================================================================
print("\n訓練 Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# 5-fold 交叉驗證
print("\n執行 5-fold 交叉驗證...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_scaled, y, cv=kfold, scoring='accuracy', n_jobs=-1)

print(f"\n交叉驗證結果:")
print(f"  平均準確率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  各 fold: {cv_scores}")

# ====================================================================
# 5. 訓練最終模型
# ====================================================================
print("\n訓練最終模型...")
rf.fit(X_scaled, y)

# 在訓練集上評估
y_pred = rf.predict(X_scaled)
train_acc = accuracy_score(y, y_pred)

print(f"\n訓練集準確率: {train_acc:.4f}")
print("\n分類報告:")
print(classification_report(y, y_pred, target_names=['NC', 'AD']))

# ====================================================================
# 6. 特徵重要性分析
# ====================================================================
print("\n分析特徵重要性...")
feature_importance = rf.feature_importances_
roi_names = list(IMPORTANT_ROIS.keys())

importance_df = pd.DataFrame({
    'ROI': roi_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n前 10 個最重要的腦區:")
print(importance_df.head(10).to_string(index=False))

# ====================================================================
# 7. 儲存結果
# ====================================================================
print("\n儲存模型和結果...")

# 儲存模型
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print(f"✅ 模型已儲存至: {MODEL_DIR}")

# 儲存 ROI 重要性
importance_df.to_csv(os.path.join(OUTPUT_DIR, "roi_importance.csv"), index=False)
print(f"✅ ROI 重要性已儲存至: {OUTPUT_DIR}roi_importance.csv")

# 儲存訓練結果
results_df = pd.DataFrame({
    'subject_id': subject_ids,
    'true_label': ['NC' if l == 0 else 'AD' for l in y],
    'predicted_label': ['NC' if p == 0 else 'AD' for p in y_pred],
    'correct': y == y_pred
})
results_df.to_csv(os.path.join(OUTPUT_DIR, "training_results.csv"), index=False)
print(f"✅ 訓練結果已儲存至: {OUTPUT_DIR}training_results.csv")

# 儲存訓練摘要
summary = {
    'cv_mean_accuracy': cv_scores.mean(),
    'cv_std_accuracy': cv_scores.std(),
    'train_accuracy': train_acc,
    'n_samples': len(y),
    'n_features': X.shape[1],
    'n_nc': np.sum(y==0),
    'n_ad': np.sum(y==1),
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(OUTPUT_DIR, "training_summary.csv"), index=False)
print(f"✅ 訓練摘要已儲存至: {OUTPUT_DIR}training_summary.csv")

print("\n" + "="*80)
print("訓練完成！")
print("="*80)
print(f"\n結果摘要:")
print(f"  - 交叉驗證準確率: {cv_scores.mean():.2%}")
print(f"  - 訓練集準確率: {train_acc:.2%}")
print(f"  - 最重要的腦區: {importance_df.iloc[0]['ROI']}")
print(f"\n所有檔案已儲存至:")
print(f"  - 模型: {MODEL_DIR}")
print(f"  - 結果: {OUTPUT_DIR}")
