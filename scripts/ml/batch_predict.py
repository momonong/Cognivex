"""
批次預測腳本 - 對所有資料進行預測並生成統計報表

輸出：
- output/ml/batch_predictions.csv - 所有樣本的預測結果
- output/ml/prediction_report.txt - 詳細統計報表
- output/ml/confidence_analysis.csv - 信心度分析
"""

import numpy as np
import pandas as pd
import glob
import os
import joblib
from nilearn import datasets, image as nimg
from nilearn.maskers import NiftiLabelsMasker
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# 配置
# ====================================================================
DATA_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/"
MODEL_DIR = "model/ml/"
OUTPUT_DIR = "output/ml/"

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
print("批次預測與統計分析")
print("="*80)

# ====================================================================
# 1. 載入模型
# ====================================================================
print("\n載入模型...")
model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
print("✅ 模型載入成功")

# ====================================================================
# 2. 初始化 AAL Atlas
# ====================================================================
print("\n初始化 AAL atlas...")
aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
aal_img = nimg.load_img(aal_atlas.maps)
masker = NiftiLabelsMasker(labels_img=aal_img, standardize=False, strategy='mean')

ref_t1_path = glob.glob(os.path.join(DATA_ROOT, "*", "*_T1.nii.gz"))[0]
ref_img = nimg.load_img(ref_t1_path)
masker.fit(ref_img)
print("✅ AAL atlas 初始化完成")

# ====================================================================
# 3. 批次預測
# ====================================================================
print("\n開始批次預測...")
label_map = {"NC": 0, "AD": 1}
results = []

for label_name, true_label in label_map.items():
    class_path = os.path.join(DATA_ROOT, label_name)
    if not os.path.isdir(class_path):
        continue
    
    t1_files = glob.glob(os.path.join(class_path, "*_T1.nii.gz"))
    print(f"\n處理 {label_name}: {len(t1_files)} 個樣本")
    
    for t1_path in tqdm(t1_files, desc=f"  預測 {label_name}", leave=False):
        try:
            # 提取特徵
            t1_img = nimg.load_img(t1_path)
            roi_features = masker.transform(t1_img).flatten()
            important_roi_indices = [i-1 for i in IMPORTANT_ROIS.values()]
            roi_features = roi_features[important_roi_indices].reshape(1, -1)
            
            # 標準化
            features_scaled = scaler.transform(roi_features)
            
            # 預測
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            # 記錄結果
            subject_id = os.path.basename(t1_path).replace("_T1.nii.gz", "")
            results.append({
                'subject_id': subject_id,
                'true_label': label_name,
                'true_label_id': true_label,
                'predicted_label': 'NC' if prediction == 0 else 'AD',
                'predicted_label_id': prediction,
                'confidence_nc': probability[0],
                'confidence_ad': probability[1],
                'confidence_max': max(probability),
                'correct': (prediction == true_label)
            })
            
        except Exception as e:
            tqdm.write(f"    錯誤：{os.path.basename(t1_path)}: {e}")

# 轉換為 DataFrame
results_df = pd.DataFrame(results)

print(f"\n✅ 預測完成！共處理 {len(results_df)} 個樣本")

# ====================================================================
# 4. 儲存預測結果
# ====================================================================
print("\n儲存預測結果...")
results_df.to_csv(os.path.join(OUTPUT_DIR, "batch_predictions.csv"), index=False)
print(f"✅ 預測結果已儲存至: {OUTPUT_DIR}batch_predictions.csv")

# ====================================================================
# 5. 生成統計報表
# ====================================================================
print("\n生成統計報表...")

report_lines = []
report_lines.append("="*80)
report_lines.append("批次預測統計報表")
report_lines.append("="*80)

# 5.1 整體準確率
overall_acc = accuracy_score(results_df['true_label_id'], results_df['predicted_label_id'])
report_lines.append(f"\n【整體效能】")
report_lines.append(f"  總樣本數: {len(results_df)}")
report_lines.append(f"  整體準確率: {overall_acc:.2%}")
report_lines.append(f"  預測正確: {results_df['correct'].sum()}/{len(results_df)}")
report_lines.append(f"  預測錯誤: {(~results_df['correct']).sum()}/{len(results_df)}")

# 5.2 各類別效能
report_lines.append(f"\n【各類別效能】")
for label_name in ['NC', 'AD']:
    subset = results_df[results_df['true_label'] == label_name]
    acc = subset['correct'].mean()
    n_correct = subset['correct'].sum()
    n_total = len(subset)
    report_lines.append(f"\n  {label_name}:")
    report_lines.append(f"    樣本數: {n_total}")
    report_lines.append(f"    準確率: {acc:.2%} ({n_correct}/{n_total})")
    
    # 平均信心度
    if label_name == 'NC':
        avg_conf = subset['confidence_nc'].mean()
    else:
        avg_conf = subset['confidence_ad'].mean()
    report_lines.append(f"    平均信心度: {avg_conf:.2%}")

# 5.3 混淆矩陣
report_lines.append(f"\n【混淆矩陣】")
cm = confusion_matrix(results_df['true_label_id'], results_df['predicted_label_id'])
report_lines.append(f"\n           預測 NC    預測 AD")
report_lines.append(f"  真實 NC    {cm[0,0]:3d}        {cm[0,1]:3d}")
report_lines.append(f"  真實 AD    {cm[1,0]:3d}        {cm[1,1]:3d}")

# 計算指標
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率 (AD)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特異度 (NC)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # 精確度 (AD)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

report_lines.append(f"\n【詳細指標】")
report_lines.append(f"  靈敏度 (Sensitivity/Recall): {sensitivity:.2%}")
report_lines.append(f"    → 正確識別 AD 的能力")
report_lines.append(f"  特異度 (Specificity):        {specificity:.2%}")
report_lines.append(f"    → 正確識別 NC 的能力")
report_lines.append(f"  精確度 (Precision):          {precision:.2%}")
report_lines.append(f"    → 預測為 AD 時的正確率")
report_lines.append(f"  F1 分數:                     {f1:.2%}")

# 5.4 信心度分析
report_lines.append(f"\n【信心度分析】")

# 整體信心度
avg_confidence = results_df['confidence_max'].mean()
report_lines.append(f"\n  整體平均信心度: {avg_confidence:.2%}")

# 正確 vs 錯誤預測的信心度
correct_conf = results_df[results_df['correct']]['confidence_max'].mean()
incorrect_conf = results_df[~results_df['correct']]['confidence_max'].mean()
report_lines.append(f"  正確預測的平均信心度: {correct_conf:.2%}")
report_lines.append(f"  錯誤預測的平均信心度: {incorrect_conf:.2%}")

# 信心度分布
report_lines.append(f"\n  信心度分布:")
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    n_high_conf = (results_df['confidence_max'] >= threshold).sum()
    pct = n_high_conf / len(results_df) * 100
    report_lines.append(f"    ≥ {threshold:.0%}: {n_high_conf:3d} 個樣本 ({pct:.1f}%)")

# 5.5 錯誤案例分析
report_lines.append(f"\n【錯誤案例分析】")
errors = results_df[~results_df['correct']].sort_values('confidence_max', ascending=False)

# 初始化變數
fp_cases = pd.DataFrame()
fn_cases = pd.DataFrame()

if len(errors) > 0:
    report_lines.append(f"\n  共 {len(errors)} 個錯誤預測:")
    
    # False Positives (預測為 AD 但實際是 NC)
    fp_cases = errors[errors['predicted_label'] == 'AD']
    report_lines.append(f"\n  假陽性 (False Positive): {len(fp_cases)} 個")
    report_lines.append(f"    → 實際是 NC，但預測為 AD")
    if len(fp_cases) > 0:
        report_lines.append(f"    前 5 個案例:")
        for idx, row in fp_cases.head(5).iterrows():
            report_lines.append(f"      - {row['subject_id']}: 信心度 {row['confidence_ad']:.2%}")
    
    # False Negatives (預測為 NC 但實際是 AD)
    fn_cases = errors[errors['predicted_label'] == 'NC']
    report_lines.append(f"\n  假陰性 (False Negative): {len(fn_cases)} 個")
    report_lines.append(f"    → 實際是 AD，但預測為 NC")
    if len(fn_cases) > 0:
        report_lines.append(f"    前 5 個案例:")
        for idx, row in fn_cases.head(5).iterrows():
            report_lines.append(f"      - {row['subject_id']}: 信心度 {row['confidence_nc']:.2%}")
else:
    report_lines.append(f"\n  🎉 完美！沒有任何錯誤預測！")

# 5.6 結論與建議
report_lines.append(f"\n【結論與建議】")
report_lines.append(f"\n  模型表現評估:")

if overall_acc >= 0.9:
    report_lines.append(f"    ✅ 優秀 (≥90%): 模型表現非常好")
elif overall_acc >= 0.8:
    report_lines.append(f"    ✅ 良好 (80-90%): 模型表現良好")
elif overall_acc >= 0.7:
    report_lines.append(f"    ⚠️  尚可 (70-80%): 模型表現尚可，有改進空間")
else:
    report_lines.append(f"    ❌ 需改進 (<70%): 模型需要進一步優化")

if correct_conf - incorrect_conf > 0.1:
    report_lines.append(f"    ✅ 信心度區分良好: 正確預測的信心度明顯高於錯誤預測")
else:
    report_lines.append(f"    ⚠️  信心度區分不明顯: 模型可能過度自信或不夠自信")

report_lines.append(f"\n  建議:")
if len(fp_cases) > len(fn_cases):
    report_lines.append(f"    - 假陽性較多，考慮提高 AD 預測的閾值")
elif len(fn_cases) > len(fp_cases):
    report_lines.append(f"    - 假陰性較多，考慮降低 AD 預測的閾值")

if avg_confidence < 0.7:
    report_lines.append(f"    - 整體信心度偏低，考慮增加訓練資料或調整模型")

report_lines.append(f"\n" + "="*80)

# 儲存報表
report_text = "\n".join(report_lines)
with open(os.path.join(OUTPUT_DIR, "prediction_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
print(f"\n✅ 報表已儲存至: {OUTPUT_DIR}prediction_report.txt")

# ====================================================================
# 6. 信心度分析 CSV
# ====================================================================
print("\n生成信心度分析...")

confidence_analysis = []
for threshold in np.arange(0.5, 1.0, 0.05):
    high_conf = results_df[results_df['confidence_max'] >= threshold]
    if len(high_conf) > 0:
        acc = high_conf['correct'].mean()
        confidence_analysis.append({
            'threshold': threshold,
            'n_samples': len(high_conf),
            'accuracy': acc,
            'percentage': len(high_conf) / len(results_df) * 100
        })

confidence_df = pd.DataFrame(confidence_analysis)
confidence_df.to_csv(os.path.join(OUTPUT_DIR, "confidence_analysis.csv"), index=False)
print(f"✅ 信心度分析已儲存至: {OUTPUT_DIR}confidence_analysis.csv")

# ====================================================================
# 7. 視覺化 (使用英文避免字體問題)
# ====================================================================
print("\n生成視覺化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7.1 混淆矩陣
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['NC', 'AD'], yticklabels=['NC', 'AD'])
axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('True Label', fontsize=10)
axes[0, 0].set_xlabel('Predicted Label', fontsize=10)

# 7.2 信心度分布
axes[0, 1].hist([results_df[results_df['correct']]['confidence_max'],
                 results_df[~results_df['correct']]['confidence_max']],
                bins=20, label=['Correct', 'Incorrect'], alpha=0.7, color=['green', 'red'])
axes[0, 1].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Confidence', fontsize=10)
axes[0, 1].set_ylabel('Number of Samples', fontsize=10)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 7.3 各類別信心度
nc_data = results_df[results_df['true_label'] == 'NC']['confidence_nc']
ad_data = results_df[results_df['true_label'] == 'AD']['confidence_ad']
bp = axes[1, 0].boxplot([nc_data, ad_data], labels=['NC', 'AD'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)
axes[1, 0].set_title('Prediction Confidence by Class', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Confidence', fontsize=10)
axes[1, 0].set_xlabel('True Label', fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 7.4 信心度 vs 準確率
if len(confidence_df) > 0:
    axes[1, 1].plot(confidence_df['threshold'], confidence_df['accuracy'], 
                    marker='o', linewidth=2, markersize=6, color='steelblue')
    axes[1, 1].set_title('Confidence Threshold vs Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Confidence Threshold', fontsize=10)
    axes[1, 1].set_ylabel('Accuracy', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prediction_analysis.png"), dpi=300, bbox_inches='tight')
print(f"✅ 視覺化已儲存至: {OUTPUT_DIR}prediction_analysis.png")

print("\n" + "="*80)
print("批次預測完成！")
print("="*80)
print(f"\n所有結果已儲存至: {OUTPUT_DIR}")
print(f"  - batch_predictions.csv: 詳細預測結果")
print(f"  - prediction_report.txt: 統計報表")
print(f"  - confidence_analysis.csv: 信心度分析")
print(f"  - prediction_analysis.png: 視覺化圖表")
