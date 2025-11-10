"""
多模態資料 EDA - 診斷資料問題
"""

import numpy as np
import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
from nilearn import datasets, image as nimg
import pandas as pd

DATA_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/"

print("="*80)
print("多模態資料 EDA")
print("="*80)

# ====================================================================
# 1. 檢查資料夾結構和檔案數量
# ====================================================================
print("\n【1. 資料夾結構】")
print("-"*80)

label_map = {"NC": 0, "MCI": 1, "AD": 2}
all_subjects = []

for label_name, label_id in label_map.items():
    class_path = os.path.join(DATA_ROOT, label_name)
    
    if not os.path.isdir(class_path):
        print(f"❌ 找不到資料夾: {class_path}")
        continue
    
    t1_files = glob.glob(os.path.join(class_path, "*_T1.nii.gz"))
    t2_files = glob.glob(os.path.join(class_path, "*_T2_FLAIR.nii.gz"))
    dwi_files = glob.glob(os.path.join(class_path, "*_DWI.nii.gz"))
    
    print(f"\n{label_name} 類別:")
    print(f"  T1 檔案: {len(t1_files)}")
    print(f"  T2 檔案: {len(t2_files)}")
    print(f"  DWI 檔案: {len(dwi_files)}")
    
    # 檢查配對
    complete_subjects = 0
    for t1_path in t1_files:
        base_name = t1_path.replace("_T1.nii.gz", "")
        subject_id = os.path.basename(base_name)
        t2_path = base_name + "_T2_FLAIR.nii.gz"
        dwi_path = base_name + "_DWI.nii.gz"
        
        if os.path.exists(t2_path) and os.path.exists(dwi_path):
            complete_subjects += 1
            all_subjects.append({
                "subject_id": subject_id,
                "label": label_name,
                "label_id": label_id,
                "t1": t1_path,
                "t2": t2_path,
                "dwi": dwi_path
            })
    
    print(f"  完整配對: {complete_subjects}/{len(t1_files)}")

print(f"\n總共找到 {len(all_subjects)} 個完整的受試者")

# ====================================================================
# 2. 檢查每個模態的資料品質
# ====================================================================
print("\n【2. 資料品質檢查】")
print("-"*80)

# 隨機抽樣 5 個受試者進行詳細檢查
sample_size = min(5, len(all_subjects))
sample_subjects = np.random.choice(len(all_subjects), sample_size, replace=False)

stats_data = []

for idx in sample_subjects:
    subject = all_subjects[idx]
    print(f"\n檢查受試者: {subject['subject_id']} ({subject['label']})")
    
    for modality, path_key in [("T1", "t1"), ("T2", "t2"), ("DWI", "dwi")]:
        try:
            img = nib.load(subject[path_key])
            data = img.get_fdata()
            
            # 計算統計資訊
            stats = {
                "subject_id": subject['subject_id'],
                "label": subject['label'],
                "modality": modality,
                "shape": data.shape,
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "median": float(np.median(data)),
                "has_nan": bool(np.isnan(data).any()),
                "has_inf": bool(np.isinf(data).any()),
                "num_zeros": int(np.sum(data == 0)),
                "zero_percentage": float(np.sum(data == 0) / data.size * 100)
            }
            
            stats_data.append(stats)
            
            print(f"  {modality}:")
            print(f"    形狀: {stats['shape']}")
            print(f"    數值範圍: [{stats['min']:.2f}, {stats['max']:.2f}]")
            print(f"    平均值: {stats['mean']:.2f} ± {stats['std']:.2f}")
            print(f"    中位數: {stats['median']:.2f}")
            print(f"    零值比例: {stats['zero_percentage']:.2f}%")
            
            if stats['has_nan']:
                print(f"    ⚠️ 警告: 包含 NaN 值！")
            if stats['has_inf']:
                print(f"    ⚠️ 警告: 包含 Inf 值！")
            
        except Exception as e:
            print(f"  ❌ 錯誤: 無法載入 {modality}: {e}")

# 建立統計 DataFrame
stats_df = pd.DataFrame(stats_data)

# ====================================================================
# 3. 統計摘要
# ====================================================================
print("\n【3. 統計摘要】")
print("-"*80)

if len(stats_df) > 0:
    print("\n各模態的數值範圍:")
    summary = stats_df.groupby('modality')[['min', 'max', 'mean', 'std']].agg(['min', 'max', 'mean'])
    print(summary)
    
    print("\n各類別的樣本數:")
    class_counts = pd.DataFrame(all_subjects).groupby('label').size()
    print(class_counts)
    
    # 儲存統計資料
    output_dir = "output/multiclass/eda/"
    os.makedirs(output_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(output_dir, "data_statistics.csv"), index=False)
    print(f"\n✅ 統計資料已儲存至: {output_dir}data_statistics.csv")

# ====================================================================
# 4. 檢查 AAL Atlas 對齊
# ====================================================================
print("\n【4. AAL Atlas 對齊檢查】")
print("-"*80)

try:
    # 載入 AAL atlas
    aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
    aal_img = nimg.load_img(aal_atlas.maps)
    aal_data = aal_img.get_fdata()
    
    print(f"AAL Atlas:")
    print(f"  形狀: {aal_data.shape}")
    print(f"  數值範圍: [{aal_data.min():.0f}, {aal_data.max():.0f}]")
    print(f"  唯一 ROI 數量: {len(np.unique(aal_data)) - 1}")  # -1 排除背景
    
    # 檢查與資料的對齊
    if len(all_subjects) > 0:
        test_subject = all_subjects[0]
        test_img = nib.load(test_subject['t1'])
        test_data = test_img.get_fdata()
        
        print(f"\n測試影像 ({test_subject['subject_id']}):")
        print(f"  形狀: {test_data.shape}")
        
        if test_data.shape == aal_data.shape:
            print("  ✅ 形狀與 AAL 匹配")
        else:
            print(f"  ⚠️ 形狀不匹配！需要重採樣")
            print(f"  嘗試重採樣 AAL...")
            aal_img_resampled = nimg.resample_to_img(aal_img, test_img, interpolation='nearest')
            aal_data_resampled = aal_img_resampled.get_fdata()
            print(f"  重採樣後 AAL 形狀: {aal_data_resampled.shape}")
            
            if aal_data_resampled.shape == test_data.shape:
                print("  ✅ 重採樣成功")
            else:
                print("  ❌ 重採樣後仍不匹配")

except Exception as e:
    print(f"❌ 錯誤: 無法載入 AAL atlas: {e}")

# ====================================================================
# 5. 診斷結論和建議
# ====================================================================
print("\n【5. 診斷結論】")
print("="*80)

issues = []
recommendations = []

# 檢查數值範圍
if len(stats_df) > 0:
    for modality in ['T1', 'T2', 'DWI']:
        modality_stats = stats_df[stats_df['modality'] == modality]
        if len(modality_stats) > 0:
            max_val = modality_stats['max'].max()
            mean_val = modality_stats['mean'].mean()
            
            if max_val < 10:
                issues.append(f"{modality} 的最大值太小 ({max_val:.2f})，可能已被過度標準化")
                recommendations.append(f"檢查 {modality} 的預處理流程")
            
            if mean_val < 0.1:
                issues.append(f"{modality} 的平均值太小 ({mean_val:.4f})，可能導致梯度消失")
                recommendations.append(f"考慮調整 {modality} 的標準化方法")

# 檢查類別不平衡
class_counts = pd.DataFrame(all_subjects).groupby('label').size()
if len(class_counts) > 0:
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 3:
        issues.append(f"類別嚴重不平衡 (比例: {imbalance_ratio:.1f}:1)")
        recommendations.append("使用類別權重或過採樣/欠採樣")

# 檢查樣本數量
total_samples = len(all_subjects)
if total_samples < 100:
    issues.append(f"樣本數量較少 ({total_samples} 個)")
    recommendations.append("考慮使用更簡單的模型或遷移學習")

# 輸出結論
if issues:
    print("\n⚠️ 發現的問題:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n💡 建議:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
else:
    print("\n✅ 資料看起來正常")

print("\n" + "="*80)
print("EDA 完成")
print("="*80)

# ====================================================================
# 6. 建議的下一步
# ====================================================================
print("\n【建議的下一步】")
print("-"*80)

print("""
根據 EDA 結果，建議採取以下行動：

1. 如果資料數值範圍正常 (T1: 0-1000, T2: 0-1000, DWI: 0-1):
   → 執行: python scripts/multiclass/train_v41_simple_3dcnn.py
   
2. 如果資料已被過度標準化 (數值接近 0):
   → 需要重新預處理資料，或修改標準化方法
   
3. 如果樣本數量太少 (< 100):
   → 考慮使用 ROI 特徵 + 簡單 MLP
   → 執行: python scripts/multiclass/train_v40_simplified.py
   
4. 如果類別嚴重不平衡:
   → 確保使用類別權重 (已在訓練腳本中實作)
   → 考慮使用 SMOTE 或其他平衡技術

請根據上述 EDA 結果選擇適合的訓練腳本。
""")
