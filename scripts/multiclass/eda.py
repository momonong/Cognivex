import numpy as np
import nibabel as nib
import os
import glob
import re
import pandas as pd

# ====================================================================
# 【1. 設定與配置】
# ====================================================================
DATA_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal/" 
print(f"--- Cognivex 計畫 B (Multi-Modal) 資料品質控制 (QC) ---")

# ====================================================================
# 【2. 資料集掃描邏輯】(與訓練腳本 V2/V3 相同)
# ====================================================================
subjects = []
label_map = {"NC": 0, "MCI": 1, "AD": 2}

print(f"正在掃描多模態資料集: {DATA_ROOT}")
for label_name, label_id in label_map.items():
    class_path = os.path.join(DATA_ROOT, label_name)
    if not os.path.isdir(class_path): continue
    
    t1_files = glob.glob(os.path.join(class_path, "*_T1.nii.gz"))
    for t1_path in t1_files:
        base_name = t1_path.replace("_T1.nii.gz", "")
        t2_path = base_name + "_T2_FLAIR.nii.gz"
        dwi_path = base_name + "_DWI.nii.gz"
        bval_path = base_name + "_DWI.bval"
        bvec_path = base_name + "_DWI.bvec"
        
        if os.path.exists(t2_path) and os.path.exists(dwi_path) and os.path.exists(bval_path) and os.path.exists(bvec_path):
            subjects.append({
                "t1": t1_path, "t2": t2_path, "dwi": dwi_path,
                "bval": bval_path, "bvec": bvec_path, "label": label_id,
                "subject_id": os.path.basename(base_name)
            })

print(f"資料集掃描完成。總共找到 {len(subjects)} 位完整多模態資料的病患。")

# ====================================================================
# 【3. 核心 QC 迴圈】
# ====================================================================
print("\n--- 開始對每筆資料進行維度與梯度檢查 ---")

all_results = []
bad_subjects = []

for i, sub in enumerate(subjects):
    subject_id = sub['subject_id']
    print(f"\n-> 正在檢查病患 ({i+1}/{len(subjects)}): {subject_id}")
    
    try:
        # 1. 檢查 T1
        t1_img = nib.load(sub["t1"])
        t1_shape = t1_img.shape
        
        # 2. 檢查 T2
        t2_img = nib.load(sub["t2"])
        t2_shape = t2_img.shape

        # 3. 檢查 DWI (最關鍵)
        dwi_img = nib.load(sub["dwi"])
        dwi_shape = dwi_img.shape
        
        # 🚨 關鍵檢查 1：DWI 必須是 4D
        if len(dwi_shape) != 4:
            print(f"   ❌ 錯誤：DWI 影像不是 4D！Shape: {dwi_shape}")
            raise ValueError(f"DWI 影像不是 4D，Shape 為 {dwi_shape}")
            
        num_volumes_in_dwi = dwi_shape[3]

        # 4. 檢查 bvals 和 bvecs
        bvals = np.loadtxt(sub["bval"])
        bvecs = np.loadtxt(sub["bvec"])
        
        # 確保 bvals 是 1D
        if bvals.ndim == 0: bvals = bvals.reshape(1)
        if bvals.ndim > 1:
            print(f"   ⚠️ 警告：bvals 是 {bvals.ndim}D，將嘗試壓縮。")
            bvals = bvals.flatten()

        # 確保 bvecs 是 2D (N, 3)
        if bvecs.ndim == 1:
            if bvecs.shape[0] == 3: bvecs = bvecs.reshape(1, 3)
            else: raise ValueError(f"bvecs 是 1D 但 shape 不是 (3,)。")
        
        if bvecs.shape[0] == 3 and bvecs.shape[1] > 3:
            bvecs = bvecs.T # 轉置 (3, N) -> (N, 3)
            
        if bvecs.shape[1] != 3:
            raise ValueError(f"bvecs 的最終 shape 不是 (N, 3)，而是 {bvecs.shape}")

        num_gradients_bval = bvals.shape[0]
        num_gradients_bvec = bvecs.shape[0]

        # 🚨 關鍵檢查 2：DWI 體積數必須匹配 bvals/bvecs
        if num_volumes_in_dwi != num_gradients_bval:
            print(f"   ❌ 錯誤：DWI 體積數 ({num_volumes_in_dwi}) 與 bvals ({num_gradients_bval}) 不匹配。")
            raise ValueError(f"DWI 體積 ({num_volumes_in_dwi}) 與 bvals ({num_gradients_bval}) 不匹配。")
        
        if num_volumes_in_dwi != num_gradients_bvec:
            print(f"   ❌ 錯誤：DWI 體積數 ({num_volumes_in_dwi}) 與 bvecs ({num_gradients_bvec}) 不匹配。")
            raise ValueError(f"DWI 體積 ({num_volumes_in_dwi}) 與 bvecs ({num_gradients_bvec}) 不匹配。")

        # 如果所有檢查都通過
        print(f"   ✅ OK. T1={t1_shape}, T2={t2_shape}, DWI={dwi_shape}, BVAL/VEC=(N={num_gradients_bval})")
        all_results.append({
            'subject_id': subject_id,
            'T1_Shape': str(t1_shape),
            'T2_Shape': str(t2_shape),
            'DWI_Shape': str(dwi_shape),
            'Num_Gradients': num_gradients_bval,
            'Status': 'OK'
        })

    except Exception as e:
        print(f"   🔥🔥🔥 處理 {subject_id} 時發生致命錯誤: {e}")
        bad_subjects.append(subject_id)
        all_results.append({
            'subject_id': subject_id,
            'T1_Shape': 'N/A', 'T2_Shape': 'N/A', 'DWI_Shape': 'N/A', 
            'Num_Gradients': 'N/A', 'Status': f'Error: {e}'
        })

# ====================================================================
# 【4. 總結報告】
# ====================================================================
print("\n\n--- QC 分析總結 ---")

df_results = pd.DataFrame(all_results)
summary_csv_path = os.path.join(DATA_ROOT, "_QC_analysis_summary.csv")
df_results.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')

print(f"✅ 總結報告已儲存至: {summary_csv_path}")

if not bad_subjects:
    print("🎉🎉🎉 恭喜！所有 141 筆資料都通過了維度檢查！")
else:
    print(f"🚨 警告：總共有 {len(bad_subjects)} 筆資料未通過檢查：")
    for bad_id in bad_subjects:
        print(f"   - {bad_id}")
    print("請檢查 _QC_analysis_summary.csv 檔案以獲取詳細錯誤，")
    print("並考慮在 'MultiModalDataset' 的 __init__ 函數中將這些 'bad_subjects' 排除掉。")