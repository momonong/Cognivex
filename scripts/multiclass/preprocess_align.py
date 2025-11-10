import os
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
import warnings
import time

# 🚨 檢查相依性 (nilearn)
try:
    from nilearn import image as nimg
    from nilearn import datasets
except ImportError:
    print("🚨 致命錯誤：缺少 'nilearn' 套件。")
    print("請執行: pip install nilearn")
    exit()

# 隱藏 nilearn 的 UserWarning 和 FutureWarning
warnings.filterwarnings("ignore", category=UserWarning, module='nilearn')
warnings.filterwarnings("ignore", category=FutureWarning)

# ====================================================================
# 【1. 設定】
# ====================================================================
SOURCE_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal/" 
TARGET_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/" 

# 取得所有類別
LABELS = ["AD", "MCI", "NC"]

print(f"--- V27 預處理：空間標準化 (對齊至 MNI 模板) ---")
print(f"來源 (原始資料): {SOURCE_ROOT}")
print(f"目標 (對齊資料): {TARGET_ROOT}")

# ====================================================================
# 【2. 載入 MNI 模板】
# ====================================================================
print("\n正在載入 MNI152 (2mm) 標準模板...")
# 我們使用 2mm 解析度的模板，這是一個常用且高效的標準
# 您的所有影像都將被重採樣到這個模板的 Shape 和 Affine
try:
    # `resolution=2` 下載 2mm 模板
    mni_template = datasets.load_mni152_template(resolution=2) 
    print(f"✅ MNI 模板載入完畢。目標 Shape: {mni_template.shape}, Affine:\n{mni_template.affine}")
except Exception as e:
    print(f"🚨 錯誤：無法下載 MNI 模板。請檢查您的網路連線。")
    print(f"錯誤訊息: {e}")
    exit()

# ====================================================================
# 【3. 建立目標資料夾】
# ====================================================================
for label in LABELS:
    os.makedirs(os.path.join(TARGET_ROOT, label), exist_ok=True)

# ====================================================================
# 【4. 掃描並處理所有病患】
# ====================================================================
# 我們假設資料夾已清理乾淨 (n=136)，所有 _DWI 都是 3D FA
# 我們以 T1 檔案為基礎來掃描
all_t1_files = glob.glob(os.path.join(SOURCE_ROOT, "*", "*_T1.nii.gz"))

if not all_t1_files:
    print(f"🚨 錯誤：在 {SOURCE_ROOT} 下找不到任何 *_T1.nii.gz 檔案。")
    exit()

print(f"\n--- 找到 {len(all_t1_files)} 位病患。開始進行空間標準化... ---")
print("⚠️ 這將會花費 1-2 小時。如果中斷，重新執行即可繼續。")

# 使用 tqdm 遍歷所有 T1 檔案
for t1_path in tqdm(all_t1_files, unit="subject"):
    
    start_time = time.time()
    
    try:
        # --- a. 取得路徑和標籤 ---
        base_name = t1_path.replace("_T1.nii.gz", "")
        subject_id = os.path.basename(base_name)
        
        # 取得標籤 (AD, MCI, or NC)
        label = os.path.basename(os.path.dirname(t1_path))
        if label not in LABELS:
            tqdm.write(f"警告：找到未知標籤 '{label}' (在 {t1_path})，跳過此病患。")
            continue
            
        # 找出對應的 T2 和 3D FA (DWI) 路徑
        t2_path = base_name + "_T2_FLAIR.nii.gz"
        fa_path = base_name + "_DWI.nii.gz" # 我們已確認這是 3D FA

        # --- b. 檢查檔案是否存在 ---
        if not os.path.exists(t2_path) or not os.path.exists(fa_path):
            tqdm.write(f"警告：病患 {subject_id} 缺少 T2 或 DWI 檔案，跳過。")
            continue

        # --- c. 定義輸出路徑 (V27 核心) ---
        target_label_dir = os.path.join(TARGET_ROOT, label)
        out_t1_path = os.path.join(target_label_dir, f"{subject_id}_T1.nii.gz")
        out_t2_path = os.path.join(target_label_dir, f"{subject_id}_T2_FLAIR.nii.gz")
        out_fa_path = os.path.join(target_label_dir, f"{subject_id}_DWI.nii.gz")

        # --- d. (重要) 檢查是否已處理過 ---
        if os.path.exists(out_t1_path) and os.path.exists(out_t2_path) and os.path.exists(out_fa_path):
            # tqdm.write(f"病患 {subject_id} 已處理過，跳過。") # 保持進度條乾淨
            continue

        # --- e. 載入影像 ---
        # 使用 float32 載入以避免 V25 的 dtype 錯誤
        t1_img = nimg.load_img(t1_path, dtype=np.float32)
        t2_img = nimg.load_img(t2_path, dtype=np.float32)
        fa_img = nimg.load_img(fa_path, dtype=np.float32)
        
        # --- f. 核心：重採樣到 MNI 模板 ---
        # nilearn.image.resample_to_img 
        # 會執行線性配準 (Affine Registration)
        # 將 'source_img' 扭曲/縮放/平移，使其對齊 'target_img' (MNI 模板)
        
        t1_aligned = nimg.resample_to_img(t1_img, mni_template)
        t2_aligned = nimg.resample_to_img(t2_img, mni_template)
        fa_aligned = nimg.resample_to_img(fa_img, mni_template)

        # --- g. 儲存已對齊的影像 ---
        t1_aligned.to_filename(out_t1_path)
        t2_aligned.to_filename(out_t2_path)
        fa_aligned.to_filename(out_fa_path)
        
        duration = time.time() - start_time
        tqdm.write(f"✅ 病患 {subject_id} (類別: {label}) 處理完畢 (耗時 {duration:.2f}s)")

    except Exception as e:
        tqdm.write(f"\n🔥🔥🔥 錯誤：處理病患 {subject_id} 時發生致命錯誤，跳過。")
        tqdm.write(f"       錯誤訊息: {e}")
        # 如果檔案已損壞，刪除可能已生成的壞檔案
        if os.path.exists(out_t1_path): os.remove(out_t1_path)
        if os.path.exists(out_t2_path): os.remove(out_t2_path)
        if os.path.exists(out_fa_path): os.remove(out_fa_path)

print("\n--- ✅ V27 預處理：空間標準化全部完成！ ---")
print(f"所有 136 筆對齊後的影像現已儲存於：")
print(f"{TARGET_ROOT}")
print("\n下一步，我們將使用這個新資料夾來訓練 V28 (Atlas-Patch-based) 模型。")