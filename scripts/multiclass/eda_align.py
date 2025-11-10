import os
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
import warnings

# ====================================================================
# 【1. 設定】
# ====================================================================
DATA_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI" 
print(f"--- V26-EDA：極速 Header 檢查 (Shape & Affine) ---")
print(f"正在掃描資料夾: {DATA_ROOT}")

# ====================================================================
# 【2. 掃描所有 T1 影像】
# ====================================================================
t1_files = glob.glob(os.path.join(DATA_ROOT, "*", "*_T1.nii.gz"))

if not t1_files:
    print(f"🚨 錯誤：在 {DATA_ROOT} 下找不到任何 *_T1.nii.gz 檔案。")
    exit()

print(f"總共找到 {len(t1_files)} 筆 T1 影像 (應為 136)。")
print("--- 開始檢查所有 T1 的 Shape 和 Affine 是否一致 ---")

# ====================================================================
# 【3. 關鍵：循環比較 Header】
# ====================================================================
try:
    # 載入第一筆資料作為「參考標準」
    ref_img = nib.load(t1_files[0])
    ref_shape = ref_img.shape
    ref_affine = ref_img.affine
    
    print(f"參考標準 (來自 {os.path.basename(t1_files[0])}):")
    print(f"  - Shape: {ref_shape}")
    print(f"  - Affine:\n{ref_affine}\n")

    is_aligned = True
    
    # 從第二筆開始比較 (tqdm 從 1 開始)
    for f in tqdm(t1_files[1:], desc="正在比較 Headers"):
        img = nib.load(f)
        current_shape = img.shape
        current_affine = img.affine

        # 1. 比較 Shape
        if current_shape != ref_shape:
            print(f"\n🚨 警告：Shape 不一致！")
            print(f"  - 檔案: {os.path.basename(f)}")
            print(f"  - Shape: {current_shape} (應為: {ref_shape})")
            is_aligned = False
            break # 找到一個不一致就停止

        # 2. 比較 Affine 矩陣
        if not np.array_equal(current_affine, ref_affine):
            print(f"\n🚨 警告：Affine 座標不一致！")
            print(f"  - 檔案: {os.path.basename(f)}")
            print(f"  - Affine:\n{current_affine}\n  (應為:\n{ref_affine})")
            is_aligned = False
            break # 找到一個不一致就停止

    # ====================================================================
    # 【4. 總結報告】
    # ====================================================================
    if is_aligned:
        print("\n--- ✅ 檢查完畢：完美對齊 ---")
        print("恭喜！所有 136 筆 T1 影像的 Shape 和 Affine 完全一致。")
        print("我們可以**直接**進行 V27 (Atlas-Patch-based) 訓練。")
    else:
        print("\n--- 🚨 檢查完畢：資料未對齊 ---")
        print("警告：您的 T1 影像彼此之間未對齊 (座標或維度不同)。")
        print("我們**必須**在訓練前，先進行一個「空間標準化」預處理步驟。")

except Exception as e:
    print(f"\n🚨 致命錯誤：在讀取 Header 時失敗。")
    print(f"錯誤訊息: {e}")