import os
import nibabel as nib
import numpy as np
from nilearn import plotting

# --- 1. 設定檔案路徑 ---
nifti_file_path = "data/dmn/IC7_DMN.nii.gz"
output_png_path = "figures/dmn/dmn_ic7_preview_autothresh.png"

# --- 2. 檢查檔案 ---
if not os.path.exists(nifti_file_path):
    print(f"錯誤：找不到檔案 '{nifti_file_path}'")
else:
    # --- 3. 載入並檢查數據 ---
    print(f"正在載入檔案: {nifti_file_path}")
    nii_image = nib.load(nifti_file_path)
    data = nii_image.get_fdata()

    # 找出所有正值的數據
    positive_data = data[data > 0]
    
    print("\n--- 數據檢查報告 ---")
    if positive_data.size > 0:
        print(f"最大值: {data.max():.2f}")
        print(f"最小值: {data.min():.2f}")
        print(f"平均值 (僅正數): {positive_data.mean():.2f}")

        # --- 4. 動態計算門檻值 ---
        # 我們選擇所有正值中，強度排名前 10% 的點 (90th percentile)
        # 这是一个比較寬鬆的標準，能確保看到大部分腦區
        # 您可以嘗試調整 90 這個數字 (例如改成 95 會更嚴格)
        adaptive_threshold = np.percentile(positive_data, 00)
        print(f"動態計算出的門檻值 (Top 10%): {adaptive_threshold:.2f}")
    else:
        print("警告：這個檔案中沒有找到任何正值的數據！")
        adaptive_threshold = None # 如果沒有正值，就不設門檻

    # --- 5. 使用新的門檻值來繪圖 ---
    print("\n正在使用動態門檻值重新繪圖...")
    output_dir = os.path.dirname(output_png_path)
    os.makedirs(output_dir, exist_ok=True)

    display = plotting.plot_stat_map(
        nifti_file_path,
        display_mode="mosaic",
        title=f'DMN Preview (Threshold > {adaptive_threshold:.2f})',
        threshold=adaptive_threshold, # <--- 使用我們動態計算出的門檻！
        colorbar=True
    )
    
    display.savefig(output_png_path)
    
    print("-" * 30)
    print(f"🎉 成功！新圖片已儲存至: {output_png_path}")
    print("-" * 30)
    