import os
from nilearn import plotting

# --- 1. 設定檔案路徑 ---
# 您要呈現的 Yeo 17 網路圖譜檔案
nifti_file_path = "data/dmn/Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask_colin27.nii"

# 預計輸出的 PNG 圖片檔案
output_png_path = "figures/dmn/Yeo17_Atlas_preview.png"
output_mosaic_path = "figures/dmn/Yeo17_Atlas_mosaic_preview.png"

# --- 2. 檢查檔案是否存在 ---
if not os.path.exists(nifti_file_path):
    print(f"錯誤：找不到指定的 NIfTI 檔案！")
    print(f"請確認路徑是否正確: {nifti_file_path}")
else:
    print(f"成功找到圖譜檔案: {nifti_file_path}")
    
    # 建立輸出資料夾
    output_dir = os.path.dirname(output_png_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- 3. 繪圖並儲存 ---
    print("正在產生預覽圖...")
    
    # 使用 plot_roi 函式，這是專門用來繪製【腦區圖譜】的
    # 它會自動為每一個不同的數值（代表一個網路）分配一個獨特的顏色
    display = plotting.plot_roi(
        roi_img=nifti_file_path,
        display_mode='mosaic',
        title='Yeo 2011 17-Network Atlas',
        # 對於 ROI 圖譜，我們通常不需要設定 threshold
    )
    display.savefig(output_png_path)
    
    print("-" * 30)
    print(f"🎉 成功！標準三視圖已儲存至: {output_png_path}")
    
    # --- 4. 額外加碼：使用馬賽克模式看全貌 ---
    # 馬賽克模式對於看 Atlas 的全貌非常有用
    print("正在產生馬賽克全覽圖...")
    display_mosaic = plotting.plot_roi(
        roi_img=nifti_file_path,
        display_mode='z', # z 代表軸狀切面
        cut_coords=range(-40, 61, 5), # 從 z=-40 到 z=60，每 5mm 一個切片
        title='Yeo 2011 17-Network Atlas (Mosaic View)',
    )
    display_mosaic.savefig(output_mosaic_path, dpi=300) # 提高解析度讓圖更清晰
    
    print(f"🎉 成功！馬賽克全覽圖已儲存至: {output_mosaic_path}")
    print("-" * 30)
    