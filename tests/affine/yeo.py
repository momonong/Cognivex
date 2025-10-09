import os
from nilearn import plotting
from nilearn.image import math_img

# --- 1. 設定檔案和參數 ---
full_atlas_path = "data/dmn/Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask_colin27.nii"

# Yeo 17 網路圖譜中，DMN 的三個子網路標籤
DMN_SUBNETWORK_LABELS = [14, 15, 16]

# 設定輸出路徑
output_dir = "figures/dmn"
os.makedirs(output_dir, exist_ok=True)
output_png_path = os.path.join(output_dir, "Yeo17_DMN_Combined.png")

# --- 2. 檢查檔案 ---
if not os.path.exists(full_atlas_path):
    print(f"錯誤：找不到圖譜檔案！請確認路徑： {full_atlas_path}")
else:
    # --- 3. 從完整圖譜中提取【所有】DMN 子網路 ---
    # 建立計算公式，將三個子網路合併
    formula = "+".join([f"(img == {label})" for label in DMN_SUBNETWORK_LABELS])
    print(f"正在使用公式提取 DMN: {formula}")
    
    # 執行公式，建立一個包含所有 DMN 腦區的遮罩 (mask)
    dmn_combined_mask = math_img(formula, img=full_atlas_path)

    # --- 4. 繪製合併後的完整 DMN 疊圖 ---
    print("正在繪製合併後的 DMN 預覽圖...")
    display = plotting.plot_roi(
        roi_img=dmn_combined_mask,
        title='Yeo 2011 - Full DMN (Sub-networks 14+15+16)',
        display_mode='mosaic',
        draw_cross=False,
    )
    display.savefig(output_png_path, dpi=300)
    
    print("-" * 30)
    print(f"🎉 成功！合併後的 DMN 疊圖已儲存至: {output_png_path}")
    print("-" * 30)
    
    plotting.plot_glass_brain(dmn_combined_mask, title='Full DMN (Glass Brain)')