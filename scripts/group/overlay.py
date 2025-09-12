import os
import nibabel as nib
import numpy as np
from nilearn import plotting, image, datasets

# --- 1. 設定與參數 ---
# [來自程式碼 A] 輸入檔案路徑
GROUP_ACT_PATH = "output/group/group_mean_activation.nii.gz"
YEO_ATLAS_PATH = "data/yeo/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz"

# 輸出路徑
OUTPUT_DIR = "figures/group/"
OUTPUT_FILENAME = "group_activation_with_dmn_overlay.png"

# DMN 在 Yeo 7-Network 圖譜中的標籤 ID
DMN_LABEL_ID = 7

# 視覺化參數
ACTIVATION_THRESHOLD = 0.1  # 激活圖的顯示閾值
ROI_ALPHA = 0.6             # DMN 實心色塊的透明度
CONTOUR_COLOR = 'magenta'      # DMN 輪廓線顏色

def create_combined_visualization():
    """
    整合大腦激活圖與 DMN 腦區遮罩，生成一張包含熱圖、
    半透明實心 ROI 區域以及輪廓線的複合影像。
    """
    print("--- 開始執行整合視覺化腳本 ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    # --- 2. 載入所有需要的影像資料 ---
    print("正在載入影像檔案...")
    try:
        group_act_img = nib.load(GROUP_ACT_PATH)
        yeo_img = nib.load(YEO_ATLAS_PATH)
    except FileNotFoundError as e:
        print(f"❌ 錯誤：找不到必要的檔案！ {e}")
        return

    mni_template = datasets.load_mni152_template()

    # --- 3. 建立 DMN 的二元遮罩 (Binary Mask) ---
    print(f"正在從 Yeo Atlas 中提取 DMN (Label ID: {DMN_LABEL_ID})...")
    yeo_data = yeo_img.get_fdata()
    dmn_mask_data = (yeo_data == DMN_LABEL_ID)
    dmn_mask_img = image.new_img_like(yeo_img, dmn_mask_data.astype(np.int32))

    # --- [3.5] 【關鍵修正】確保遮罩影像是 3D ---
    # 檢查 DMN 遮罩的維度，如果它是 4D (通常是 X, Y, Z, 1 的形狀)，
    # 我們就使用 index_img 選取第一個 (也是唯一一個) 3D 體積。
    if dmn_mask_img.ndim == 4:
        print(f"偵測到 4D 遮罩 (Shape: {dmn_mask_img.shape})，正在將其轉換為 3D...")
        dmn_mask_img = image.index_img(dmn_mask_img, 0)
        print(f"轉換後 Shape: {dmn_mask_img.shape}")
    
    # --- 4. 空間重採樣以確保完美對齊 ---
    print("正在將所有圖層重採樣至 MNI 模板空間以確保對齊...")
    act_aligned = image.resample_to_img(
        source_img=group_act_img,
        target_img=mni_template,
        interpolation='continuous',
        copy_header=True,
        force_resample=True
    )
    dmn_mask_aligned = image.resample_to_img(
        source_img=dmn_mask_img, # 現在傳入的是保證 3D 的遮罩
        target_img=mni_template,
        interpolation='nearest',
        copy_header=True,
        force_resample=True
    )

    # --- 5. 繪製並疊加所有圖層 ---
    print("正在繪製基礎激活圖 (Activation Map)...")
    display = plotting.plot_stat_map(
        act_aligned,
        bg_img=mni_template,
        display_mode="mosaic",
        threshold=ACTIVATION_THRESHOLD,
        title="Group Activation with DMN Overlay",
        annotate=True,
        black_bg=False,
        symmetric_cbar=False
    )

    print("正在疊加 DMN 半透明實心色塊 (ROI)...")
    display.add_overlay(
        dmn_mask_aligned,
        cmap='Set1',
        threshold=0.1,
        alpha=ROI_ALPHA
    )

    print("正在疊加 DMN 輪廓線 (Contour)...")
    # 現在傳遞給 add_contours 的 dmn_mask_aligned 保證是 3D 的，錯誤將會解決
    display.add_contours(
        dmn_mask_aligned,
        colors=CONTOUR_COLOR,
        linewidths=1
    )

    # --- 6. 儲存最終結果 ---
    print(f"正在儲存最終結果圖至: {output_path}")
    display.savefig(output_path, dpi=300)
    print("✅ 整合視覺化完成！")


# --- 執行主程式 ---
if __name__ == "__main__":
    create_combined_visualization()