import os
import torch
import nibabel as nib
import numpy as np
from nilearn import plotting, datasets
from nilearn.image import math_img, resample_to_img

# (您的轉換函式 activation_to_nifti_nilearn 保持不變)
def activation_to_nifti_nilearn(
    activation_tensor: torch.Tensor,
    original_small_affine: np.ndarray,
    reference_nii_path: str,
    output_path: str,
):
    temp_nii = nib.Nifti1Image(activation_tensor.numpy(), original_small_affine)
    ref_img = nib.load(reference_nii_path)
    resampled_nii = resample_to_img(
        source_img=temp_nii,
        target_img=ref_img,
        interpolation='continuous'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(resampled_nii, output_path)
    print(f"-> 升維後的影像已儲存至: {output_path}")

# ===============================================================
#  主驗證流程 (修正版)
# ===============================================================
if __name__ == '__main__':
    # --- 1. 設定 ---
    print("--- 步驟 1: 設定參數 ---")
    YOUR_MODEL_ACTIVATION_SHAPE = (19, 24, 19) # 根據您上次的輸出設定
    full_atlas_path = "data/dmn/Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask_colin27.nii"
    output_dir = "figures/validation"
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. 準備「標準答案」 ---
    print("\n--- 步驟 2: 準備高解析度的『標準答案』DMN 圖 ---")
    mni_template = datasets.load_mni152_template(resolution=2)
    reference_path = os.path.join(output_dir, "mni152_template.nii.gz")
    nib.save(mni_template, reference_path)
    
    # 從 Yeo 圖譜提取 DMN
    formula = "+".join([f"(img == {label})" for label in [14, 15, 16]])
    raw_dmn_mask = math_img(formula, img=full_atlas_path)
    
    # ======================================================================
    #  【關鍵修正！】將提取出的 DMN 重新取樣到標準 MNI 模板上，
    #  以確保它有一個乾淨、對角化的 affine 座標系統。
    # ======================================================================
    print("-> 正在將 DMN 標準答案重新取樣至 MNI 模板以統一座標系...")
    ground_truth_dmn = resample_to_img(source_img=raw_dmn_mask, target_img=mni_template, interpolation='nearest')
    
    ground_truth_path = os.path.join(output_dir, "ground_truth_dmn_resampled.nii.gz")
    nib.save(ground_truth_dmn, ground_truth_path)
    print(f"-> 標準答案 DMN (已校正座標) 已儲存至: {ground_truth_path}")

    # --- 3. 模擬模型輸出 (降維) ---
    print("\n--- 步驟 3: 執行『降維』，模擬您的模型輸出 ---")
    small_affine = mni_template.affine.copy()
    for i in range(3):
        small_affine[i, i] = mni_template.affine[i, i] * (mni_template.shape[i] / YOUR_MODEL_ACTIVATION_SHAPE[i])
    
    downsampled_dmn = resample_to_img(
        source_img=ground_truth_dmn,
        target_img=nib.Nifti1Image(np.zeros(YOUR_MODEL_ACTIVATION_SHAPE), small_affine),
        interpolation='nearest'
    )
    simulated_activation_tensor = torch.from_numpy(downsampled_dmn.get_fdata().astype(np.float32))
    print(f"-> 已產生模擬的激活 Tensor，尺寸: {simulated_activation_tensor.shape}")

    # --- 4. 執行您的函式 (升維) ---
    print("\n--- 步驟 4: 執行您的轉換函式進行『升維』---")
    round_trip_path = os.path.join(output_dir, "round_trip_dmn.nii.gz")
    activation_to_nifti_nilearn(
        activation_tensor=simulated_activation_tensor,
        original_small_affine=small_affine,
        reference_nii_path=reference_path,
        output_path=round_trip_path,
    )

    # ... (前面步驟 1 到 4 的程式碼完全不變) ...

    # ===============================================================
    #  步驟 5: 視覺化比對 (最終美化版)
    # ===============================================================
    print("\n--- 步驟 5: 視覺化比對『標準答案』與『往返後』的結果 ---")

    # 1. 我們先用 plot_roi 畫出【標準答案】的乾淨輪廓
    #    設置 colorbar=False 和 draw_cross=False 讓畫面更簡潔
    fig = plotting.plot_roi(
        ground_truth_path,
        bg_img=reference_path,
        display_mode='ortho',
        cut_coords=(-2, -53, 26),
        title="Validation: Ground Truth (Outline) vs. Round Trip (Color Fill)",
        draw_cross=False
    )

    # 2. 接著，使用 add_overlay 將【往返後的結果】作為【半透明的彩色填充】疊加上去
    #    alpha=0.75 設置了 75% 的透明度
    #    cmap='hot' 使用了醒目的熱力圖顏色
    fig.add_overlay(
        round_trip_path,
        cmap='hot',
        alpha=0.75
    )

    output_validation_png = os.path.join(output_dir, "AFFINE_VALIDATION_BEAUTIFIED.png")
    fig.savefig(output_validation_png, dpi=300)

    print("\n" + "="*50)
    print("✅ 最終驗證完成！")
    print(f"請打開美化後的圖片: {output_validation_png}")
    print("檢查圖中的【彩色填充區域】是否完美地填滿了【黑色輪廓線】。")
    print("這次的結果將會非常清晰且有說服力！")
    print("="*50)
