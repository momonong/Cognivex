import torch
import nibabel as nib
import numpy as np
from nilearn import datasets
from nilearn.image import resample_to_img, index_img
from nilearn import plotting
import os

# ===============================================================
#  最終修正版函式 (這個函式是正確的，無需修改)
# ===============================================================
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
    print(f"使用 Nilearn 轉換的影像已儲存至: {output_path}")

# ===============================================================
#  測試腳本主體
# ===============================================================
if __name__ == '__main__':
    # --- 準備數據 ---
    print("--- 準備數據 ---")
    mni_template = datasets.load_mni152_template(resolution=2)
    smith_atlas = datasets.fetch_atlas_smith_2009()
    
    rsn10_maps_path = smith_atlas.rsn10
    dmn_map_orig_nii = index_img(rsn10_maps_path, 0)
    
    output_dir = "output/affine_test"
    figures_dir = "figures/affine_test"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    reference_path = os.path.join(output_dir, "mni152_template.nii.gz")
    nib.save(mni_template, reference_path)
    
    dmn_map_aligned = resample_to_img(source_img=dmn_map_orig_nii, target_img=mni_template, interpolation='continuous')
    original_dmn_path = os.path.join(output_dir, "dmn_original.nii.gz")
    nib.save(dmn_map_aligned, original_dmn_path)
    print(f"原始 DMN 圖譜已儲存: {original_dmn_path}")

    # --- 降維 ---
    print("\n--- 降維 (模擬模型輸出) ---")
    target_shape_small = (20, 24, 20)
    original_small_affine = mni_template.affine.copy()
    for i in range(3):
        original_small_affine[i, i] = mni_template.affine[i, i] * (mni_template.shape[i] / target_shape_small[i])

    dmn_map_downsampled = resample_to_img(
        source_img=dmn_map_aligned,
        target_img=nib.Nifti1Image(np.zeros(target_shape_small), original_small_affine),
        interpolation='continuous'
    )
    dmn_tensor_small = torch.from_numpy(dmn_map_downsampled.get_fdata().astype(np.float32))

    # --- 升維 ---
    print("\n--- 升維 (執行座標轉換函式) ---")
    final_upsampled_path = os.path.join(output_dir, "test_final_dmn_upsampled.nii.gz")
    activation_to_nifti_nilearn(
        activation_tensor=dmn_tensor_small,
        original_small_affine=original_small_affine,
        reference_nii_path=reference_path,
        output_path=final_upsampled_path,
    )

    # --- 繪製 2D 示意圖 ---
    print("\n--- 正在產生 2D 示意圖 ---")
    final_dmn_nii_for_calc = nib.load(final_upsampled_path)
    final_dmn_data = final_dmn_nii_for_calc.get_fdata()
    threshold_value = np.percentile(final_dmn_data[final_dmn_data > 0], 98)

    display_thresholded = plotting.plot_stat_map(
        final_upsampled_path, bg_img=reference_path,
        title="Thresholded DMN Map (Top 2%)", display_mode='ortho',
        colorbar=True, threshold=threshold_value
    )
    output_image_path_thresholded = os.path.join(figures_dir, "test_final_dmn_thresholded.png")
    display_thresholded.savefig(output_image_path_thresholded)
    print(f"閾值化 DMN 示意圖已儲存至: {output_image_path_thresholded}")

    # ===============================================================
    #  繪製 3D 表面渲染圖 (最終修正版)
    # ===============================================================
    print("\n--- 正在產生 3D 表面渲染圖 ---")
    from nilearn.surface import vol_to_surf
    from nilearn.plotting import plot_surf_stat_map

    fsaverage = datasets.fetch_surf_fsaverage()

    # 步驟 1: 將 3D 體積數據投影到左右大腦半球的表面
    print("正在將體積數據投影到表面...")
    texture_left = vol_to_surf(final_upsampled_path, fsaverage.pial_left)
    texture_right = vol_to_surf(final_upsampled_path, fsaverage.pial_right)

    # 步驟 2: 將投影後的表面數據 (texture) 繪製出來
    print("正在繪製左半球...")
    plot_surf_stat_map(
        surf_mesh=fsaverage.pial_left,
        stat_map=texture_left, # <--- 使用投影後的 NumPy 陣列
        bg_map=fsaverage.sulc_left,
        threshold=threshold_value,
        view='lateral',
        title='DMN on Left Hemisphere'
    )

    print("正在繪製右半球...")
    plot_surf_stat_map(
        surf_mesh=fsaverage.pial_right,
        stat_map=texture_right, # <--- 使用投影後的 NumPy 陣列
        bg_map=fsaverage.sulc_right,
        threshold=threshold_value,
        view='lateral',
        title='DMN on Right Hemisphere'
    )
    
    print("\n所有流程執行完畢！互動式 3D 圖已顯示。")
    # ... (前面的程式碼不變) ...

    # ===============================================================
    #  方法一：繪製玻璃腦圖 (Glass Brain)
    # ===============================================================
    print("\n--- 方法一：正在產生玻璃腦圖 ---")

    # 我們需要 final_upsampled_path 和 threshold_value 這兩個變數
    display_glass = plotting.plot_glass_brain(
        final_upsampled_path,
        title='DMN Full Network (Glass Brain)',
        threshold=threshold_value,
        colorbar=True
    )

    output_image_path_glass = os.path.join(figures_dir, "final_dmn_glass_brain.png")
    display_glass.savefig(output_image_path_glass)
    print(f"玻璃腦圖已儲存至: {output_image_path_glass}")


    # ===============================================================
    #  方法一：繪製玻璃腦圖 (Glass Brain)
    # ===============================================================
    print("\n--- 方法一：正在產生玻璃腦圖 ---")

    # 我們需要 final_upsampled_path 和 threshold_value 這兩個變數
    display_glass = plotting.plot_glass_brain(
        final_upsampled_path,
        title='DMN Full Network (Glass Brain)',
        threshold=threshold_value,
        colorbar=True
    )

    output_image_path_glass = os.path.join(figures_dir, "final_dmn_glass_brain.png")
    display_glass.savefig(output_image_path_glass)
    print(f"玻璃腦圖已儲存至: {output_image_path_glass}")

    # ... (前面的程式碼不變) ...

    # ===============================================================
    #  方法一：繪製玻璃腦圖 (擴展版：探索不同門檻的效果)
    # ===============================================================
    print("\n--- 方法一：正在產生玻璃腦圖 (探索不同門檻) ---")

    # --- 實驗一：使用比較寬鬆的門檻 (例如 Top 5%) ---
    print("正在使用 Top 5% 的寬鬆門檻...")
    threshold_value_lenient = np.percentile(final_dmn_data[final_dmn_data > 0], 95) # 改成 95

    display_glass_lenient = plotting.plot_glass_brain(
        final_upsampled_path,
        title='DMN Full Network (Lenient Threshold: Top 5%)',
        threshold=threshold_value_lenient,
        colorbar=True
    )
    output_image_path_glass_lenient = os.path.join(figures_dir, "final_dmn_glass_brain_lenient.png")
    display_glass_lenient.savefig(output_image_path_glass_lenient)
    print(f"寬鬆門檻的玻璃腦圖已儲存至: {output_image_path_glass_lenient}")

    # --- 實驗二：完全不設門檻，看看原始數據分佈 ---
    print("正在顯示無門檻的原始數據...")
    display_glass_raw = plotting.plot_glass_brain(
        final_upsampled_path,
        title='DMN Full Network (No Threshold)',
        threshold=0, # 設為 0 等於不篩選
        colorbar=True
    )
    output_image_path_glass_raw = os.path.join(figures_dir, "final_dmn_glass_brain_raw.png")
    display_glass_raw.savefig(output_image_path_glass_raw)
    print(f"無門檻的玻璃腦圖已儲存至: {output_image_path_glass_raw}")


    # --- 實驗三：使用一個固定的數值作為門檻 (常用方法) ---
    print("正在使用固定的數值門檻...")
    # 這個數值需要根據 colorbar 來觀察決定，我們先試試 5.0
    display_glass_fixed = plotting.plot_glass_brain(
        final_upsampled_path,
        title='DMN Full Network (Fixed Threshold > 5.0)',
        threshold=5.0,
        colorbar=True
    )
    output_image_path_glass_fixed = os.path.join(figures_dir, "final_dmn_glass_brain_fixed.png")
    display_glass_fixed.savefig(output_image_path_glass_fixed)
    print(f"固定門檻的玻璃腦圖已儲存至: {output_image_path_glass_fixed}")


