import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from nilearn import plotting
# 🚨 修正點 1: 使用別名確保 MNI 模板載入函式範圍正確
from nilearn.datasets import load_mni152_template as nilearn_load_mni_template 
from nilearn.image import resample_to_img 
import matplotlib.pyplot as plt
import os
import glob

# ====================================================================
# 【設定與配置】
# ====================================================================

ACTIVATION_DIR = 'output/cnn_3d/activations/'
OUTPUT_DIR = 'output/cnn_3d/xai_heatmaps/'

# 初始目標尺寸 (會被 MNI 模板的實際尺寸覆蓋)
TARGET_SHAPE = (99, 117, 95) 

# 【核心修正點 2: 動態載入 MNI 模板並設定 TARGET_SHAPE】
MNI_TEMPLATE_IMG = None
try:
    MNI_TEMPLATE_IMG = nilearn_load_mni_template() 
    
    # 🚨 修正點：將 TARGET_SHAPE 修正為 MNI 模板的實際維度
    TARGET_SHAPE = MNI_TEMPLATE_IMG.get_fdata().shape 
    
    print(f"✅ MNI 模板載入成功，TARGET_SHAPE 已動態修正為: {TARGET_SHAPE}")
except Exception as e:
    print(f"🚨 錯誤：載入 Nilearn MNI 模板失敗。PNG 視覺化將被跳過。錯誤: {e}")
    MNI_TEMPLATE_IMG = None

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ====================================================================
# 【核心處理函數】
# ====================================================================

def process_single_activation(pt_path, npy_path, target_shape, mni_template_img):
    
    base_name = os.path.basename(pt_path)
    subject_id = base_name.replace('_activation.pt', '')
    output_nii_path = os.path.join(OUTPUT_DIR, f'{subject_id}_xai_heatmap.nii.gz')
    
    print(f"\n-> 處理病患 ID: {subject_id}")

    if mni_template_img is None:
        print("  🚨 致命錯誤：無法執行空間對齊，跳過儲存。")
        return

    try:
        # 1. 載入與通道平均
        activation_tensor = torch.load(pt_path)
        affine_matrix = np.load(npy_path)
        activation_np = activation_tensor.numpy()
        
        if activation_np.ndim == 5 and activation_np.shape[0] == 1:
            activation_np = activation_np.squeeze(axis=0)
        
        heatmap_raw = np.mean(activation_np, axis=0) 
        
        # 2. 上取樣 (Upsampling) - 使用動態修正後的 TARGET_SHAPE
        current_shape = heatmap_raw.shape
        zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
        
        print(f"  DEBUG: 熱圖原始 Shape: {current_shape}, Zoom Factors: {zoom_factors}")
        heatmap_upsampled = zoom(heatmap_raw, zoom=zoom_factors, order=1)
        
        if heatmap_upsampled.shape != target_shape:
            # 這應該不會發生，但以防萬一
            raise RuntimeError(f"上取樣後尺寸 {heatmap_upsampled.shape} 與目標 {target_shape} 不符。")
        
        # 3. 正規化與閾值化
        min_val = heatmap_upsampled.min()
        max_val = heatmap_upsampled.max()
        heatmap_normalized = (heatmap_upsampled - min_val) / (max_val - min_val + 1e-8)
        
        nonzero_values = heatmap_normalized[heatmap_normalized > 0]
        threshold = np.percentile(nonzero_values, 98) if len(nonzero_values) >= 10 else 0
        heatmap_final = np.where(heatmap_normalized >= threshold, heatmap_normalized, 0)
        
        
        # ===============================================================
        # 【核心修正點 3: 強制 X 軸翻轉與 Affine 替換】
        # ===============================================================
        
        # 翻轉 X 軸 (軸向 0) 以修正 L/R 鏡像錯誤
        heatmap_data_to_save = np.flip(heatmap_final, axis=0)
        print("  ✅ 已執行強制 X 軸翻轉，修正 L/R 鏡像。")

        # 強制替換 Affine 為 MNI 模板的 Affine
        aligned_affine = mni_template_img.affine 
        
        # 創建最終 NIfTI 影像
        nii_img_aligned = nib.Nifti1Image(heatmap_data_to_save.astype(np.float32), aligned_affine)
        
        
        # 4. 儲存最終 NIfTI 熱圖
        nib.save(nii_img_aligned, output_nii_path)
        print(f"  ✅ 已成功儲存最終 NIfTI 熱圖至: {output_nii_path}")

        
        # 5. 自動生成 2D 視覺化 PNG
        viz_output_path = output_nii_path.replace('.nii.gz', '_viz.png')
        vmax_val = nii_img_aligned.get_fdata().max()

        plotting.plot_stat_map(
            stat_map_img=nii_img_aligned, 
            bg_img=mni_template_img,    
            title=f"Activation Map - {subject_id}",
            display_mode='z',           
            cut_coords=8,              
            cmap='hot',                
            black_bg=True,             
            colorbar=True,
            output_file=viz_output_path,
            vmax=vmax_val
        )
        
        plotting.plot_stat_map(
            stat_map_img=nii_img_aligned,
            bg_img=mni_template_img,
            display_mode='x',
            cut_coords=5,
            cmap='hot',
            black_bg=True,
            colorbar=False,
            output_file=viz_output_path.replace('.png', '_sagittal.png'),
            vmax=vmax_val
        )
        
        print(f"  🖼️ 已成功生成視覺化 PNG 至: {viz_output_path}")

    except Exception as e:
        print(f"  ❌ 處理 {subject_id} 時發生致命錯誤: {e}")
        return


# ====================================================================
# 【主執行區塊】
# ====================================================================

def main():
    
    pt_files = sorted(glob.glob(os.path.join(ACTIVATION_DIR, '*_activation.pt')))
    
    if not pt_files:
        print(f"🚨 錯誤：在 {ACTIVATION_DIR} 中找不到任何 .pt 激活檔案。請檢查階段 1 的輸出。")
        return

    total_processed = 0
    for pt_path in pt_files:
        npy_path = pt_path.replace('_activation.pt', '_affine.npy')
        
        if not os.path.exists(npy_path):
            print(f"⚠️ 警告：找不到對應的 Affine 檔案: {npy_path}。跳過此病患。")
            continue
        
        # 呼叫時傳遞 MNI 模板物件和修正後的 TARGET_SHAPE
        process_single_activation(pt_path, npy_path, TARGET_SHAPE, MNI_TEMPLATE_IMG) 
        total_processed += 1
    
    print(f"\n--- 流程結束 ---")
    print(f"總共嘗試處理 {total_processed} 個病患。")


if __name__ == '__main__':
    main()