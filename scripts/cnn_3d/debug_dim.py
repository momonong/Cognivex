import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from nilearn import plotting
from nilearn.datasets import load_mni152_template as nilearn_load_mni_template
import os
import glob
import matplotlib.pyplot as plt

# ====================================================================
# 【設定與配置】
# ====================================================================

ACTIVATION_DIR = 'output/cnn_3d/activations/'
OUTPUT_DIR = 'output/cnn_3d/xai_heatmaps_debug/' # 使用獨立的 DEBUG 輸出目錄
TARGET_SHAPE = (99, 117, 95) 

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 載入 Nilearn 標準 MNI 模板
MNI_TEMPLATE_IMG = None
try:
    MNI_TEMPLATE_IMG = nilearn_load_mni_template() 
    print(f"✅ MNI 模板載入成功 (Shape: {MNI_TEMPLATE_IMG.get_fdata().shape})")
except Exception as e:
    print(f"🚨 致命錯誤：無法載入 MNI 模板。錯誤: {e}")
    exit() # 如果模板載入失敗，則終止腳本

MNI_AFFINE = MNI_TEMPLATE_IMG.affine


# ====================================================================
# 【核心修正與診斷函數】
# ====================================================================

def debug_flip_and_plot(heatmap_data, mni_affine, subject_id, flip_axes=None, description="Original"):
    """ 
    對熱圖數據進行指定的翻轉，強制賦予 MNI Affine，並繪圖儲存。
    """
    if flip_axes is None:
        flipped_data = heatmap_data
    else:
        # 對指定軸進行翻轉
        flipped_data = np.flip(heatmap_data, axis=flip_axes)
    
    # 強制賦予 MNI Affine
    nii_img_debug = nib.Nifti1Image(flipped_data.astype(np.float32), mni_affine)
    
    viz_output_path = os.path.join(OUTPUT_DIR, f'{subject_id}_{description.replace(" ", "_")}_debug.png')
    
    try:
        # 繪製 Axial (Z) 視圖進行確認
        plotting.plot_stat_map(
            stat_map_img=nii_img_debug, 
            bg_img=MNI_TEMPLATE_IMG,    
            title=f"DEBUG: {description}",
            display_mode='z',           
            cut_coords=5,              
            cmap='hot',                
            black_bg=True,             
            colorbar=False,
            output_file=viz_output_path,
        )
        print(f"  🖼️  已儲存: {description} 診斷圖")

    except Exception as e:
        print(f"  ❌ 繪圖失敗 ({description}): {e}")


def prepare_heatmap_data(pt_path, npy_path, target_shape):
    """ 執行載入、通道平均、上取樣和正規化，返回數據陣列。"""
    
    # ... (載入和通道平均邏輯, 假設 activation_tensor 和 affine_matrix 已載入)
    try:
        activation_tensor = torch.load(pt_path)
        affine_matrix = np.load(npy_path)
    except Exception as e:
        print(f"  ❌ 載入檔案失敗: {e}")
        return None, None
        
    activation_np = activation_tensor.numpy()
    if activation_np.ndim == 5 and activation_np.shape[0] == 1:
        activation_np = activation_np.squeeze(axis=0)
    heatmap_raw = np.mean(activation_np, axis=0) 

    # 上取樣
    current_shape = heatmap_raw.shape
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
    heatmap_upsampled = zoom(heatmap_raw, zoom=zoom_factors, order=1)
    
    # 正規化和閾值化
    min_val = heatmap_upsampled.min()
    max_val = heatmap_upsampled.max()
    heatmap_normalized = (heatmap_upsampled - min_val) / (max_val - min_val + 1e-8)
    nonzero_values = heatmap_normalized[heatmap_normalized > 0]
    threshold = np.percentile(nonzero_values, 98) if len(nonzero_values) >= 10 else 0
    heatmap_final = np.where(heatmap_normalized >= threshold, heatmap_normalized, 0)
    
    return heatmap_final, affine_matrix

# ====================================================================
# 【主執行區塊】
# ====================================================================

def main():
    print("=======================================================")
    print("🧠 空間軸向與對齊診斷啟動")
    print("=======================================================")
    
    pt_files = sorted(glob.glob(os.path.join(ACTIVATION_DIR, '*_activation.pt')))
    
    if not pt_files:
        print(f"🚨 錯誤：在 {ACTIVATION_DIR} 中找不到任何 Activation 檔案。")
        return

    # 選擇第一個病患進行診斷
    first_pt_path = pt_files[0]
    first_npy_path = first_pt_path.replace('_activation.pt', '_affine.npy')
    subject_id = os.path.basename(first_pt_path).replace('_activation.pt', '')
    
    print(f"-> 診斷病患 ID: {subject_id}")

    heatmap_data, _ = prepare_heatmap_data(first_pt_path, first_npy_path, TARGET_SHAPE)
    
    if heatmap_data is None:
        print("🚨 數據準備失敗，終止診斷。")
        return

    # ----------------------------------------------------
    # 執行所有軸向翻轉組合測試
    # ----------------------------------------------------
    
    # 軸向：0=X, 1=Y, 2=Z
    flip_tests = {
        "0-Original": None,
        "1-Flip X (Left-Right)": 0,
        "2-Flip Y (Anterior-Posterior)": 1,
        "3-Flip Z (Superior-Inferior)": 2,
        "4-Flip X & Y": (0, 1),
        "5-Flip Y & Z": (1, 2),
        "6-Flip X & Z": (0, 2),
        "7-Flip X, Y, Z": (0, 1, 2)
    }

    for desc, axes in flip_tests.items():
        debug_flip_and_plot(heatmap_data, MNI_AFFINE, subject_id, axes, desc)
        
    print("\n=======================================================")
    print(f"✅ 診斷完成。請檢查 {OUTPUT_DIR} 中的 8 張 PNG 圖片。")
    print("請找出熱圖與腦結構完美對齊的圖片，並記住對應的翻轉組合名稱。")
    print("=======================================================")


if __name__ == '__main__':
    # 為了運行方便，請將這個邏輯直接放在您的腳本中，或者複製到一個新的 Python 檔案中運行。
    main()