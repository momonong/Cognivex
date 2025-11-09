import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
from scipy.ndimage import zoom
from nilearn import plotting
from nilearn.datasets import load_mni152_template, fetch_atlas_aal
from captum.attr import GuidedBackprop
from nilearn.image import resample_to_img, smooth_img 
import matplotlib.pyplot as plt
import pandas as pd 
import glob
import time

# ====================================================================
# 【1. 設定與配置】
# ====================================================================
MODEL_DIR = 'model/cnn_3d/' 
DATA_DIR = "E:/fMRI/Model/sMRI_data" 
OUTPUT_DIR = "output/cnn_3d/final_analysis_results/" # 最終成果目錄
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================================================================
# 【2. 模型定義 (已修正 ReLU)】
# ====================================================================
class Simple3DCNN_InstanceNorm(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(Simple3DCNN_InstanceNorm, self).__init__()
        def create_conv_block(in_c, out_c, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=padding),
                nn.InstanceNorm3d(out_c),
                nn.ReLU(inplace=False), 
                nn.MaxPool3d(kernel_size=2, stride=2))
        self.block1 = create_conv_block(in_channels, 16)
        self.block2 = create_conv_block(16, 32)
        self.block3 = create_conv_block(32, 64)
        self.block4 = create_conv_block(64, 128)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        x = self.global_avg_pool(x); x = torch.flatten(x, 1); x = self.classifier(x)
        return x

# ====================================================================
# 【3. 輔助函數】
# ====================================================================
gradients = {}; activations = {}
def backward_hook(module, grad_in, grad_out): gradients['value'] = grad_out[0]
def forward_hook(module, input, output): activations['value'] = output

def load_nifti_as_tensor(file_path):
    img = nib.load(file_path)
    affine = img.affine
    data_np = img.get_fdata(dtype=np.float32)
    tensor = torch.tensor(data_np).unsqueeze(0).unsqueeze(0) 
    return tensor, affine

def normalize_map(map_data):
    min_val, max_val = map_data.min(), map_data.max()
    return (map_data - min_val) / (max_val - min_val + 1e-8)

def get_guided_gradcam(model, input_tensor, target_class_index, mni_target_shape, original_affine, mni_template):
    """
    為「單一模型」計算「單一」Guided Grad-CAM 熱圖 (已對齊到 MNI)。
    """
    # --- 任務 1：Grad-CAM (低解析度) ---
    model.block4.register_forward_hook(forward_hook); model.block4.register_backward_hook(backward_hook)
    output = model(input_tensor)
    one_hot_output = torch.zeros((1, output.size()[-1]), device=DEVICE); one_hot_output[0][target_class_index] = 1
    model.zero_grad(); output.backward(gradient=one_hot_output, retain_graph=True)
    guided_gradients = gradients['value'][0]; feature_maps = activations['value'][0]   
    weights = torch.mean(guided_gradients, dim=[1, 2, 3])
    grad_cam_map_low_res = torch.zeros(feature_maps.shape[1:], device=DEVICE)
    for i in range(feature_maps.shape[0]): grad_cam_map_low_res += weights[i] * feature_maps[i, :, :, :]
    grad_cam_map_low_res = torch.relu(grad_cam_map_low_res)
    grad_cam_np_low_res = grad_cam_map_low_res.cpu().detach().numpy()

    # --- 任務 2：Guided Backprop (高解析度) ---
    model.zero_grad(); gbp = GuidedBackprop(model)
    guided_grads_tensor = gbp.attribute(input_tensor, target=target_class_index)
    guided_grads_np = guided_grads_tensor.squeeze(0).squeeze(0).cpu().detach().numpy()

    # --- 任務 3：結合與空間對齊 ---
    zoom_factors = [t / c for t, c in zip(mni_target_shape, grad_cam_np_low_res.shape)]
    grad_cam_high_res = zoom(grad_cam_np_low_res, zoom=zoom_factors, order=1)
    gbp_nii_raw = nib.Nifti1Image(guided_grads_np, original_affine)
    
    gbp_nii_aligned = resample_to_img(
        source_img=gbp_nii_raw, target_img=mni_template,
        interpolation='continuous', force_resample=True, copy_header=True
    )
    guided_grads_aligned_np = gbp_nii_aligned.get_fdata()

    grad_cam_norm = normalize_map(grad_cam_high_res)
    guided_grads_norm = normalize_map(np.abs(guided_grads_aligned_np))
    guided_grad_cam_map = grad_cam_norm * guided_grads_norm
    heatmap_flipped = np.flip(guided_grad_cam_map, axis=0) 
    
    return heatmap_flipped

# ====================================================================
# 【5. 主執行腳本 (批次處理版)】
# ====================================================================

def main():
    print(f"--- Cognivex V3 批次分析啟動 ---")
    print(f"使用設備: {DEVICE}")

    # 1. 載入 MNI 模板與 AAL Atlas (僅執行一次)
    try:
        print("... 正在載入 MNI 模板與 AAL Atlas...")
        mni_template = load_mni152_template() 
        MNI_TARGET_SHAPE = mni_template.get_fdata().shape
        MNI_AFFINE = mni_template.affine
        mni_brain_mask = mni_template.get_fdata() > 0 
        
        aal_atlas_data = fetch_atlas_aal(version='SPM12')
        aal_label_names = aal_atlas_data.labels
        # 🚨 修正：將 AAL 索引轉換為整數列表
        aal_label_indices_int = [int(i) for i in aal_atlas_data.indices] 
        
        aal_img = nib.load(aal_atlas_data.maps)
        
        print("... 正在將 AAL Atlas 配準到 MNI 模板空間 (僅執行一次)...")
        aal_img_resampled = resample_to_img(aal_img, mni_template, interpolation='nearest', force_resample=True, copy_header=True)
        aal_data_np = aal_img_resampled.get_fdata() # 這是配準後的 AAL 數據 (整數)
        
        print(f"✅ MNI 模板載入成功，目標尺寸: {MNI_TARGET_SHAPE}")
        print(f"✅ AAL Atlas 載入並配準成功。")
    except Exception as e:
        print(f"🚨 致命錯誤：載入 MNI/AAL 模板失敗。錯誤: {e}"); return

    # 2. 載入 5 個模型路徑
    model_paths = sorted(glob.glob(os.path.join(MODEL_DIR, 'cnn_3d_fold_*.pth')))
    if not model_paths:
        print(f"🚨 錯誤：在 {MODEL_DIR} 中找不到任何 'cnn_3d_fold_*.pth' 模型權重。")
        return
    print(f"找到 {len(model_paths)} 個模型 Fold。")

    # 3. 遞迴搜尋所有 sMRI 檔案
    print(f"正在 {DATA_DIR} 中「遞迴」搜尋 .nii 和 .nii.gz 檔案...")
    search_path_nii = os.path.join(DATA_DIR, '**', '*.nii')
    search_path_niigz = os.path.join(DATA_DIR, '**', '*.nii.gz')
    data_files = sorted(glob.glob(search_path_nii, recursive=True) + glob.glob(search_path_niigz, recursive=True))
    
    if not data_files:
        print(f"🚨 錯誤：在 {DATA_DIR} 及其子目錄中找不到任何 .nii 或 .nii.gz 檔案。")
        return
    print(f"找到 {len(data_files)} 筆 sMRI 資料。開始執行「批次集成歸因」...")

    # ----------------------------------------------------
    # 外層迴圈：遍歷所有病患
    # ----------------------------------------------------
    start_time_total = time.time()
    for file_index, file_path in enumerate(data_files):
        
        start_time_patient = time.time()
        subject_id = os.path.basename(file_path).replace('.nii.gz', '').replace('.nii', '')
        print(f"\n--- 正在處理病患 {file_index + 1} / {len(data_files)} ---")
        print(f"   ID: {subject_id}")
        
        try:
            # 載入資料
            input_tensor, original_affine = load_nifti_as_tensor(file_path)
            input_tensor = input_tensor.to(DEVICE); input_tensor.requires_grad_()
            
            fold_heatmaps = [] # 儲存 5 個 Fold 的熱圖 (NumPy array)
            target_class_index = 1 # 假設 1 = AD
            
            # 內層迴圈：遍歷 5 個模型
            for model_path in model_paths:
                model = Simple3DCNN_InstanceNorm(num_classes=2).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                
                # 計算單一模型的 Guided Grad-CAM
                heatmap_single_fold = get_guided_gradcam(
                    model, input_tensor, target_class_index, 
                    MNI_TARGET_SHAPE, original_affine, mni_template
                )
                fold_heatmaps.append(heatmap_single_fold)
            
            # 5. 集成 (Ensemble) 熱圖：計算平均值
            stacked_heatmaps = np.stack(fold_heatmaps, axis=0)
            heatmap_ensembled = np.mean(stacked_heatmaps, axis=0)
            
            # 6. 應用腦部遮罩
            heatmap_masked = heatmap_ensembled * mni_brain_mask 
            nii_aligned = nib.Nifti1Image(heatmap_masked, MNI_AFFINE) 
            
            print(f"   ✅ [任務 3] 集成與空間對齊完成。")

            # 7. 腦區量化 (AAL Atlas)
            gradcam_data = nii_aligned.get_fdata() 
            region_activations = []
            for label_name, label_id in zip(aal_label_names, aal_label_indices_int):
                if label_id == 0: continue
                region_mask = (aal_data_np == label_id) 
                activations_in_region = gradcam_data[region_mask]
                total_activation = np.sum(activations_in_region)
                num_voxels = np.sum(region_mask)
                average_activation = total_activation / num_voxels if num_voxels > 0 else 0
                region_activations.append({'Region': label_name, 'Label_ID': label_id, 'Average_Activation': average_activation})

            df_activations = pd.DataFrame(region_activations)
            df_activations = df_activations.sort_values(by='Average_Activation', ascending=False)
            csv_path = os.path.join(OUTPUT_DIR, f"{subject_id}_brain_region_activations.csv")
            df_activations.to_csv(csv_path, index=False)
            print(f"   ✅ [任務 4] 腦區量化完成 -> {csv_path}")
            
            # 8. 視覺化 (平滑並儲存)
            nii_path = os.path.join(OUTPUT_DIR, f"{subject_id}_GUIDED_gradcam_ALIGNED_MASKED.nii.gz") 
            png_path = os.path.join(OUTPUT_DIR, f"{subject_id}_GUIDED_gradcam_VISUAL_XYZ_MASKED_SMOOTHED.png")
            nib.save(nii_aligned, nii_path) 
            
            nii_smoothed = smooth_img(nii_aligned, fwhm=4) # 4mm 平滑
            smoothed_data = nii_smoothed.get_fdata()
            nonzero_values_smoothed = smoothed_data[smoothed_data > 0]
            stat_threshold_smoothed = np.percentile(nonzero_values_smoothed, 85) if len(nonzero_values_smoothed) > 0 else 0
            
            fig, axes = plt.subplots(nrows=3, figsize=(15, 10))
            fig.suptitle(f"Guided Grad-CAM (Ensembled & Smoothed) - {subject_id}", fontsize=16, y=1.02)
            plotting.plot_stat_map(nii_smoothed, bg_img=mni_template, display_mode='z', cut_coords=7, cmap='hot', black_bg=True, colorbar=True, threshold=stat_threshold_smoothed, axes=axes[0], title="Axial (Z)")
            plotting.plot_stat_map(nii_smoothed, bg_img=mni_template, display_mode='x', cut_coords=6, cmap='hot', black_bg=True, colorbar=False, threshold=stat_threshold_smoothed, axes=axes[1], title="Sagittal (X)")
            plotting.plot_stat_map(nii_smoothed, bg_img=mni_template, display_mode='y', cut_coords=6, cmap='hot', black_bg=True, colorbar=False, threshold=stat_threshold_smoothed, axes=axes[2], title="Coronal (Y)")
            fig.savefig(png_path, bbox_inches='tight', dpi=150)
            plt.close(fig) 
            
            print(f"   ✅ [任務 8] 視覺化完成 -> {png_path}")
            
        except Exception as e:
            print(f"   ❌❌❌ 處理 {subject_id} 時發生致命錯誤: {e}")
            if 'out of memory' in str(e):
                print("   🚨 偵測到 CUDA Out of Memory！請嘗試重新啟動。")
                break # 停止批次處理
            
        elapsed_patient = time.time() - start_time_patient
        print(f"   ... 單一病患處理完成，耗時: {elapsed_patient:.2f} 秒")

    elapsed_total = time.time() - start_time_total
    print(f"\n--- 批次處理全部完成 ---")
    print(f"總共處理了 {len(data_files)} 筆資料，總耗時: {elapsed_total / 60:.2f} 分鐘")

if __name__ == '__main__':
    main()