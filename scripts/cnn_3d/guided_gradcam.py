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

# ====================================================================
# 【1. 設定與配置】(保持不變)
# ====================================================================
TEST_DATA_PATH = "E:/fMRI/Model/sMRI_data/AD/T1_3D_MPRAGE_SAG_0003_008/T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii"
TEST_MODEL_PATH = "model/cnn_3d/cnn_3d_fold_1.pth" 
OUTPUT_DIR = "output/cnn_3d/guided_gradcam_test/" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================================================================
# 【2. 模型定義 (已修正 ReLU)】(保持不變)
# ====================================================================
class Simple3DCNN_InstanceNorm(nn.Module):
    # ... (您的模型定義保持不變，此處省略)
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
# 【3. 輔助函數】(保持不變)
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

# ====================================================================
# 【4. 主測試腳本 (🚨 修正 任務 4)】
# ====================================================================

def run_single_test():
    print(f"--- 開始單一檔案 Guided Grad-CAM 測試 ---")
    print(f"使用設備: {DEVICE}")

    # 1. 載入 MNI 模板與 AAL Atlas
    try:
        mni_template = load_mni152_template() 
        MNI_TARGET_SHAPE = mni_template.get_fdata().shape
        MNI_AFFINE = mni_template.affine
        mni_brain_mask = mni_template.get_fdata() > 0 
        
        aal_atlas_data = fetch_atlas_aal(version='SPM12')
        aal_label_names = aal_atlas_data.labels
        aal_label_indices = aal_atlas_data.indices # 這是 ['0', '2001', '2002', ...] (字串)
        
        aal_img = nib.load(aal_atlas_data.maps)
        
        print("... Z 正在將 AAL Atlas 配準到 MNI 模板空間...")
        aal_img_resampled = resample_to_img(aal_img, mni_template, interpolation='nearest', force_resample=True, copy_header=True)
        aal_data_np = aal_img_resampled.get_fdata() # 這是配準後的 AAL 數據 (整數)
        
        print(f"✅ MNI 模板載入成功，目標尺寸: {MNI_TARGET_SHAPE}")
        print(f"✅ AAL Atlas 載入並配準成功。")
    except Exception as e:
        print(f"🚨 致命錯誤：載入 MNI/AAL 模板失敗。錯誤: {e}"); return

    # 2. 載入模型 (略)
    model = Simple3DCNN_InstanceNorm(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(TEST_MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ 模型載入成功: {TEST_MODEL_PATH}")

    # 3. 載入單一 NIfTI 檔案
    input_tensor, original_affine = load_nifti_as_tensor(TEST_DATA_PATH)
    input_tensor = input_tensor.to(DEVICE); input_tensor.requires_grad_()
    subject_id = os.path.basename(TEST_DATA_PATH).replace('.nii', '').replace('.nii.gz', '')
    print(f"✅ 測試資料載入成功: {TEST_DATA_PATH}")

    # --- 任務 1 & 2 (Grad-CAM & Guided Backprop) ---
    # ... (此部分邏輯保持不變，此處省略)
    print("... [任務 1] 正在計算 Grad-CAM (低解析度)...")
    model.block4.register_forward_hook(forward_hook); model.block4.register_backward_hook(backward_hook)
    output = model(input_tensor); target_class_index = 1 
    one_hot_output = torch.zeros((1, output.size()[-1]), device=DEVICE); one_hot_output[0][target_class_index] = 1
    model.zero_grad(); output.backward(gradient=one_hot_output, retain_graph=True)
    guided_gradients = gradients['value'][0]; feature_maps = activations['value'][0]   
    weights = torch.mean(guided_gradients, dim=[1, 2, 3])
    grad_cam_map_low_res = torch.zeros(feature_maps.shape[1:], device=DEVICE)
    for i in range(feature_maps.shape[0]): grad_cam_map_low_res += weights[i] * feature_maps[i, :, :, :]
    grad_cam_map_low_res = torch.relu(grad_cam_map_low_res)
    grad_cam_np_low_res = grad_cam_map_low_res.cpu().detach().numpy()
    print("... [任務 2] 正在計算 Guided Backprop (高解析度)...")
    model.zero_grad(); gbp = GuidedBackprop(model)
    guided_grads_tensor = gbp.attribute(input_tensor, target=target_class_index)
    guided_grads_np = guided_grads_tensor.squeeze(0).squeeze(0).cpu().detach().numpy()

    # --- 任務 3：結合與空間對齊 ---
    # ... (此部分邏輯保持不變，此處省略)
    print("... [任務 3] 正在結合熱圖並執行空間對齊...")
    zoom_factors = [t / c for t, c in zip(MNI_TARGET_SHAPE, grad_cam_np_low_res.shape)]
    grad_cam_high_res = zoom(grad_cam_np_low_res, zoom=zoom_factors, order=1)
    gbp_nii_raw = nib.Nifti1Image(guided_grads_np, original_affine)
    print("... [任務 3b] 正在將 Guided Backprop 配準到 MNI 空間...")
    gbp_nii_aligned = resample_to_img(source_img=gbp_nii_raw, target_img=mni_template, interpolation='continuous', force_resample=True, copy_header=True)
    guided_grads_aligned_np = gbp_nii_aligned.get_fdata()
    grad_cam_norm = normalize_map(grad_cam_high_res)
    guided_grads_norm = normalize_map(np.abs(guided_grads_aligned_np))
    guided_grad_cam_map = grad_cam_norm * guided_grads_norm
    heatmap_flipped = np.flip(guided_grad_cam_map, axis=0) 
    heatmap_masked = heatmap_flipped * mni_brain_mask 
    nii_aligned = nib.Nifti1Image(heatmap_masked, MNI_AFFINE) 
    print("✅ 空間對齊完成！")

    # ----------------------------------------------------
    # 【🚨 核心修正點：任務 4 - 腦區量化 (AAL Atlas)】
    # ----------------------------------------------------
    print("... [任務 4] 正在進行腦區量化 (AAL Atlas)...")
    
    gradcam_data = nii_aligned.get_fdata() 
    
    region_activations = []
    
    # 🚨 修正：同時遍歷標籤名稱 (label_name) 和 標籤 ID (label_id)
    for label_name, label_id_str in zip(aal_label_names, aal_label_indices):
        
        # 🚨 修正：將 label_id 從 'str' 轉換為 'int'
        label_id_int = int(label_id_str)
        
        if label_id_int == 0: # 跳過 'Background'
            continue
            
        # 🚨 修正：使用整數 ID (label_id_int) 進行比較
        region_mask = (aal_data_np == label_id_int) 
        
        activations_in_region = gradcam_data[region_mask]
        total_activation = np.sum(activations_in_region)
        num_voxels = np.sum(region_mask) # 這次 Num_Voxels 將會 > 0
        average_activation = total_activation / num_voxels if num_voxels > 0 else 0
        
        region_activations.append({
            'Region': label_name,
            'Label_ID': label_id_int,
            'Total_Activation': total_activation,
            'Average_Activation': average_activation,
            'Num_Voxels': num_voxels
        })

    df_activations = pd.DataFrame(region_activations)
    df_activations = df_activations.sort_values(by='Average_Activation', ascending=False)
    
    csv_path = os.path.join(OUTPUT_DIR, f"{subject_id}_brain_region_activations.csv")
    df_activations.to_csv(csv_path, index=False)
    
    print(f"✅ 腦區量化完成！結果已儲存至: {csv_path}")
    print("\n--- 最活躍的腦區 (前 10 名) ---")
    print(df_activations.head(10).to_string(index=False))

    # ----------------------------------------------------
    # 【任務 8：儲存結果 (視覺化平滑)】
    # ----------------------------------------------------
    nii_path = os.path.join(OUTPUT_DIR, f"{subject_id}_GUIDED_gradcam_ALIGNED_MASKED.nii.gz") 
    png_path = os.path.join(OUTPUT_DIR, f"{subject_id}_GUIDED_gradcam_VISUAL_XYZ_MASKED_SMOOTHED.png")
    
    nib.save(nii_aligned, nii_path) 
    print(f"💾 已儲存對齊的 NIfTI 檔案至: {nii_path}")
    
    print("... 正在平滑熱圖以進行視覺化...")
    nii_smoothed = smooth_img(nii_aligned, fwhm=4) # FWHM=4mm 平滑
    
    smoothed_data = nii_smoothed.get_fdata()
    nonzero_values_smoothed = smoothed_data[smoothed_data > 0]
    if len(nonzero_values_smoothed) > 0:
        stat_threshold_smoothed = np.percentile(nonzero_values_smoothed, 85) # 保持 85th
    else:
        stat_threshold_smoothed = 0
    print(f"✅ 平滑後統計閾值 (85th percentile) 計算完成: {stat_threshold_smoothed:.4f}")
    
    fig, axes = plt.subplots(nrows=3, figsize=(15, 10))
    fig.suptitle(f"Guided Grad-CAM (Smoothed & Masked) - {subject_id}", fontsize=16, y=1.02)
    
    plotting.plot_stat_map(nii_smoothed, bg_img=mni_template, display_mode='z', cut_coords=7, cmap='hot', black_bg=True, colorbar=True, threshold=stat_threshold_smoothed, axes=axes[0], title="Axial (Z)")
    plotting.plot_stat_map(nii_smoothed, bg_img=mni_template, display_mode='x', cut_coords=6, cmap='hot', black_bg=True, colorbar=False, threshold=stat_threshold_smoothed, axes=axes[1], title="Sagittal (X)")
    plotting.plot_stat_map(nii_smoothed, bg_img=mni_template, display_mode='y', cut_coords=6, cmap='hot', black_bg=True, colorbar=False, threshold=stat_threshold_smoothed, axes=axes[2], title="Coronal (Y)")
    
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig) 
    
    print(f"🖼️  已儲存「平滑後」的 XYZ 三視圖 PNG 檔案至: {png_path}")
    print(f"--- 高解析度測試完成 ---")

if __name__ == '__main__':
    run_single_test()