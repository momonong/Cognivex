import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
from scipy.ndimage import zoom
from nilearn import plotting
# 修正匯入
from nilearn.datasets import load_mni152_template 
from captum.attr import GuidedBackprop
from nilearn.image import resample_to_img 
# 🚨 引入 Matplotlib 用於組合視圖
import matplotlib.pyplot as plt

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
# 【3. Grad-CAM 與 Guided Backprop 實作】(保持不變)
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
# 【4. 主測試腳本 (🚨 核心修正點：任務 3 和 8)】
# ====================================================================

def run_single_test():
    print(f"--- 開始單一檔案 Guided Grad-CAM 測試 ---")
    print(f"使用設備: {DEVICE}")

    # 1. 載入 MNI 模板
    try:
        mni_template = load_mni152_template() 
        MNI_TARGET_SHAPE = mni_template.get_fdata().shape
        MNI_AFFINE = mni_template.affine
        print(f"✅ MNI 模板載入成功，目標尺寸 (Target Shape) 設為: {MNI_TARGET_SHAPE}")
    except Exception as e:
        print(f"🚨 致命錯誤：無法載入 Nilearn MNI 模板。錯誤: {e}"); return

    # 2. 載入模型 (略)
    model = Simple3DCNN_InstanceNorm(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(TEST_MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ 模型載入成功: {TEST_MODEL_PATH}")

    # 3. 載入單一 NIfTI 檔案 (略)
    input_tensor, original_affine = load_nifti_as_tensor(TEST_DATA_PATH)
    input_tensor = input_tensor.to(DEVICE); input_tensor.requires_grad_()
    print(f"✅ 測試資料載入成功: {TEST_DATA_PATH}")

    # --- 任務 1：計算 Grad-CAM (低解析度) ---
    print("... [任務 1] 正在計算 Grad-CAM (低解析度)...")
    # ... (此部分邏輯保持不變，此處省略)
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
    print(f"✅ 低解析度 Grad-CAM 熱圖已產生 (Shape: {grad_cam_np_low_res.shape})")

    # --- 任務 2：計算 Guided Backpropagation (高解析度) ---
    print("... [任務 2] 正在計算 Guided Backprop (高解析度)...")
    # ... (此部分邏輯保持不變，此處省略)
    model.zero_grad(); gbp = GuidedBackprop(model)
    guided_grads_tensor = gbp.attribute(input_tensor, target=target_class_index)
    guided_grads_np = guided_grads_tensor.squeeze(0).squeeze(0).cpu().detach().numpy()
    print(f"✅ 高解析度 Guided Backprop 熱圖已產生 (Shape: {guided_grads_np.shape})")

    # ----------------------------------------------------
    # 【任務 3：結合與空間對齊 (修正閾值)】
    # ----------------------------------------------------
    print("... [任務 3] 正在結合熱圖並執行空間對齊...")

    # 3a. 上取樣 (Zoom) 低解析度的 Grad-CAM
    zoom_factors = [t / c for t, c in zip(MNI_TARGET_SHAPE, grad_cam_np_low_res.shape)]
    grad_cam_high_res = zoom(grad_cam_np_low_res, zoom=zoom_factors, order=1)
    
    # 3b. 建立 Guided Backprop 的 NIfTI 物件 (使用原始 Affine)
    gbp_nii_raw = nib.Nifti1Image(guided_grads_np, original_affine)
    
    # 3c. 將 Guided Backprop 配準 (Resample) 到 MNI 模板空間
    print("... [任務 3b] 正在將 Guided Backprop 配準到 MNI 空間...")
    gbp_nii_aligned = resample_to_img(
        source_img=gbp_nii_raw,
        target_img=mni_template,
        interpolation='continuous'
    )
    guided_grads_aligned_np = gbp_nii_aligned.get_fdata()

    # 3d. 正規化兩張「均在 MNI 空間」的熱圖
    grad_cam_norm = normalize_map(grad_cam_high_res)
    guided_grads_norm = normalize_map(np.abs(guided_grads_aligned_np))

    # 3e. 結合 (核心)：Guided Grad-CAM = Grad-CAM * Guided Backprop
    guided_grad_cam_map = grad_cam_norm * guided_grads_norm
    
    # 3f. 軸向翻轉 (修正鏡像問題)
    heatmap_flipped = np.flip(guided_grad_cam_map, axis=0) 
    
    # 3g. 🚨 修正點：計算統計閾值 (95th percentile)
    # 我們只看非零激活的體素
    nonzero_values = heatmap_flipped[heatmap_flipped > 0]
    if len(nonzero_values) > 0:
        stat_threshold = np.percentile(nonzero_values, 95) # 顯示最強的 5%
    else:
        stat_threshold = heatmap_flipped.max() # 如果全為 0
    print(f"✅ 統計閾值 (95th percentile) 計算完成: {stat_threshold:.4f}")

    # 3h. 強制賦予 MNI Affine
    nii_aligned = nib.Nifti1Image(heatmap_flipped, MNI_AFFINE)
    
    print("✅ 空間對齊完成！")

    # ----------------------------------------------------
    # 【任務 8：儲存結果 (🚨 核心修正點：XYZ 三視圖)】
    # ----------------------------------------------------
    subject_id = os.path.basename(TEST_DATA_PATH).replace('.nii', '').replace('.nii.gz', '')
    nii_path = os.path.join(OUTPUT_DIR, f"{subject_id}_GUIDED_gradcam_ALIGNED.nii.gz")
    png_path = os.path.join(OUTPUT_DIR, f"{subject_id}_GUIDED_gradcam_VISUAL_XYZ.png")
    
    nib.save(nii_aligned, nii_path)
    print(f"💾 已儲存對齊的 NIfTI 檔案至: {nii_path}")
    
    # 創建一個 3x1 的 Matplotlib 畫布
    fig, axes = plt.subplots(nrows=3, figsize=(15, 10))
    fig.suptitle(f"Guided Grad-CAM (Aligned) - {subject_id}", fontsize=16, y=1.02)
    
    # 繪製 Z 視圖 (Axial)
    plotting.plot_stat_map(
        nii_aligned, 
        bg_img=mni_template,    
        display_mode='z',           
        cut_coords=7, # 7 個切片
        cmap='hot',                
        black_bg=True,             
        colorbar=True,
        threshold=stat_threshold, # 應用統計閾值
        axes=axes[0], # 繪製在第一個子圖
        title="Axial (Z)"
    )
    
    # 繪製 X 視圖 (Sagittal)
    plotting.plot_stat_map(
        nii_aligned, 
        bg_img=mni_template,    
        display_mode='x',           
        cut_coords=6, # 6 個切片
        cmap='hot',                
        black_bg=True,             
        colorbar=False, # 關閉多餘的 colorbar
        threshold=stat_threshold,
        axes=axes[1],
        title="Sagittal (X)"
    )
    
    # 繪製 Y 視圖 (Coronal)
    plotting.plot_stat_map(
        nii_aligned, 
        bg_img=mni_template,    
        display_mode='y',           
        cut_coords=6,
        cmap='hot',                
        black_bg=True,             
        colorbar=False,
        threshold=stat_threshold,
        axes=axes[2],
        title="Coronal (Y)"
    )
    
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig) # 關閉畫布
    
    print(f"🖼️  已儲存 XYZ 三視圖 PNG 檔案至: {png_path}")
    print(f"--- 高解析度測試完成 ---")

if __name__ == '__main__':
    run_single_test()