import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
from scipy.ndimage import zoom
from nilearn import plotting
# 使用別名確保 MNI 模板載入函式範圍正確
from nilearn.datasets import load_mni152_template as nilearn_load_mni_template 

# ====================================================================
# 【1. 設定與配置 (專注於單一檔案)】
# ====================================================================
# 🚨 請確認以下兩個檔案路徑是正確的
TEST_DATA_PATH = "E:/fMRI/Model/sMRI_data/AD/T1_3D_MPRAGE_SAG_0003_008/T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii"
TEST_MODEL_PATH = "model/cnn_3d/cnn_3d_fold_1.pth" # 我們只測試 Fold 1

OUTPUT_DIR = "output/cnn_3d/gradcam_test/" # 儲存測試結果的目錄
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ====================================================================
# 【2. 模型定義 (您提供的真實模型)】
# ====================================================================
class Simple3DCNN_InstanceNorm(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(Simple3DCNN_InstanceNorm, self).__init__()
        
        def create_conv_block(in_c, out_c, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=padding),
                nn.InstanceNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2)
            )
        
        self.block1 = create_conv_block(in_channels, 16)
        self.block2 = create_conv_block(16, 32)
        self.block3 = create_conv_block(32, 64)
        self.block4 = create_conv_block(64, 128) # <--- 我們將 Hook 這一層
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x) # <--- Hook 點
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ====================================================================
# 【3. Grad-CAM 實作】
# ====================================================================

# 儲存梯度和激活的 Hook 函數
gradients = {}
activations = {}

def backward_hook(module, grad_in, grad_out):
    gradients['value'] = grad_out[0]

def forward_hook(module, input, output):
    activations['value'] = output

def load_nifti_as_tensor(file_path):
    img = nib.load(file_path)
    affine = img.affine
    data_np = img.get_fdata(dtype=np.float32)
    # 確保數據是 5D: (Batch=1, Channel=1, D, H, W)
    tensor = torch.tensor(data_np).unsqueeze(0).unsqueeze(0) 
    return tensor, affine, img.shape # 返回原始 shape 以供驗證

# ====================================================================
# 【4. 主測試腳本】
# ====================================================================

def run_single_test():
    print(f"--- 開始單一檔案 Grad-CAM 測試 ---")
    print(f"使用設備: {DEVICE}")

    # 1. 載入 MNI 模板 (這是我們的黃金標準)
    try:
        mni_template = nilearn_load_mni_template()
        MNI_TARGET_SHAPE = mni_template.get_fdata().shape
        MNI_AFFINE = mni_template.affine
        print(f"✅ MNI 模板載入成功，目標尺寸 (Target Shape) 設為: {MNI_TARGET_SHAPE}")
    except Exception as e:
        print(f"🚨 致命錯誤：無法載入 Nilearn MNI 模板。錯誤: {e}")
        return

    # 2. 載入模型
    try:
        model = Simple3DCNN_InstanceNorm(num_classes=2).to(DEVICE)
        model.load_state_dict(torch.load(TEST_MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"✅ 模型載入成功: {TEST_MODEL_PATH}")
    except Exception as e:
        print(f"🚨 致命錯誤：載入模型失敗。請確保模型定義 100% 正確。錯誤: {e}")
        return

    # 3. 載入單一 NIfTI 檔案
    try:
        input_tensor, original_affine, original_shape = load_nifti_as_tensor(TEST_DATA_PATH)
        input_tensor = input_tensor.to(DEVICE)
        print(f"✅ 測試資料載入成功: {TEST_DATA_PATH} (Shape: {original_shape})")
    except Exception as e:
        print(f"🚨 致命錯誤：載入 NIfTI 檔案失敗。錯誤: {e}")
        return

    # 4. 註冊 Hooks
    # 我們 Hook 最後一個卷積層 (block4)
    model.block4.register_forward_hook(forward_hook)
    model.block4.register_backward_hook(backward_hook)
    
    # 5. 執行正向與反向傳播 (Grad-CAM 核心)
    print("... 正在執行正向與反向傳播以取得梯度...")
    output = model(input_tensor)
    
    # 假設我們要解釋「為什麼判斷為 AD」(Class 1)
    target_class_index = 1 
    one_hot_output = torch.zeros((1, output.size()[-1]), device=DEVICE)
    one_hot_output[0][target_class_index] = 1
    
    model.zero_grad()
    output.backward(gradient=one_hot_output, retain_graph=True) # 觸發 backward_hook
    
    print("✅ 梯度與激活已取得。")

    # 6. 計算 Grad-CAM
    # 取得梯度和激活
    guided_gradients = gradients['value'][0] # (Batch, C, D, H, W) -> (C, D, H, W)
    feature_maps = activations['value'][0]   # (C, D, H, W)
    
    # 計算權重 (Global Average Pooling 梯度)
    weights = torch.mean(guided_gradients, dim=[1, 2, 3]) # (C)
    
    # 產生熱圖
    grad_cam_map = torch.zeros(feature_maps.shape[1:], device=DEVICE) # (D, H, W)
    for i in range(feature_maps.shape[0]):
        grad_cam_map += weights[i] * feature_maps[i, :, :, :]
        
    # ReLU
    grad_cam_map = torch.relu(grad_cam_map)
    
    # 轉換為 NumPy
    heatmap_low_res = grad_cam_map.cpu().detach().numpy() # (D, H, W) (低解析度, 8x8x8)
    print(f"✅ 低解析度 Grad-CAM 熱圖已產生 (Shape: {heatmap_low_res.shape})")

    # 7. 空間對齊 (我們學到的所有教訓)
    print("... 正在執行空間對齊 (Zoom, Flip, Affine Assign)...")
    
    # 7a. 上取樣 (Zoom)
    zoom_factors = [t / c for t, c in zip(MNI_TARGET_SHAPE, heatmap_low_res.shape)]
    heatmap_high_res = zoom(heatmap_low_res, zoom=zoom_factors, order=1) # 線性插值
    
    # 7b. 正規化
    min_val, max_val = heatmap_high_res.min(), heatmap_high_res.max()
    heatmap_norm = (heatmap_high_res - min_val) / (max_val - min_val + 1e-8)
    
    # 7c. 軸向翻轉 (修正我們之前看到的鏡像問題)
    # 假設翻轉 X 軸 (axis=0)
    heatmap_flipped = np.flip(heatmap_norm, axis=0) 
    
    # 7d. 強制賦予 MNI Affine
    nii_aligned = nib.Nifti1Image(heatmap_flipped, MNI_AFFINE)
    
    print("✅ 空間對齊完成！")

    # 8. 儲存結果
    subject_id = os.path.basename(TEST_DATA_PATH).replace('.nii', '').replace('.nii.gz', '')
    nii_path = os.path.join(OUTPUT_DIR, f"{subject_id}_gradcam_ALIGNED.nii.gz")
    png_path = os.path.join(OUTPUT_DIR, f"{subject_id}_gradcam_VISUAL.png")
    
    nib.save(nii_aligned, nii_path)
    print(f"💾 已儲存對齊的 NIfTI 檔案至: {nii_path}")
    
    plotting.plot_stat_map(
        nii_aligned, 
        bg_img=mni_template,    
        title=f"Grad-CAM (Aligned) - {subject_id}",
        display_mode='z',           
        cut_coords=8,              
        cmap='hot',                
        black_bg=True,             
        colorbar=True,
        output_file=png_path,
    )
    print(f"🖼️  已儲存視覺化 PNG 檔案至: {png_path}")
    print(f"--- 測試完成 ---")

if __name__ == '__main__':
    run_single_test()