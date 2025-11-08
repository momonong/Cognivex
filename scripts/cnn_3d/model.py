import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
import glob
from captum.attr import IntegratedGradients

# ====================================================================
# 【1. 設定與配置】
# ====================================================================
MODEL_DIR = 'model/cnn_3d/' 
DATA_DIR = "E:/fMRI/Model/sMRI_data" 
OUTPUT_DIR = 'output/cnn_3d/ig_attributions_ensembled/' 

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
        self.block4 = create_conv_block(64, 128)
        
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
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ====================================================================
# 【3. 核心歸因函數 (核心修正點)】
# ====================================================================

def get_ig_attribution(model, input_tensor, target_class_index):
    """
    執行 Integrated Gradients (IG) 歸因。
    """
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor).to(DEVICE)
    
    # 🚨 修正點：
    # 1. n_steps=20 (保持不變)
    # 2. 新增 internal_batch_size=5 (將 20 步拆成 4 批執行)
    #    這將大幅降低 VRAM 峰值，解決 OOM 問題。
    #    如果 5 仍然 OOM (在 20GB VRAM 上極不可能)，請嘗試 2。
    attribution = ig.attribute(input_tensor,
                               baselines=baseline,
                               target=target_class_index,
                               n_steps=20,
                               internal_batch_size=5) # <--- 修正於此
    
    return attribution

def load_nifti_as_tensor(file_path):
    img = nib.load(file_path)
    affine = img.affine
    data_np = img.get_fdata(dtype=np.float32)
    tensor = torch.tensor(data_np).unsqueeze(0).unsqueeze(0) 
    return tensor, affine

# ====================================================================
# 【4. 主執行腳本 (使用遞迴搜尋)】
# ====================================================================

def main():
    print(f"使用設備: {DEVICE}")
    
    # 1. 載入模型路徑
    model_paths = sorted(glob.glob(os.path.join(MODEL_DIR, 'cnn_3d_fold_*.pth')))
    if not model_paths:
        print(f"🚨 錯誤：在 {MODEL_DIR} 中找不到任何 'cnn_3d_fold_*.pth' 模型權重。")
        return
    print(f"找到 {len(model_paths)} 個模型 Fold。")

    # 2. 遞迴搜尋 .nii 和 .nii.gz
    print(f"正在 {DATA_DIR} 中「遞迴」搜尋 .nii 和 .nii.gz 檔案...")
    search_path_nii = os.path.join(DATA_DIR, '**', '*.nii')
    search_path_niigz = os.path.join(DATA_DIR, '**', '*.nii.gz')
    nii_files = glob.glob(search_path_nii, recursive=True)
    niigz_files = glob.glob(search_path_niigz, recursive=True)
    data_files = sorted(nii_files + niigz_files)
    
    if not data_files:
        print(f"🚨 錯誤：在 {DATA_DIR} 及其子目錄中找不到任何 .nii 或 .nii.gz 檔案。")
        return

    print(f"找到 {len(data_files)} 筆 sMRI 資料。開始執行「集成歸因」...")

    # 外層迴圈：遍歷所有病患
    for file_path in data_files:
        
        base_name = os.path.basename(file_path)
        if file_path.endswith('.nii.gz'):
            subject_id = base_name.replace('.nii.gz', '')
        elif file_path.endswith('.nii'):
            subject_id = base_name.replace('.nii', '')
        else:
            subject_id = base_name
            
        print(f"\n-> 處理檔案: {file_path}")
        print(f"   (病患 ID 設為: {subject_id})")
        
        try:
            input_tensor, affine = load_nifti_as_tensor(file_path)
            input_tensor = input_tensor.to(DEVICE)
            input_tensor.requires_grad_()
            
            fold_attributions = []
            target_class_index = 1 # 假設 1 = AD
            
            # 內層迴圈：遍歷 5 個模型
            for i, model_path in enumerate(model_paths):
                print(f"  ... 載入 Fold {i+1}")
                
                model = Simple3DCNN_InstanceNorm(num_classes=2).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                
                # 4. 執行 IG (現在使用 internal_batch_size=5)
                attr_tensor = get_ig_attribution(model, input_tensor, target_class_index)
                fold_attributions.append(attr_tensor)
            
            # 5. 集成 (Ensemble) 歸因圖
            stacked_attributions = torch.stack(fold_attributions, dim=0)
            mean_attribution_tensor = torch.mean(stacked_attributions, dim=0)
            
            # 6. 處理並儲存歸因圖
            attribution_np = mean_attribution_tensor.squeeze(0).squeeze(0).cpu().detach().numpy()
            nii_img_out = nib.NiftiImage(attribution_np, affine)
            
            output_path = os.path.join(OUTPUT_DIR, f'{subject_id}_ig_ensembled.nii.gz')
            nib.save(nii_img_out, output_path)
            
            print(f"  ✅ 已成功儲存「集成歸因圖」至: {output_path}")

        except Exception as e:
            print(f"  ❌ 處理 {subject_id} 時發生錯誤: {e}")
            if 'out of memory' in str(e):
                print("--- 偵測到 CUDA Out of Memory ---")
                print("請嘗試在 get_ig_attribution 函數中進一步降低 internal_batch_size (例如 2 或 1)。")
                break 

    print("\n--- 集成歸因流程結束 ---")

if __name__ == '__main__':
    main()