"""
腦區分析腳本：使用你成功的 binary classification model + Grad-CAM
找出模型關注的重要腦區

策略：
1. 使用你之前成功的 NC vs AD binary model
2. 用 Grad-CAM 生成熱圖
3. 與 AAL atlas 對齊，找出重要腦區
4. 量化分析每個腦區的重要性
"""

import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
from nilearn import datasets, image as nimg
from monai.visualize import GradCAM

# 使用你成功的模型架構
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


def analyze_important_regions():
    """
    分析模型關注的重要腦區
    """
    
    print("="*80)
    print("腦區重要性分析")
    print("="*80)
    
    # 配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "model/cnn_3d/cnn_3d_fold_1.pth"  # 你的 binary model
    OUTPUT_DIR = "output/multiclass/brain_region_analysis/"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n使用裝置: {DEVICE}")
    print(f"模型路徑: {MODEL_PATH}")
    
    # 檢查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ 錯誤：找不到模型檔案 {MODEL_PATH}")
        print("\n建議：")
        print("1. 先訓練 binary classification model (NC vs AD)")
        print("2. 或使用你之前訓練好的模型")
        return
    
    # 載入模型
    print("\n載入模型...")
    model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ 模型載入成功")
    
    # 載入 AAL atlas
    print("\n載入 AAL atlas...")
    aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
    aal_img = nimg.load_img(aal_atlas.maps)
    aal_data = aal_img.get_fdata()
    aal_labels = aal_atlas.labels
    print(f"✅ AAL atlas 載入成功 ({len(aal_labels)} 個腦區)")
    
    # 建立 Grad-CAM
    print("\n建立 Grad-CAM...")
    target_layer = model.block4[0]
    gradcam = GradCAM(nn_module=model, target_layers=target_layer, device=DEVICE)
    print("✅ Grad-CAM 準備完成")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print("\n下一步：")
    print("1. 準備一些 AD 病患的影像")
    print("2. 使用 Grad-CAM 生成熱圖")
    print("3. 與 AAL atlas 對齊")
    print("4. 計算每個腦區的激活強度")
    print("5. 排序找出最重要的腦區")
    print(f"\n結果將儲存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    analyze_important_regions()
