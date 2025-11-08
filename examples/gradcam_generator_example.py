"""
範例: 使用 GradCAMGenerator 生成 Grad-CAM 熱圖

此範例展示如何使用重構後的 GradCAMGenerator 類別來:
1. 載入集成模型
2. 生成單一模型的 Grad-CAM
3. 生成集成 Grad-CAM
4. 儲存為 NIfTI 格式
"""

import os
import sys
import torch
import numpy as np

# 加入專案根目錄到路徑
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from app.core.xai import GradCAMGenerator
from app.core.cnn_3d.model import Simple3DCNN_InstanceNorm


def load_ensemble_models(weights_dir: str, num_folds: int = 5, device: torch.device = None):
    """
    載入集成模型
    
    Args:
        weights_dir: 模型權重目錄
        num_folds: fold 數量
        device: 計算裝置
        
    Returns:
        模型列表
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    models = []
    for i in range(num_folds):
        weight_path = os.path.join(weights_dir, f"cnn_3d_fold_{i + 1}.pth")
        
        if not os.path.exists(weight_path):
            print(f"⚠️ 警告: 找不到權重檔案 {weight_path}")
            continue
        
        model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
        print(f"✅ 載入模型 fold {i + 1}")
    
    return models


def main():
    """主函式"""
    
    # 設定
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "model/cnn_3d")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/cnn_3d/gradcam_examples")
    
    print("=" * 60)
    print("GradCAMGenerator 使用範例")
    print("=" * 60)
    print(f"裝置: {DEVICE}")
    print(f"權重目錄: {WEIGHTS_DIR}")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print()
    
    # 1. 載入模型
    print("步驟 1: 載入集成模型...")
    models = load_ensemble_models(WEIGHTS_DIR, num_folds=5, device=DEVICE)
    
    if not models:
        print("❌ 錯誤: 無法載入任何模型")
        return
    
    print(f"✅ 成功載入 {len(models)} 個模型\n")
    
    # 2. 建立 GradCAMGenerator
    print("步驟 2: 建立 GradCAMGenerator...")
    generator = GradCAMGenerator(
        models=models,
        device=DEVICE,
        target_layer_name="block4"
    )
    print("✅ GradCAMGenerator 已建立\n")
    
    # 3. 建立模擬輸入 (實際使用時應該從 NIfTI 載入)
    print("步驟 3: 建立模擬輸入張量...")
    # 模擬一個 128x128x128 的 3D 影像
    mock_input = torch.randn(1, 1, 128, 128, 128).to(DEVICE)
    print(f"✅ 輸入張量形狀: {mock_input.shape}\n")
    
    # 4. 生成單一模型的 Grad-CAM
    print("步驟 4: 生成單一模型的 Grad-CAM...")
    single_heatmap = generator.generate_single_model(
        model=models[0],
        input_tensor=mock_input,
        target_class=1  # AD
    )
    print(f"✅ 單一模型熱圖形狀: {single_heatmap.shape}")
    print(f"   數值範圍: [{single_heatmap.min():.4f}, {single_heatmap.max():.4f}]\n")
    
    # 5. 生成集成 Grad-CAM
    print("步驟 5: 生成集成 Grad-CAM...")
    ensemble_heatmap = generator.generate_ensemble(
        input_tensor=mock_input,
        target_class=1,  # AD
        threshold_percentile=95.0,
        aggregation_method="mean"
    )
    print(f"✅ 集成熱圖形狀: {ensemble_heatmap.shape}")
    print(f"   數值範圍: [{ensemble_heatmap.min():.4f}, {ensemble_heatmap.max():.4f}]")
    
    # 6. 取得統計資訊
    stats = generator.get_statistics(ensemble_heatmap)
    print(f"\n統計資訊:")
    print(f"  - 非零體素數量: {stats['non_zero_count']}")
    print(f"  - 非零體素百分比: {stats['non_zero_percentage']:.2f}%")
    print(f"  - 非零區域平均值: {stats['non_zero_mean']:.4f}")
    print()
    
    # 7. 儲存為 NIfTI
    print("步驟 6: 儲存為 NIfTI 格式...")
    # 建立模擬的 affine 矩陣 (實際使用時應該從原始 NIfTI 取得)
    mock_affine = np.eye(4)
    mock_affine[:3, :3] *= 1.0  # 1mm 解析度
    
    output_path = generator.save_as_nifti(
        heatmap=ensemble_heatmap,
        affine=mock_affine,
        output_path=OUTPUT_DIR,
        subject_id="example_subject",
        target_class="AD"
    )
    print(f"✅ 熱圖已儲存至: {output_path}\n")
    
    # 8. 測試上採樣功能
    print("步驟 7: 測試上採樣功能...")
    upsampled_heatmap = generator.upsample_to_original(
        heatmap=ensemble_heatmap,
        target_shape=(256, 256, 256),
        order=1
    )
    print(f"✅ 上採樣後形狀: {upsampled_heatmap.shape}\n")
    
    print("=" * 60)
    print("✅ 範例執行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
