import os
import sys
import numpy as np
import argparse
import nibabel as nib
from tqdm import tqdm

# --- 警告過濾器 ---
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader

# 匯入 MONAI
from monai.data import Dataset # (使用 Dataset 而非 CacheDataset, 因為我們只處理一個檔案)
from monai.visualize import GradCAM
from monai.utils import set_determinism

# --- (*** 關鍵 ***) ---
# 1. 將專案根目錄加入路徑
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. 從 app/core/ 匯入我們的定義
from app.core.cnn_3d.model_def import Simple3DCNN_InstanceNorm
from app.core.cnn_3d.transforms_def import test_transforms, PATCH_SIZE
# --- (*** 結束 ***) ---


# --- 1. 配置 ---

set_determinism(seed=42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5

# (與 predict.py 相同)
MODEL_WEIGHTS_PATHS = [
    f"trained_models/cnn_3d/cognivex_instancenorm_cnn_best_fold_{i + 1}.pth" for i in range(K_FOLDS)
]

# (我們的新輸出目錄)
OUTPUT_DIR = "output/cnn_3d/xai_heatmaps/"


# --- 2. 核心函式 ---

def load_models():
    """載入 5 個集成模型"""
    models = []
    print(f"--- 正在載入 {K_FOLDS}-Fold 集成模型 ---")
    for i, path in enumerate(MODEL_WEIGHTS_PATHS):
        print(f"  正在載入 Fold {i+1} 權重: {path} ...")
        try:
            model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            models.append(model)
        except FileNotFoundError:
            print(f"  錯誤: 找不到權重檔案 '{path}'。")
            return None
    print(f"✅ 成功載入 {len(models)} 個模型。\n")
    return models

def load_single_subject_data(nifti_path):
    """使用 MONAI transforms 載入單一 NIfTI 檔案"""
    
    # 建立一個只包含一個檔案的 data dictionary
    # "label" 欄位在這裡是假的 (0), 因為我們只是要載入影像
    data_dict = [{"image": nifti_path, "label": 0}] 
    
    try:
        # Dataset (非 CacheDataset) 適合單一檔案
        ds = Dataset(data=data_dict, transform=test_transforms)
        # DataLoader batch_size=1
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        
        # 載入資料
        batch_data = next(iter(loader))
        
        # 提取影像張量和關鍵的 'affine' 矩陣
        inputs = batch_data["image"].to(DEVICE)
        
        # (*** 關鍵 ***)
        # MONAI 的 transforms (例如 CenterSpatialCropd) 會更新 affine
        # 我們必須使用這個「更新後」的 affine 來儲存我們的熱圖
        # [0] 是因為 batch size 是 1
        output_affine = batch_data["image_meta_dict"]["affine"][0].numpy()
        
        print(f"✅ 成功載入並處理: {os.path.basename(nifti_path)}")
        print(f"  輸入 Shape: {inputs.shape}")
        
        return inputs, output_affine
        
    except Exception as e:
        print(f"❌ 錯誤: 載入或處理檔案 {nifti_path} 失敗: {e}")
        return None, None

def save_nifti_heatmap(heatmap_3d, affine, subject_id, output_dir):
    """將 3D numpy 陣列儲存為 NIfTI 檔案"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用 nibabel 建立 NIfTI 影像物件
    nii_image = nib.Nifti1Image(heatmap_3d, affine)
    
    # 定義輸出路徑
    output_filename = os.path.join(output_dir, f"{subject_id}_ensemble_gradcam.nii.gz")
    
    # 儲存
    nib.save(nii_image, output_filename)
    print(f"\n✅ 成功！集成 XAI 熱圖已儲存至:\n  {output_filename}")


def main(args):
    """
    執行集成 Grad-CAM 的主函式
    """
    
    # 1. 檢查輸入
    if not os.path.exists(args.nifti_path):
        print(f"❌ 錯誤: 找不到輸入的 NIfTI 檔案: {args.nifti_path}")
        return
        
    target_class_idx = 1 if args.target_class.upper() == "AD" else 0
    print(f"--- 執行 3D 集成 Grad-CAM ---")
    print(f"  目標檔案: {args.nifti_path}")
    print(f"  目標類別: {args.target_class.upper()} (Class Index: {target_class_idx})")
    print(f"----------------------------------")
    
    # 2. 載入模型
    models = load_models()
    if models is None:
        return

    # 3. 載入單一病患資料 (影像 + Affine)
    inputs, output_affine = load_single_subject_data(args.nifti_path)
    if inputs is None:
        return

    # 4. (*** 核心邏輯: 集成 Grad-CAM ***)
    print("\n--- 正在計算 5-Fold 集成熱圖 ---")
    
    all_heatmaps = []
    
    for i, model in enumerate(tqdm(models, desc="計算各 Fold 的 Grad-CAM")):
        
        # (*** 關鍵 ***)
        # 我們將目標層(target_layers)設定為 'model.block4'
        # 這是你模型中的最後一個卷積區塊
        
        # 註: MONAI GradCAM 會自動尋找該區塊的第一個 Conv3d 層
        # 為了更精確, 我們指向 block4 的第一個 Conv3d 層
        target_layer = model.block4[0] 
        
        # 建立 GradCAM
        gradcam = GradCAM(
            model=model,
            target_layers=target_layer,
            device=DEVICE
        )
        
        # 計算熱圖 (結果是 (1, 1, 128, 128, 128))
        heatmap_tensor = gradcam(
            x=inputs,
            class_idx=target_class_idx,
        )
        
        # 移除 Batch 和 Channel 維度 (1, 1, 128...) -> (128...)
        # 並轉換為 numpy
        heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
        all_heatmaps.append(heatmap_np)

    # 5. 平均熱圖
    # 將 5 個 (128, 128, 128) 堆疊成 (5, 128, 128, 128)
    heatmap_stack = np.stack(all_heatmaps, axis=0)
    
    # 沿著 Fold 維度 (axis=0) 取平均
    mean_heatmap = np.mean(heatmap_stack, axis=0)
    
    print(f"✅ 成功計算 5-Fold 平均熱圖 (Shape: {mean_heatmap.shape})")

    # 6. 儲存 NIfTI
    subject_id = os.path.basename(args.nifti_path).split('.')[0] # 從 "AD_001.nii.gz" 取得 "AD_001"
    
    save_nifti_heatmap(
        heatmap_3d=mean_heatmap,
        affine=output_affine,
        subject_id=subject_id,
        output_dir=OUTPUT_DIR
    )


# --- 7. 允許此腳本從命令列執行 ---

if __name__ == "__main__":
    
    # (*** 關鍵 ***)
    # 我們使用 argparse 讓你可以從終端機「傳入」檔案路徑
    #
    # 範例用法:
    # python scripts/cnn_3d/03_generate_xai.py -i "E:\fMRI\Model\sMRI_data\AD\T1_3D_MPRAGE_SAG_0003_008\T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii" -c AD
    #
    
    parser = argparse.ArgumentParser(description="執行 3D 集成 Grad-CAM")
    
    parser.add_argument(
        "-i", 
        "--nifti_path", 
        type=str, 
        required=True, 
        help="要分析的單一 NIfTI 檔案的完整路徑。"
    )
    
    parser.add_argument(
        "-c", 
        "--target_class", 
        type=str, 
        required=True, 
        choices=["AD", "NC"],
        help="要生成熱圖的目標類別 (AD 或 NC)。"
    )

    args = parser.parse_args()
    main(args)