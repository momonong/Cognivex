# app/core/cnn_3d/xai.py (批次處理版 V14 - "nn_module" Fix)

import os
import sys
import numpy as np
import nibabel as nib
from tqdm import tqdm
import glob
import scipy.ndimage

# --- 警告過濾器 ---
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='monai.transforms.spatial.array')

import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv

# --- 匯入 MONAI ---
from monai.visualize import GradCAM
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    Spacing,
    Orientation,
    ScaleIntensityRange,
    CenterSpatialCrop,
    EnsureType
)

# --- 導入 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    print(f"將專案根目錄加入路徑: {PROJECT_ROOT}")
    sys.path.append(PROJECT_ROOT)

print("正在從 app.core.cnn_3d.model 導入定義...")
try:
    from app.core.cnn_3d.model import Simple3DCNN_InstanceNorm
except ImportError as e:
    try:
        from app.core.cnn_3d.model_def import Simple3DCNN_InstanceNorm
    except ImportError:
        print(f"❌ 錯誤: 導入 app.core.cnn_3d.model 模組失敗: {e}")
        sys.exit(1)


# --- 1. 配置 ---

# 設定隨機種子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5

# 載入 .env 檔案
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if not os.path.exists(dotenv_path):
    print(f"⚠️ 警告: 找不到 .env 檔案於: {dotenv_path}")
else:
    print(f"✅ 成功載入 .env 檔案: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)

# 權重和輸出路徑 (使用你最新的路徑)
MODEL_WEIGHTS_PATHS = [
    os.path.join(PROJECT_ROOT, f"model/cnn_3d/cnn_3d_fold_{i + 1}.pth") 
    for i in range(K_FOLDS)
]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/cnn_3d/xai_heatmaps/")

# 常數
PATCH_SIZE = (128, 128, 128)
TARGET_VOXEL_SIZE = (1.0, 1.0, 1.0)
A_MIN = 0.0
A_MAX = 1000.0
THRESHOLD_PERCENTILE = 95.0 # (你可以把它改回 99.0)


# --- 3. 核心函式 ---

def load_models():
    """載入 5 個集成模型 (不變)"""
    models = []
    print(f"--- 正在載入 {K_FOLDS}-Fold 集成模型 ---")
    for i, path in enumerate(MODEL_WEIGHTS_PATHS):
        if not os.path.exists(path):
            print(f"  ❌ 錯誤: 找不到權重檔案 '{path}'。")
            return None
        model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        models.append(model)
    print(f"✅ 成功載入 {len(models)} 個模型。\n")
    return models

def find_nii_files(input_dir):
    """遞迴掃描資料夾以尋找 NIfTI 檔案 (不變)"""
    print(f"--- 正在 {input_dir} 中遞迴掃描 .nii 和 .nii.gz 檔案 ---")
    files = glob.glob(os.path.join(input_dir, "**", "*.nii"), recursive=True)
    files.extend(glob.glob(os.path.join(input_dir, "**", "*.nii.gz"), recursive=True))
    
    if not files:
        print(f"⚠️ 警告: 在 {input_dir} 及其子資料夾中找不到 .nii 或 .nii.gz 檔案。")
    else:
        print(f"✅ 找到 {len(files)} 個 NIfTI 檔案。")
    return files

def update_affine_after_zoom(affine, zoom_factors):
    """手動計算 Scipy zoom 之後的新 Affine (不變)"""
    new_affine = np.copy(affine)
    np.fill_diagonal(new_affine, new_affine.diagonal() * np.append(zoom_factors, 1))
    return new_affine

def update_affine_after_crop(affine, crop_start_indices):
    """手動計算 Numpy crop 之後的新 Affine (不變)"""
    new_affine = np.copy(affine)
    new_affine[:3, 3] = new_affine[:3, 3] + new_affine[:3, :3] @ crop_start_indices
    return new_affine


def load_single_subject_data_manual(nifti_path):
    """ (*** V10 版 - 已證明有效 ***)
    完全手動的 Nibabel / Scipy / Numpy 處理管線
    """
    try:
        # 1. 載入 (Nibabel)
        img = nib.load(nifti_path)
        
        # 2. Re-orient to RAS (Nibabel)
        img_ras = nib.as_closest_canonical(img)
        data_ras = img_ras.get_fdata()
        affine_ras = img_ras.affine
        
        if affine_ras is None:
            raise ValueError("檔案的 Affine 矩陣為 None (空)")

        # 3. Resample (Scipy)
        current_voxel_size = img_ras.header.get_zooms()[:3]
        zoom_factors = [c / t for c, t in zip(current_voxel_size, TARGET_VOXEL_SIZE)]
        data_resampled = scipy.ndimage.zoom(data_ras, zoom_factors, order=1)
        affine_resampled = update_affine_after_zoom(affine_ras, zoom_factors)

        # 4. Scale (Numpy)
        data_scaled = (data_resampled - A_MIN) / (A_MAX - A_MIN + 1e-6)
        data_scaled = np.clip(data_scaled, 0.0, 1.0)
        
        # 5. Crop (Numpy)
        (h, w, d) = data_scaled.shape
        (ch, cw, cd) = PATCH_SIZE
        h_start = (h // 2) - (ch // 2)
        w_start = (w // 2) - (cw // 2)
        d_start = (d // 2) - (cd // 2)
        data_cropped = data_scaled[
            h_start : h_start + ch,
            w_start : w_start + cw,
            d_start : d_start + cd
        ]
        affine_cropped = update_affine_after_crop(affine_resampled, [h_start, w_start, d_start])
        
        # 6. Channel & Tensor (Numpy / Torch)
        data_final_np = np.expand_dims(data_cropped, axis=0) # (1, H, W, D)
        final_tensor = torch.from_numpy(data_final_np.copy()).float().to(DEVICE)
        
        return final_tensor.unsqueeze(0), affine_cropped
        
    except Exception as e:
        print(f"❌ 錯誤: 載入或處理檔案 {os.path.basename(nifti_path)} 失敗: {e}")
        return None, None

def save_nifti_heatmap(heatmap_3d, affine, subject_id, output_dir, target_class):
    """將 3D numpy 陣列儲存為 NIfTI 檔案 (不變)"""
    os.makedirs(output_dir, exist_ok=True)
    nii_image = nib.Nifti1Image(heatmap_3d.astype(np.float32), affine) # 確保是 float32
    output_filename = os.path.join(output_dir, f"{subject_id}_gradcam_ensemble_{target_class.upper()}.nii.gz")
    nib.save(nii_image, output_filename)


def main():
    """
    (批次處理版 V14)
    執行集成 Grad-CAM 的主函式
    """
    
    input_dir = os.getenv("XAI_INPUT_DIR")
    target_class = os.getenv("XAI_TARGET_CLASS")

    if not input_dir or not target_class or not os.path.isdir(input_dir) or target_class.upper() not in ["AD", "NC"]:
        print("❌ 錯誤: .env 配置不正確或找不到路徑。請檢查 .env 檔案。")
        return

    target_class_idx = 1 if target_class.upper() == "AD" else 0
    print(f"--- 執行 3D 集成 Grad-CAM (批次處理模式 V14 - 'nn_module' Fix) ---")
    print(f"  專案根目錄: {PROJECT_ROOT}")
    print(f"  目標資料夾: {input_dir}")
    print(f"  目標類別: {target_class.upper()} (Class Index: {target_class_idx})")
    print(f"  (*** 視覺化: 將保留 Top {100-THRESHOLD_PERCENTILE:.0f}% 的訊號 ***)")
    print(f"----------------------------------")
    
    models = load_models()
    if models is None:
        return

    subject_file_list = find_nii_files(input_dir)
    if not subject_file_list:
        return

    print(f"\n--- 開始批次處理 {len(subject_file_list)} 個檔案 ---")
    
    successful_runs = 0
    failed_runs = 0
    main_pbar = tqdm(subject_file_list, desc="批次 XAI 處理中", unit="file")
    
    for nifti_path in main_pbar:
        
        subject_id = os.path.basename(nifti_path).split('.')[0]
        main_pbar.set_postfix({"file": f"{subject_id}.nii..."})

        inputs, output_affine = load_single_subject_data_manual(nifti_path)
        
        if inputs is None or output_affine is None:
            print(f"  跳過檔案: {subject_id}")
            failed_runs += 1
            continue

        try:
            all_heatmaps = []
            for model in models:
                target_layer = model.block4[0] 
                
                # --- (*** 關鍵的 V14 修正 ***) ---
                # 參數名稱是 'nn_module'
                gradcam = GradCAM(
                    nn_module=model,  # <--- 修正!
                    target_layers=target_layer, 
                    device=DEVICE
                )
                # --- (*** V14 修正結束 ***) ---
                
                heatmap_tensor = gradcam(x=inputs, class_idx=target_class_idx)
                heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
                all_heatmaps.append(heatmap_np)

            # 1. 平均熱圖
            heatmap_stack = np.stack(all_heatmaps, axis=0)
            mean_heatmap = np.mean(heatmap_stack, axis=0)
            
            # 2. 標準化 (0-1)
            norm_map = (mean_heatmap - np.min(mean_heatmap)) / (np.max(mean_heatmap) - np.min(mean_heatmap) + 1e-8)
            
            # 3. 取得閾值 (例如: 95.0)
            threshold = np.percentile(norm_map, THRESHOLD_PERCENTILE)
            
            # 4. 遮罩 (Masking)
            masked_map = np.where(norm_map >= threshold, norm_map, 0)
            
            # 5. 儲存「處理過」的熱圖
            save_nifti_heatmap(
                heatmap_3d=masked_map,
                affine=output_affine,
                subject_id=subject_id,
                output_dir=OUTPUT_DIR,
                target_class=target_class
            )
            successful_runs += 1
            
        except Exception as e:
            print(f"❌ 錯誤: 處理 {subject_id} 時 GradCAM 或儲存失敗: {e}")
            failed_runs += 1
            continue
    
    print("\n==================================================")
    print(f"✅ 批次處理完成！")
    print(f"  成功處理: {successful_runs} / {len(subject_file_list)} 個檔案")
    print(f"  失敗/跳過: {failed_runs} / {len(subject_file_list)} 個檔案")
    print(f"  所有成功熱圖皆已儲存至: {OUTPUT_DIR}")
    print("==================================================")


if __name__ == "__main__":
    main()