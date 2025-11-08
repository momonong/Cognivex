# scripts/cnn_3d/03_generate_activations.py
#
# 流程 1: 執行推論並儲存「中間產物」 (AD 和 NC)
# (100% No-MONAI-Transforms)

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

import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv

# --- 導入 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
        print("  請確認 'app/core/cnn_3d/' 資料夾中有 'model.py' 或 'model_def.py'")
        sys.exit(1)


# --- 1. 配置 ---
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
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
OUTPUT_DIR_ACTIVATIONS = os.path.join(PROJECT_ROOT, "output/cnn_3d/activations/")

# 常數
PATCH_SIZE = (128, 128, 128)
TARGET_VOXEL_SIZE = (1.0, 1.0, 1.0)
A_MIN = 0.0
A_MAX = 1000.0


# --- (V15 邏輯: 純 PyTorch 掛鉤) ---
class FeatureExtractor:
    """
    一個純 PyTorch 掛鉤 (hook), 用來「攔截」特徵圖
    """
    def __init__(self, net: torch.nn.Module, target_layer: torch.nn.Module):
        self.net = net
        self.target_layer = target_layer
        self.features = None
        self.hook_handle = self.target_layer.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        self.features = output.detach()
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            _ = self.net(x) # 執行前向, 觸發 save_features
        if self.features is None:
            raise RuntimeError("錯誤: PyTorch 掛鉤未能捕捉特徵圖。")
        return self.features
        
    def remove_hook(self):
        self.hook_handle.remove()

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
        img = nib.load(nifti_path)
        img_ras = nib.as_closest_canonical(img)
        data_ras = img_ras.get_fdata()
        affine_ras = img_ras.affine
        
        if affine_ras is None:
            raise ValueError("檔案的 Affine 矩陣為 None (空)")

        current_voxel_size = img_ras.header.get_zooms()[:3]
        zoom_factors = [c / t for c, t in zip(current_voxel_size, TARGET_VOXEL_SIZE)]
        data_resampled = scipy.ndimage.zoom(data_ras, zoom_factors, order=1)
        affine_resampled = update_affine_after_zoom(affine_ras, zoom_factors)

        data_scaled = (data_resampled - A_MIN) / (A_MAX - A_MIN + 1e-6)
        data_scaled = np.clip(data_scaled, 0.0, 1.0)
        
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
        
        data_final_np = np.expand_dims(data_cropped, axis=0)
        final_tensor = torch.from_numpy(data_final_np.copy()).float().to(DEVICE)
        
        return final_tensor.unsqueeze(0), affine_cropped
        
    except Exception as e:
        print(f"❌ 錯誤: 載入或處理檔案 {os.path.basename(nifti_path)} 失敗: {e}")
        return None, None

def save_intermediate_results(
    activation_tensor: torch.Tensor, 
    affine_matrix: np.ndarray, 
    subject_id: str, 
    output_dir_activations: str, 
    target_class: str
):
    """
    儲存「中間產物」:
    1. 激活值 (Activation) 張量 (.pt)
    2. Affine 矩陣 (.npy)
    """
    os.makedirs(output_dir_activations, exist_ok=True)
    
    # (*** 關鍵: 檔名現在包含 class ***)
    act_path = os.path.join(output_dir_activations, f"{subject_id}_{target_class}_activation.pt")
    aff_path = os.path.join(output_dir_activations, f"{subject_id}_{target_class}_affine.npy")
    
    torch.save(activation_tensor.cpu(), act_path)
    np.save(aff_path, affine_matrix)


def main():
    """
    (流程 1: 腳本 A - V19)
    自動推論 AD 和 NC, 並儲存 Activation 和 Affine
    """
    
    # 1. (*** 關鍵: 讀取 SMR_DATA_DIR ***)
    base_data_dir = os.getenv("SMR_DATA_DIR")

    if not base_data_dir or not os.path.isdir(base_data_dir):
        print("="*50)
        print("❌ 錯誤: .env 檔案中缺少或找不到 SMR_DATA_DIR。")
        print("  請確保 .env 檔案 (位於專案根目錄) 包含以下這行:")
        print("  SMR_DATA_DIR=\"E:/fMRI/Model/sMRI_data\"")
        print("="*50)
        return

    print(f"--- 流程 1: 產生 Activation (V19 - AD+NC 批次) ---")
    print(f"  專案根目錄: {PROJECT_ROOT}")
    print(f"  基礎資料夾: {base_data_dir}")
    print(f"  輸出資料夾: {OUTPUT_DIR_ACTIVATIONS}")
    print(f"----------------------------------")
    
    models = load_models()
    if models is None:
        return

    # (*** 關鍵: 這是你的「自動化」邏輯 ***)
    jobs_to_run = [
        ("AD", 1), # (資料夾名稱, 類別索引)
        ("NC", 0)
    ]
    
    total_successful = 0
    total_failed = 0

    # --- 2. (*** 關鍵: 外層迴圈 (AD, NC) ***) ---
    for class_name, class_idx in jobs_to_run:
        
        print(f"\n==================================================")
        print(f"--- 正在處理類別: {class_name} ---")
        
        input_dir = os.path.join(base_data_dir, class_name)
        if not os.path.isdir(input_dir):
            print(f"⚠️ 警告: 找不到資料夾 {input_dir}, 跳過...")
            continue
            
        subject_file_list = find_nii_files(input_dir)
        if not subject_file_list:
            print(f"  在 {input_dir} 中找不到檔案, 跳過...")
            continue

        print(f"\n--- 開始批次處理 {len(subject_file_list)} 個 {class_name} 檔案 ---")
        
        successful_runs = 0
        failed_runs = 0
        main_pbar = tqdm(subject_file_list, desc=f"批次推論 ({class_name})", unit="file")
        
        # --- 3. (*** 關鍵: 內層迴圈 (檔案) ***) ---
        for nifti_path in main_pbar:
            
            subject_id = os.path.basename(nifti_path).split('.')[0]
            main_pbar.set_postfix({"file": f"{subject_id}.nii..."})

            # 1. (V10) 手動載入資料和 Affine
            inputs, output_affine = load_single_subject_data_manual(nifti_path)
            
            if inputs is None or output_affine is None:
                print(f"  跳過檔案 (載入失敗): {subject_id}")
                failed_runs += 1
                continue

            try:
                # 2. (V15) 攔截 5 個模型的特徵圖
                all_activations = []
                for model in models:
                    extractor = FeatureExtractor(net=model, target_layer=model.block4)
                    features = extractor.get_features(x=inputs) # (1, 128, 8, 8, 8)
                    extractor.remove_hook()
                    all_activations.append(features)

                # 3. 平均特徵圖
                mean_activation_map = torch.mean(torch.stack(all_activations), dim=0)
                
                # 4. 儲存「中間產物」
                save_intermediate_results(
                    activation_tensor=mean_activation_map,
                    affine_matrix=output_affine,
                    subject_id=subject_id,
                    output_dir_activations=OUTPUT_DIR_ACTIVATIONS,
                    target_class=class_name
                )
                successful_runs += 1
                
            except Exception as e:
                print(f"❌ 錯誤: 處理 {subject_id} 時推論或儲存失敗: {e}")
                failed_runs += 1
                continue
        
        print(f"\n--- {class_name} 處理完成 ---")
        print(f"  成功: {successful_runs} / {len(subject_file_list)}")
        print(f"  失敗: {failed_runs} / {len(subject_file_list)}")
        
        total_successful += successful_runs
        total_failed += failed_runs

    print("\n==================================================")
    print(f"✅✅ 流程 1 (AD + NC) 全部完成！ ✅✅")
    print(f"  總共成功處理: {total_successful} / {total_successful + total_failed} 個檔案")
    print(f"  總共失敗/跳過: {total_failed} / {total_successful + total_failed} 個檔案")
    print(f"  所有 Activations 和 Affines 皆已儲存至: {OUTPUT_DIR_ACTIVATIONS}")
    print("==================================================")


if __name__ == "__main__":
    main()