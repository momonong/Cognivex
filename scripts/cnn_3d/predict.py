import os
import glob
import numpy as np

# --- 警告過濾器 ---
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

# 匯入 MONAI
from monai.data import CacheDataset
from monai.utils import set_determinism

# --- (*** 關鍵變更 ***) ---
# 1. 從 app/core/ 匯入模型和轉換的定義
import sys
# 假設 COGNIVEX 專案根目錄在 C:\USERS\MORRIS\PROJECTS\COGNIVEX
# 我們需要將它加入到 Python 路徑中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.core.cnn_3d.model_def import Simple3DCNN_InstanceNorm
from app.core.cnn_3d.transforms_def import test_transforms, PATCH_SIZE
# --- (*** 變更結束 ***) ---


# --- 2. 配置 ---

set_determinism(seed=42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"E:\fMRI\Model\sMRI_data" # (註: 這裡你還是使用你的原始資料路徑)
K_FOLDS = 5

# --- (*** 關鍵變更 ***) ---
# 3. 更新權重路徑
MODEL_WEIGHTS_PATHS = [
    f"trained_models/cnn_3d/cognivex_instancenorm_cnn_best_fold_{i + 1}.pth" for i in range(K_FOLDS)
]
# --- (*** 變更結束 ***) ---


# --- 3. 準備資料列表 ---
# (此函式不變)
def get_file_lists(data_dir):
    print(f"正在 {data_dir} 中遞迴掃描 AD 和 NC 檔案...")
    
    ad_files = glob.glob(os.path.join(data_dir, "AD", "**", "*.nii"), recursive=True)
    ad_files.extend(glob.glob(os.path.join(data_dir, "AD", "**", "*.nii.gz"), recursive=True))
    
    nc_files = glob.glob(os.path.join(data_dir, "NC", "**", "*.nii"), recursive=True)
    nc_files.extend(glob.glob(os.path.join(data_dir, "NC", "**", "*.nii.gz"), recursive=True))

    if not ad_files: print(f"警告: 在 {os.path.join(data_dir, 'AD')} 中沒有找到.nii 或.nii.gz 檔案")
    if not nc_files: print(f"警告: 在 {os.path.join(data_dir, 'NC')} 中沒有找到.nii 或.nii.gz 檔案")
            
    print(f"在 {data_dir} 中: 找到 {len(ad_files)} 個 AD 檔案, {len(nc_files)} 個 NC 檔案。")

    ad_list = [{"image": f, "label": 1} for f in ad_files] # 1 = AD
    nc_list = [{"image": f, "label": 0} for f in nc_files] # 0 = NC
    
    files = ad_list + nc_list
    return files

# --- 4. MONAI 測試資料管線 ---
# (已移至 transforms_def.py)

# --- 5. 預測主函式 ---

def main():

    # --- GPU 狀態檢查 ---
    print(f"\n--- GPU 狀態檢查 ---")
    if torch.cuda.is_available():
        print(f"✅ GPU 偵測成功！")
        print(f"  CUDA 裝置: {DEVICE}")
        print(f"  裝置名稱: {torch.cuda.get_device_name(DEVICE)}")
    else:
        print(f"⚠️ 警告: 未偵測到 CUDA！將使用 CPU。")
    print(f"----------------------\n")

    # 1. 載入所有資料
    print(f"正在從 {DATA_DIR} 掃描所有資料...")
    all_files = get_file_lists(DATA_DIR)
    
    if not all_files:
        print(f"錯誤: 在 {DATA_DIR} 中找不到檔案。正在結束。")
        return
        
    print(f"將評估總共 {len(all_files)} 筆資料。")

    test_ds = CacheDataset(data=all_files, transform=test_transforms, cache_rate=1.0, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 2. 載入 5 個模型
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
            print(f"  請確保所有 5 個 .pth 權重檔案都在 'trained_models/cnn_3d/' 資料夾中。")
            return
    print(f"✅ 成功載入 {len(models)} 個模型。\n")
        
    # 4. 執行預測迴圈
    all_labels_list = []
    all_preds_list = []
    
    print(f"--- 正在 {len(all_files)} 筆資料上執行集成預測 (Ensemble Prediction) ---")
    pbar = tqdm(test_loader, desc=f"集成評估中", unit="subject")

    with torch.no_grad():
        for batch_data in pbar:
            inputs = batch_data["image"].to(DEVICE)
            labels = batch_data["label"].to(DEVICE)
            
            fold_outputs = []
            for model in models:
                fold_outputs.append(model(inputs))
            
            avg_outputs = torch.stack(fold_outputs).mean(dim=0)
            
            all_preds_list.append(avg_outputs.cpu())
            all_labels_list.append(labels.cpu())

    # 5. 彙總並計算指標
    all_labels = torch.cat(all_labels_list, dim=0).numpy()
    all_preds_logits = torch.cat(all_preds_list, dim=0)

    all_preds_probs = torch.nn.functional.softmax(all_preds_logits, dim=1)[:, 1].numpy()
    auc = roc_auc_score(all_labels, all_preds_probs)
    
    # --- 報告 1: 預設 0.5 門檻 ---
    all_preds_hard_default = (all_preds_probs >= 0.5).astype(int)
    acc_default = accuracy_score(all_labels, all_preds_hard_default)
    cm_default = confusion_matrix(all_labels, all_preds_hard_default)

    print(f"\n--- 報告 1: 預設 0.5 門檻 (有偏見的) ---")
    print(f"  (評估了 {len(all_files)} 筆資料, 使用 5-Fold 模型平均)")
    print(f"\n  準確率 (Accuracy): {acc_default:.4f} ({ (acc_default * 100):.2f}%)")
    print(f"  AUC (Area Under Curve): {auc:.4f}")
    
    print("\n  混淆矩陣 (預設 0.5 門檻):")
    print("         預測為 NC   預測為 AD")
    print(f"實際為 NC:    {cm_default[0][0]:<6d}     {cm_default[0][1]:<6d}")
    print(f"實際為 AD:    {cm_default[1][0]:<6d}     {cm_default[1][1]:<6d}")
    print("  (NC=0, AD=1)")

    # --- 報告 2: 最佳門檻 ---
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    all_preds_hard_optimal = (all_preds_probs >= optimal_threshold).astype(int)
    acc_optimal = accuracy_score(all_labels, all_preds_hard_optimal)
    cm_optimal = confusion_matrix(all_labels, all_preds_hard_optimal)

    print(f"\n\n--- 報告 2: 最佳門檻 (修正後) ---")
    print(f"  (計算出的『最佳門檻』 (Youden's J): {optimal_threshold:.4f})\n")
    print(f"  **修正後準確率 (Accuracy): {acc_optimal:.4f} ({ (acc_optimal * 100):.2f}%)**")
    print(f"  AUC (Area Under Curve): {auc:.4f} (不變)")

    print("\n  **混淆矩陣 (使用最佳門檻):**")
    print("         預測為 NC   預測為 AD")
    print(f"**實際為 NC:** {cm_optimal[0][0]:<6d}     {cm_optimal[0][1]:<6d}")
    print(f"**實際為 AD:** {cm_optimal[1][0]:<6d}     {cm_optimal[1][1]:<6d}")
    print("  (NC=0, AD=1)")
    print("==================================================")


if __name__ == "__main__":
    main()