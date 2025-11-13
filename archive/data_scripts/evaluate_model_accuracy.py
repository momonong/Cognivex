# evaluate_model_accuracy.py
import os
import glob
import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from dotenv import load_dotenv
from pathlib import Path
import traceback

# --- 導入我們重構後的程式碼 ---
# (假設此腳本在專案根目錄, 'app' 資料夾旁邊)
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from app.core.papermodel_pipeline.model import PaperModel
    from app.core.papermodel_pipeline.preprocessing import preprocess_nii_to_slices
except ImportError:
    print("錯誤：無法導入 'app.core.papermodel_pipeline' 模組。")
    print("請確保您在專案的根目錄 (與 'app' 資料夾同層) 執行此腳本。")
    exit(1)

# --- 1. 加載配置 ---
load_dotenv() # 加載 .env 檔案

# 從 .env 讀取路徑, 如果 .env 中沒有, 則使用預設值
DATA_DIR = os.getenv("DATA_DIR")
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "model/shufflenet/shufflenet_best_model.pth")

# 這兩個名稱來自您的 train.py 檔案
CLASS_A_NAME = "AD" # 標籤為 1
CLASS_B_NAME = "NC" # 標籤為 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"將在 {DEVICE} 設備上運行")

# --- 2. 輔助函式 (來自 train.py) ---

def find_nii_files(base_dir, class_a_name, class_b_name):
    """
    遞迴搜尋 .nii 和 .nii.gz 檔案
    """
    if not base_dir:
        print("錯誤：DATA_DIR 未在 .env 檔案中設定。")
        return None, None
        
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"錯誤：DATA_DIR 路徑不存在: {base_dir}")
        return None, None

    files_a = glob.glob(str(base_path / class_a_name / "**" / "*.nii"), recursive=True)
    files_b = glob.glob(str(base_path / class_b_name / "**" / "*.nii"), recursive=True)
    files_a.extend(glob.glob(str(base_path / class_a_name / "**" / "*.nii.gz"), recursive=True))
    files_b.extend(glob.glob(str(base_path / class_b_name / "**" / "*.nii.gz"), recursive=True))

    print(f"找到 {len(files_a)} 個 {class_a_name} 檔案")
    print(f"找到 {len(files_b)} 個 {class_b_name} 檔案")

    all_files = files_a + files_b
    all_labels = [1] * len(files_a) + [0] * len(files_b) # 1=AD, 0=NC

    if len(all_files) == 0:
        print(f"錯誤：在 {base_dir} 中找不到任何 .nii 或 .nii.gz 檔案。")
        print(f"預期路徑結構範例: {base_dir}\\{class_a_name}\\[Subject_Folder]\\[file].nii")
        
    return all_files, all_labels

def load_model(weights_path, device):
    """
    加載模型並設為 eval 模式
    """
    print(f"正在從 {weights_path} 加載模型權重...")
    if not os.path.exists(weights_path):
        print(f"錯誤：模型權重未找到: {weights_path}")
        return None
    try:
        model = PaperModel().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        print("模型加載成功並設為 eval() 模式。")
        return model
    except Exception as e:
        print(f"錯誤：加載模型失敗: {e}")
        traceback.print_exc()
        return None

# --- 3. 主執行腳本 ---

def main():
    print("--- 開始模型全局準確率評估 ---")
    
    # 1. 加載模型
    model = load_model(MODEL_WEIGHTS_PATH, DEVICE)
    if model is None:
        return

    # 2. 尋找所有資料檔案
    print(f"正在從 {DATA_DIR} 搜尋資料...")
    all_files, all_labels = find_nii_files(DATA_DIR, CLASS_A_NAME, CLASS_B_NAME)
    
    if not all_files:
        print("評估中止：找不到任何資料檔案。")
        return
        
    print(f"總共找到 {len(all_files)} 個受試者進行評估。")

    # 3. 循環所有檔案並進行預測
    all_preds = []
    all_probs = []
    ground_truth_labels = []
    
    # 使用 tqdm 顯示進度條
    for i in tqdm(range(len(all_files)), desc="正在評估受試者"):
        file_path = all_files[i]
        true_label = all_labels[i]
        
        # 預處理
        slices_array = preprocess_nii_to_slices(file_path)
        
        if slices_array is None:
            print(f"\n警告：跳過檔案 {file_path}，預處理失敗。")
            continue
        
        # 轉換為 Tensor
        slices_tensor = torch.tensor(slices_array, dtype=torch.float32) / 255.0
        slices_tensor = slices_tensor.unsqueeze(0).to(DEVICE) # 增加 Batch 維度 [1, 10, 1, 128, 128]
        
        # 執行推論
        try:
            with torch.no_grad():
                logits, _ = model(slices_tensor)
                
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            prob_ad = probs[0, 1].item() # 類別 1 (AD) 的機率
            
            all_preds.append(pred_idx)
            all_probs.append(prob_ad)
            ground_truth_labels.append(true_label)
            
        except Exception as e:
            print(f"\n錯誤：在推論 {file_path} 時失敗: {e}")

    # 4. 計算最終指標
    print("\n--- 評估完成 ---")
    
    if not ground_truth_labels:
        print("錯誤：沒有任何受試者被成功評估。")
        return

    acc = accuracy_score(ground_truth_labels, all_preds)
    f1 = f1_score(ground_truth_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(ground_truth_labels, all_probs)
    except ValueError as e:
        print(f"警告：計算 AUC 失敗 (可能所有標籤都相同?): {e}")
        auc = 0.0

    cm = confusion_matrix(ground_truth_labels, all_preds)
    
    # 確保 cm 是 2x2, 即使預測都是同一類
    if cm.shape == (1, 1):
        if all_preds[0] == 0: # 都是 NC
            cm = np.array([[cm[0,0], 0], [0, 0]])
        else: # 都是 AD
            cm = np.array([[0, 0], [0, cm[0,0]]])
    
    tn, fp, fn, tp = cm.ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Sensitivity (SEN)
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity (SPE)

    # 5. 打印報告
    print(f"\n評估的總受試者數: {len(ground_truth_labels)}")
    print("=========================================")
    print(f" 準確率 (Accuracy): {acc * 100:.2f}%")
    print(f" 敏感性 (Sensitivity / Recall): {sen * 100:.2f}%")
    print(f" 特異性 (Specificity): {spe * 100:.2f}%")
    print(f" F1-Score: {f1:.4f}")
    print(f" AUC: {auc:.4f}")
    print("=========================================")
    print(" 混淆矩陣 (Confusion Matrix):")
    print("           預測 NC | 預測 AD")
    print(f"實際 NC:    {tn:4d} | {fp:4d}")
    print(f"實際 AD:    {fn:4d} | {tp:4d}")
    print("=========================================")


if __name__ == "__main__":
    main()