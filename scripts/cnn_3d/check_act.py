# scripts/cnn_3d/debug_check_activations.py
#
# 除錯腳本: 用來分析「階段 1」儲存的 .pt 檔案
# 檢查它們是否為空或訊號太弱

import os
import sys
import numpy as np
import glob
import torch
from tqdm import tqdm

# --- 導入 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    print(f"將專案根目錄加入路徑: {PROJECT_ROOT}")
    sys.path.append(PROJECT_ROOT)

# --- 1. 配置 ---
# (我們只在 CPU 上分析, 不需要 GPU)
DEVICE = torch.device("cpu")

INPUT_DIR_ACTIVATIONS = os.path.join(PROJECT_ROOT, "output/cnn_3d/activations/")

def main():
    """
    分析所有 .pt 檔案的統計數據
    """
    
    print(f"--- Activation 除錯腳本 (V16) ---")
    print(f"  正在分析資料夾: {INPUT_DIR_ACTIVATIONS}")
    print(f"----------------------------------")
    
    # 1. 掃描所有「中間產物」
    activation_files = glob.glob(os.path.join(INPUT_DIR_ACTIVATIONS, "*_activation.pt"))
    
    if not activation_files:
        print(f"❌ 錯誤: 找不到任何 '_activation.pt' 檔案於:")
        print(f"  {INPUT_DIR_ACTIVATIONS}")
        print("  請先執行 '03_generate_activations.py'")
        return
        
    print(f"✅ 找到 {len(activation_files)} 個 Activation (.pt) 檔案。")
    print(f"\n--- 開始逐一分析檔案 ---")

    # 2. 準備儲存統計數據
    all_maxes = []
    all_means = []
    all_stds = []
    all_p99s = []
    all_zero_percs = []

    # 3. (核心邏輯: 批次迴圈)
    main_pbar = tqdm(activation_files, desc="分析 Activation 中", unit="file")
    
    for act_path in main_pbar:
        base_name = os.path.basename(act_path)
        main_pbar.set_postfix({"file": f"{base_name}..."})
        
        try:
            # 載入 .pt 檔案
            # (shape: [1, 128, 8, 8, 8])
            act_tensor = torch.load(act_path, map_location=DEVICE)
            
            # --- 計算統計數據 ---
            total_elements = act_tensor.numel()
            if total_elements == 0:
                print(f"⚠️ 警告: {base_name} 是空的 (0 elements)!")
                continue

            # (*** 關鍵 ***)
            all_maxes.append(torch.max(act_tensor).item())
            all_means.append(torch.mean(act_tensor).item())
            all_stds.append(torch.std(act_tensor).item())
            
            # 檢查你的 99% 閾值
            all_p99s.append(torch.quantile(act_tensor.float(), 0.99).item())
            
            # 檢查是否都是 0
            zero_count = torch.sum(act_tensor == 0).item()
            all_zero_percs.append((zero_count / total_elements) * 100)
            
        except Exception as e:
            print(f"❌ 錯誤: 處理 {base_name} 失敗: {e}")
            continue

    # 4. (*** 關鍵: 總結報告 ***)
    print("\n\n==================================================")
    print(f"✅✅ Activation 分析完成！ ✅✅")
    print(f"  (總共分析了 {len(all_maxes)} / {len(activation_files)} 個檔案)")
    print(f"----------------------------------")
    
    if not all_maxes:
        print("❌ 沒有任何檔案被成功分析。")
        return

    print(f"\n--- 訊號強度 (Activation Max) ---")
    print(f"  所有檔案的「平均」最大值: {np.mean(all_maxes):.6f}")
    print(f"  所有檔案的「最強」最大值: {np.max(all_maxes):.6f}")
    print(f"  所有檔案的「最弱」最大值: {np.min(all_maxes):.6f}")

    print(f"\n--- 訊號平均 (Activation Mean) ---")
    print(f"  所有檔案的「平均」平均值: {np.mean(all_means):.6f}")

    print(f"\n--- 訊號標準差 (Activation Std) ---")
    print(f"  所有檔案的「平均」標準差: {np.mean(all_stds):.6f}")

    print(f"\n--- 99% 閾值 (P99) ---")
    print(f"  所有檔案的「平均」P99 值: {np.mean(all_p99s):.6f}")
    
    print(f"\n--- 零值百分比 (Zero %) ---")
    print(f"  所有檔案的「平均」零值佔比: {np.mean(all_zero_percs):.2f} %")

    print("\n==================================================")

    # (*** 最終診斷 ***)
    if np.max(all_maxes) == 0.0:
        print("\n❌ 診斷: 災難性失敗！")
        print("  所有儲存的 Activation 熱圖的「最大值」都是 0.0。")
        print("  這 100% 意味著我們的 'FeatureExtractor' (掛鉤) 邏輯是錯的。")
    elif np.mean(all_p99s) < 1e-6:
        print("\n⚠️ 診斷: 訊號極度微弱！")
        print(f"  雖然有訊號 (Max: {np.max(all_maxes):.6f}), 但 99% 閾值太低 ({np.mean(all_p99s):.6f}).")
        print("  這可能導致 '04_visualize_activations.py' 的閾值化 (thresholding) 失敗。")
    else:
         print("\n✅ 診斷: 訊號看起來正常！")
         print("  Activations (.pt) 檔案中包含有效的、非零的訊號。")
         print("  問題可能出在 '04_visualize_activations.py' 的「插值 (Interpolate)」或「儲存」邏輯。")
            

if __name__ == "__main__":
    main()