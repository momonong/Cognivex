# inspect_pt_file.py
import torch
import numpy as np

# --- 1. 請修改這裡！ ---
# 請將路徑指向【確定會導致錯誤】的那個 .pt 檔案
# 根據您的 pipeline，它應該類似於 'output/langraph/langraph_test_capsnet_conv3.pt'
PT_FILE_PATH = "output/langraph/sub-01_capsnet_conv3.pt"

print(f"--- 正在檢查檔案: {PT_FILE_PATH} ---")

try:
    # 載入檔案
    data = torch.load(PT_FILE_PATH, map_location='cpu')

    # 檢查載入的數據是什麼類型
    print(f"\n1. 載入的數據類型 (Type): {type(data)}")

    # --- 如果是字典 (Dictionary) ---
    if isinstance(data, dict):
        print(f"\n2. 這是一個字典，包含的鍵 (Keys): {list(data.keys())}")

        print("\n3. 正在檢查字典中每個鍵的內容:")
        for key, value in data.items():
            print(f"  - 鍵 (Key): '{key}'")
            print(f"    - 值的類型 (Type): {type(value)}")
            # 如果值是 Tensor 或 Numpy Array，就印出它的形狀
            if hasattr(value, 'shape'):
                print(f"    - 值的形狀 (Shape): {value.shape}")

    # --- 如果是 PyTorch Tensor ---
    elif isinstance(data, torch.Tensor):
        print(f"\n2. 這是一個 PyTorch Tensor。")
        print(f"3. Tensor 的形狀 (Shape): {data.shape}")

    # --- 如果是其他類型 ---
    else:
        print(f"\n2. 這是一個未知的數據類型。")

except Exception as e:
    print(f"\n!!! 讀取檔案時發生錯誤: {e} !!!")

print("\n--- 檢查完畢 ---")