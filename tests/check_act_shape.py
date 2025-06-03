import os
import numpy as np
import pandas as pd

# 設定資料夾路徑
activation_dir = "output/activations"

# 遍歷所有 .npy 檔案
results = []
for fname in os.listdir(activation_dir):
    if fname.endswith("_conv_volume.npy"):
        subject_id = fname.split("_")[0]
        npy_path = os.path.join(activation_dir, fname)
        volume = np.load(npy_path)
        results.append({
            "subject_id": subject_id,
            "npy_path": fname,
            "shape": volume.shape
        })

# 顯示結果表格
df = pd.DataFrame(results)
print(df)
