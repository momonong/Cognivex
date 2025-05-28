import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------- 設定 ----------
meta_csv = "output/macadnnet/activation_volume/activation_meta.csv"
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# ---------- 載入 meta ----------
df = pd.read_csv(meta_csv)

for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 分析 Activation"):
    subject_id = row["subject_id"]
    label = row["label"]
    path = row["path"]

    if not os.path.exists(path):
        print(f"❌ 檔案不存在: {path}")
        continue

    activation = np.load(path)  # shape: [n_slices, 256]
    flattened = activation.flatten()

    # 儲存分布圖
    plt.figure(figsize=(8, 4))
    sns.histplot(flattened, bins=50, kde=True, color="royalblue")
    plt.title(f"{subject_id} Activation Distribution")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{subject_id}_hist.png")
    plt.savefig(fig_path)
    plt.close()

print("✅ 所有 subject 分布圖已完成並儲存。")
