import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 設定 ----------
conv_dir = "output/macadnnet/conv_volume"
subject_id = "sub-14"
subject_path = os.path.join(conv_dir, f"{subject_id}_conv_volume.npy")
output_dir = os.path.join("figures")
os.makedirs(output_dir, exist_ok=True)

# ---------- 載入資料 ----------
volume = np.load(subject_path)  # shape: [num_slices, H, W]

# ---------- 建立平均圖 ----------
avg_slice = np.mean(volume, axis=0)  # shape: [H, W]

# ---------- 儲存熱圖 ----------
plt.figure(figsize=(6, 5))
sns.heatmap(avg_slice, cmap="viridis", cbar=True)
plt.title(f"{subject_id} conv2 Activation (Mean over Slices)")
plt.tight_layout()
output_path = os.path.join(output_dir, f"{subject_id}_mean_heatmap.png")
plt.savefig(output_path)
plt.close()

print(f"✅ {subject_id} 的 conv activation 視覺化已儲存至：{output_path}")
