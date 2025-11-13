import numpy as np
import pandas as pd
from pathlib import Path

# ---------- 設定 ----------
meta_path = Path("data/activation_slices_metadata.csv")  # 你的 slice metadata 檔案
volume_dir = Path("output/activations")
meta_df = pd.read_csv(meta_path)

# ---------- 計算每個 subject 的 T, Z 維度 ----------
tz_info = (
    meta_df.groupby("subject_id")[["t", "z"]]
    .agg({"t": "max", "z": "max"})
    .reset_index()
    .rename(columns={"t": "max_t", "z": "max_z"})
)
tz_info["T"] = tz_info["max_t"] + 1
tz_info["Z"] = tz_info["max_z"] + 1

records = []

for _, row in tz_info.iterrows():
    subject_id = row["subject_id"]
    T = row["T"]
    Z = row["Z"]
    npy_path = volume_dir / f"{subject_id}_conv_volume.npy"

    if not npy_path.exists():
        print(f"⚠️ 找不到檔案: {npy_path}")
        continue

    volume = np.load(npy_path)  # [T * Z, 9, 9]
    if volume.shape[0] != T * Z:
        print(f"❌ 維度不符: {subject_id} | shape[0]={volume.shape[0]}, 但 T*Z={T}x{Z}={T*Z}")
        continue

    volume_4d = volume.reshape(T, Z, 9, 9)
    max_value = np.max(volume_4d)
    t, z, y, x = np.unravel_index(np.argmax(volume_4d), volume_4d.shape)

    records.append({
        "subject_id": subject_id,
        "T": T,
        "Z": Z,
        "max_activation": max_value,
        "t": t,
        "z": z,
        "y": y,
        "x": x,
    })

# ---------- 儲存分析結果 ----------
result_df = pd.DataFrame(records)
result_df = result_df.sort_values(by="max_activation", ascending=False)

output_path = volume_dir / "activation_peak_report.csv"
result_df.to_csv(output_path, index=False)
print(f"✅ 已儲存 peak activation 分析至: {output_path}")
