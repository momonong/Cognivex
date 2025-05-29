import os
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import imageio
import pandas as pd

# ---------- 設定 ----------
root_dir = Path("data/raw")
output_dir = Path("data/slices_from_nii")
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = Path("data/activation_slices_metadata.csv")

# ---------- 儲存 Metadata ----------
records = []

# ---------- 遍歷每個檔案 ----------
for label in ["AD", "CN"]:
    label_dir = root_dir / label
    for subject_dir in label_dir.iterdir():
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        nii_files = list(subject_dir.glob("*.nii"))

        for nii_path in nii_files:
            img = nib.load(str(nii_path))
            data = img.get_fdata()  # shape: (X, Y, Z, T)
            X, Y, Z, T = data.shape

            for t in tqdm(range(T), desc=f"{subject_id} time"):
                for z in range(Z):
                    slice_2d = data[:, :, z, t]

                    # Normalize for saving
                    norm_slice = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-5)
                    norm_slice = (norm_slice * 255).astype(np.uint8)

                    filename = f"{subject_id}_z{z:03d}_t{t:03d}.png"
                    save_dir = output_dir / label / subject_id
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / filename

                    imageio.imwrite(save_path, norm_slice)

                    records.append({
                        "filename": filename,
                        "subject_id": subject_id,
                        "label": label,
                        "z": z,
                        "t": t,
                        "path": str(save_path)
                    })

# ---------- 儲存 CSV ----------
df = pd.DataFrame(records)
df.to_csv(csv_path, index=False)
print(f"\n✅ 所有 slices 已儲存完畢，metadata 儲存至：{csv_path}")
