import os
import nibabel as nib
import numpy as np
import pandas as pd
import cv2
import uuid
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ---------- Configuration ----------
RAW_DIR = "data/raw"  # Your raw data directory (contains AD/ and CN/)
OUTPUT_SLICE_DIR = "data/slices"  # Output slice folder
META_CSV_PATH = "data/subject_metadata.csv"
SPLIT_FOLD = 5  # Number of folds for cross-validation
DROP_LAST_SLICES = 10  # Number of last slices to drop from z-axis for each image

# ---------- Create Metadata Table ----------
records = []
for label in ["AD", "CN"]:
    label_path = os.path.join(RAW_DIR, label)
    for subject_id in os.listdir(label_path):
        subject_dir = os.path.join(label_path, subject_id)
        if not os.path.isdir(subject_dir):
            continue
        nii_files = [f for f in os.listdir(subject_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
        for nii_file in nii_files:
            records.append({
                "subject_id": subject_id,
                "diagnosis": label,
                "nii_path": os.path.join(subject_dir, nii_file)
            })

df = pd.DataFrame(records)
df["uid"] = [uuid.uuid4().hex[:8] for _ in range(len(df))]

# ---------- Split folds ----------
skf = StratifiedKFold(n_splits=SPLIT_FOLD, shuffle=True, random_state=42)
df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(df, df["diagnosis"])):
    df.loc[val_idx, "fold"] = fold

# ---------- Start slicing ----------
slice_records = []
for row in tqdm(df.itertuples(), total=len(df), desc="Slicing"):
    img = nib.load(row.nii_path).get_fdata()
    if img.sum() == 0:
        continue

    img = (img / np.max(img)) * 255  # Normalization
    z_dim, t_dim = img.shape[2], img.shape[3]

    for z in range(z_dim - DROP_LAST_SLICES):
        for t in range(t_dim):
            img_slice = img[:, :, z, t]
            img_slice = img_slice.astype(np.uint8).T
            if img_slice.sum() == 0:
                continue

            slice_filename = f"{row.diagnosis}_{row.subject_id}_z{z+1:03d}_t{t+1:03d}.png"
            slice_path = os.path.join(OUTPUT_SLICE_DIR, f"fold{row.fold}", row.diagnosis)
            os.makedirs(slice_path, exist_ok=True)
            full_path = os.path.join(slice_path, slice_filename)

            cv2.imwrite(full_path, img_slice)

            slice_records.append({
                "filename": slice_filename,
                "subject_id": row.subject_id,
                "label": row.diagnosis,
                "fold": row.fold,
                "z": z + 1,
                "t": t + 1,
                "path": full_path
            })

# ---------- Save metadata ----------
pd.DataFrame(slice_records).to_csv("data/slices_metadata.csv", index=False)
df.to_csv(META_CSV_PATH, index=False)
print("✅ Slicing completed and metadata exported: slices_metadata.csv")
