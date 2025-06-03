import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from glob import glob

# ---------- 參數 ----------
ACTIVATION_DIR = "output/activations"
METADATA_CSV = "data/activation_slices_metadata.csv"
RAW_DATA_ROOT = "data/raw"
OUTPUT_NII_DIR = "output/nifti_activation"
os.makedirs(OUTPUT_NII_DIR, exist_ok=True)

# ---------- 載入 metadata ----------
df = pd.read_csv(METADATA_CSV)
subject_list = df["subject_id"].unique().tolist()

for subject_id in tqdm(subject_list):
    subject_df = df[df["subject_id"] == subject_id].copy()
    act_path = os.path.join(ACTIVATION_DIR, f"{subject_id}_conv_volume.npy")
    
    if not os.path.exists(act_path):
        continue

    act_volume = np.load(act_path)  # shape: [N, 9, 9]
    if len(subject_df) != len(act_volume):
        continue

    label = subject_df["label"].iloc[0]
    nii_glob = glob(os.path.join(RAW_DATA_ROOT, label, subject_id, "*.nii*"))
    if not nii_glob:
        continue

    nii_img = nib.load(nii_glob[0])
    original_shape = nii_img.shape  # [X, Y, Z, T]
    affine = nii_img.affine
    activation_4d = np.zeros(original_shape, dtype=np.float32)

    for idx_in_df, row in enumerate(subject_df.itertuples()):
        z, t = int(row.z), int(row.t)
        if z >= original_shape[2] or t >= original_shape[3]:
            continue
        if idx_in_df >= len(act_volume):
            continue

        activation_map = act_volume[idx_in_df]
        if np.max(activation_map) == 0.0:
            continue

        x_center, y_center = original_shape[0] // 2, original_shape[1] // 2
        x_start, x_end = x_center - 4, x_center + 5
        y_start, y_end = y_center - 4, y_center + 5

        if (
            x_start < 0 or y_start < 0 or
            x_end > original_shape[0] or y_end > original_shape[1]
        ):
            continue

        activation_4d[x_start:x_end, y_start:y_end, z, t] = activation_map

    activation_4d *= 1000
    output_path = os.path.join(OUTPUT_NII_DIR, f"{subject_id}_activation_map.nii.gz")
    nib.save(nib.Nifti1Image(activation_4d, affine), output_path)
    print(f"✅ Saved: {output_path}")
    reloaded = nib.load(output_path).get_fdata()
    print(f"📦 Max activation value (after scaling): {reloaded.max():.4f}")

