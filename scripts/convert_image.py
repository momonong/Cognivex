# scripts/convert_images.py
import os
import glob
import time
import cv2
import numpy as np
import nibabel as nib
import argparse
from typing import Optional


def convert_4d_nii_to_png(
    input_dir: str,
    output_dir: str,
    drop_slices: int = 10,
    verbose: bool = True
) -> None:
    """
    Convert .nii.gz 4D fMRI to 2D PNG, and auto-classify into subfolders like AD, NC.
    """
    os.makedirs(output_dir, exist_ok=True)
    nii_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))

    start = time.time()

    for idx, file_path in enumerate(nii_files):
        file_name = os.path.basename(file_path)
        file_id = os.path.splitext(file_name)[0]
        img_data = nib.load(file_path).get_fdata()

        if np.sum(img_data) == 0:
            continue

        img_data = (img_data / np.max(img_data)) * 255

        # 判斷 label 資料夾
        if "AD" in file_name.upper():
            label = "AD"
        elif "CN" in file_name.upper():
            label = "CN"
        else:
            raise ValueError(f"Unknown label in file name: {file_name}")

        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        depth = img_data.shape[2]
        timepoints = img_data.shape[3]

        for z in range(depth - drop_slices):
            for t in range(timepoints):
                slice_img = img_data[:, :, z, t].astype(np.uint8).T
                if np.sum(slice_img) == 0:
                    if verbose:
                        print(f"[SKIP] empty: {file_name}, z={z}, t={t}")
                    continue

                filename = f"{file_id}_{z+1:03d}_{t+1:03d}.png"
                save_path = os.path.join(label_dir, filename)
                cv2.imwrite(save_path, slice_img)
                if verbose:
                    print(f"[SAVE] {save_path}")

    end = time.time()
    print(f"[DONE] Total time: {end - start:.2f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-d", "--drop_slices", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    convert_4d_nii_to_png(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        drop_slices=args.drop_slices,
        verbose=args.verbose,
    )
