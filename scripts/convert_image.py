import numpy as np
import nibabel as nib
import glob
import os 
import cv2
import time
import argparse

"""
This script reads nii.gz fMRI files (4D data) and extracts slices into 2D PNG format.
Enhanced to automatically split output into AD/CN folders based on filename.
Originally from: MCADNNet - https://ieeexplore.ieee.org/document/8883215
"""

def imgconverting(foldername, targetfolder, slicedrop=10, printfilename=True):
    start = time.time()
    os.makedirs(targetfolder, exist_ok=True)
    nii_files = glob.glob(os.path.join(foldername, '*.nii.gz'))

    for hh, file_path in enumerate(nii_files):
        file_name = os.path.basename(file_path)
        file_id = os.path.splitext(file_name)[0]

        # Determine label based on filename
        if "AD" in file_name.upper():
            label = "AD"
        elif "CN" in file_name.upper():
            label = "CN"
        else:
            raise ValueError(f"Unknown label in filename: {file_name}")

        # Create label-specific folder
        label_folder = os.path.join(targetfolder, label)
        os.makedirs(label_folder, exist_ok=True)

        img_data = nib.load(file_path).get_fdata()
        if img_data.sum() == 0:
            continue

        img_data = (img_data / np.max(img_data)) * 255
        depth = img_data.shape[2]
        timepoints = img_data.shape[3]

        for z in range(depth - slicedrop):
            for t in range(timepoints):
                img_slice = img_data[:, :, z, t]
                img_slice = img_slice.astype(np.uint8).T  # transpose to match expected orientation
                if img_slice.sum() == 0:
                    if printfilename:
                        print(f"[SKIP] Empty slice: {file_name} z={z}, t={t}")
                    continue

                filename = f"{label}_{file_id}_z{z+1:03d}_t{t+1:03d}.png"
                save_path = os.path.join(label_folder, filename)
                cv2.imwrite(save_path, img_slice)
                if printfilename:
                    print(f"[SAVE] {save_path}")

    end = time.time()
    print(f"[DONE] Total time: {end - start:.2f} sec")


def main(parserargs):
    input_dir = parserargs.inputfolder
    output_dir = parserargs.targetfolder
    imgconverting(input_dir, output_dir, slicedrop=parserargs.drop, printfilename=parserargs.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfolder", required=True, help="Input folder containing nii.gz files")
    parser.add_argument("-o", "--targetfolder", required=True, help="Target folder to store PNG slices")
    parser.add_argument("-d", "--drop", type=int, default=10, help="Number of slices to drop from the end. Default = 10")
    parser.add_argument("-v", "--verbose", type=bool, default=True, help="Print file names")
    args = parser.parse_args()
    main(args)
