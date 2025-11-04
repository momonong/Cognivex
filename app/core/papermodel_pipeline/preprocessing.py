# app/core/papermodel_pipeline/preprocessing.py
import numpy as np
import nibabel as nib
import cv2
import os
from typing import Optional

# --- 來自 train.py 的全局配置 ---
# 論文 3.1 節: "10 consecutive slices are selected from the center"
NUM_SLICES_PER_SUBJECT = 10
# 論文 3.1 節: "with a slice size of 128x128"
SLICE_IMG_SIZE = 128

def preprocess_nii_to_slices(nii_path: str) -> Optional[np.ndarray]:
    """
    載入一個 .nii 檔案, 並執行論文中的切片預處理。
    1. 載入 3D 影像。
    2. 選取 sagittal plane。
    3. 找到中央 10 張切片。
    4. 旋轉、標準化 (0-255) 並縮放至 128x128。

    返回:
        Numpy array, shape (NUM_SLICES, 1, SLICE_IMG_SIZE, SLICE_IMG_SIZE)
        如果處理失敗則返回 None
    """
    try:
        # 1. 載入 NIfTI 影像
        img = nib.load(nii_path)
        data = img.get_fdata()

        # 2. 選取矢状面 (sagittal plane)
        # NIfTI 儲存通常是 (Sagittal, Coronal, Axial) -> (X, Y, Z)
        sagittal_dim = 0
        num_total_slices = data.shape[sagittal_dim]

        if num_total_slices < NUM_SLICES_PER_SUBJECT:
            print(f"警告：檔案 {nii_path} 矢狀面切片數 ({num_total_slices}) 少於 {NUM_SLICES_PER_SUBJECT}。將跳過此檔案。")
            return None

        # 3. 找到中央 10 張切片
        center_slice_index = num_total_slices // 2
        start_index = center_slice_index - (NUM_SLICES_PER_SUBJECT // 2)
        end_index = start_index + NUM_SLICES_PER_SUBJECT

        # 選取切片 [start_index:end_index, :, :]
        selected_slices_data = data[start_index:end_index, :, :]

        processed_slices = []
        for i in range(NUM_SLICES_PER_SUBJECT):
            slice_2d = selected_slices_data[i, :, :]

            # (重要) 旋轉影像使其方向正確
            slice_2d = np.rot90(slice_2d)

            # (重要) 將體素強度標準化到 0-255 (灰階圖片)
            if np.max(slice_2d) > 0:
                slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))
            slice_2d_uint8 = (slice_2d * 255).astype(np.uint8)

            # 4. 縮放到 128x128
            resized_slice = cv2.resize(slice_2d_uint8, (SLICE_IMG_SIZE, SLICE_IMG_SIZE),
                                       interpolation=cv2.INTER_CUBIC)

            processed_slices.append(resized_slice)

        # 堆疊成 (10, 128, 128)
        stacked_slices = np.stack(processed_slices)

        # 增加通道維度 -> (10, 1, 128, 128)
        # 10 張切片, 1 個灰階通道, 128x128 像素
        return stacked_slices[:, np.newaxis, :, :]

    except Exception as e:
        print(f"錯誤：處理檔案 {nii_path} 失敗: {e}")
        return None