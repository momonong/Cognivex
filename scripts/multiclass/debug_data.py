"""
除錯腳本：檢查資料是否正常
"""

import torch
import numpy as np
import nibabel as nib
import glob
import os
from nilearn import datasets, image as nimg

DATA_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/"

print("="*60)
print("資料除錯檢查")
print("="*60)

# 1. 檢查檔案
print("\n1. 檢查檔案...")
nc_files = glob.glob(os.path.join(DATA_ROOT, "NC", "*_T1.nii.gz"))
mci_files = glob.glob(os.path.join(DATA_ROOT, "MCI", "*_T1.nii.gz"))
ad_files = glob.glob(os.path.join(DATA_ROOT, "AD", "*_T1.nii.gz"))

print(f"NC: {len(nc_files)} 個檔案")
print(f"MCI: {len(mci_files)} 個檔案")
print(f"AD: {len(ad_files)} 個檔案")

if len(nc_files) == 0:
    print("❌ 錯誤：找不到 NC 檔案！")
    exit()

# 2. 載入第一個檔案
print("\n2. 載入第一個 NC 樣本...")
test_file = nc_files[0]
print(f"檔案: {os.path.basename(test_file)}")

img = nib.load(test_file)
data = img.get_fdata()

print(f"形狀: {data.shape}")
print(f"數值範圍: [{data.min():.2f}, {data.max():.2f}]")
print(f"平均值: {data.mean():.2f}")
print(f"標準差: {data.std():.2f}")
print(f"是否有 NaN: {np.isnan(data).any()}")
print(f"是否有 Inf: {np.isinf(data).any()}")

# 3. 檢查 AAL atlas
print("\n3. 檢查 AAL atlas...")
aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
aal_img = nimg.load_img(aal_atlas.maps)
aal_data = aal_img.get_fdata()

print(f"AAL 形狀: {aal_data.shape}")
print(f"AAL 數值範圍: [{aal_data.min():.0f}, {aal_data.max():.0f}]")
print(f"AAL 唯一值數量: {len(np.unique(aal_data))}")

# 4. 檢查對齊
print("\n4. 檢查影像與 AAL 的對齊...")
if data.shape != aal_data.shape:
    print(f"⚠️ 警告：形狀不匹配！")
    print(f"  影像: {data.shape}")
    print(f"  AAL: {aal_data.shape}")
    
    # 嘗試重採樣
    print("\n  嘗試重採樣 AAL...")
    aal_img_resampled = nimg.resample_to_img(aal_img, img, interpolation='nearest')
    aal_data_resampled = aal_img_resampled.get_fdata()
    print(f"  重採樣後 AAL 形狀: {aal_data_resampled.shape}")
    aal_data = aal_data_resampled
else:
    print("✅ 形狀匹配")

# 5. 提取 ROI 特徵
print("\n5. 測試 ROI 提取...")
roi_id = 37  # 左側海馬迴
roi_mask = (aal_data == roi_id)
roi_voxels = data[roi_mask]

print(f"ROI {roi_id} (左側海馬迴):")
print(f"  體素數量: {len(roi_voxels)}")
if len(roi_voxels) > 0:
    print(f"  平均值: {roi_voxels.mean():.4f}")
    print(f"  標準差: {roi_voxels.std():.4f}")
    print(f"  範圍: [{roi_voxels.min():.2f}, {roi_voxels.max():.2f}]")
else:
    print("  ❌ 錯誤：ROI 中沒有體素！")

# 6. 測試標準化
print("\n6. 測試標準化...")
if len(roi_voxels) > 0:
    # Min-Max 標準化
    normalized = (roi_voxels - roi_voxels.min()) / (roi_voxels.max() - roi_voxels.min() + 1e-8)
    print(f"Min-Max 標準化後: [{normalized.min():.4f}, {normalized.max():.4f}], 平均: {normalized.mean():.4f}")
    
    # Z-score 標準化
    z_normalized = (roi_voxels - roi_voxels.mean()) / (roi_voxels.std() + 1e-8)
    print(f"Z-score 標準化後: [{z_normalized.min():.4f}, {z_normalized.max():.4f}], 平均: {z_normalized.mean():.4f}")

# 7. 測試 PyTorch tensor
print("\n7. 測試 PyTorch tensor 轉換...")
tensor = torch.tensor(data, dtype=torch.float32)
print(f"Tensor 形狀: {tensor.shape}")
print(f"Tensor 數值範圍: [{tensor.min():.2f}, {tensor.max():.2f}]")
print(f"Tensor 是否有 NaN: {torch.isnan(tensor).any()}")
print(f"Tensor 是否有 Inf: {torch.isinf(tensor).any()}")

# 8. 測試 patch 提取
print("\n8. 測試 32x32x32 patch 提取...")
patch_size = 32
if data.shape[0] >= patch_size and data.shape[1] >= patch_size and data.shape[2] >= patch_size:
    patch = data[:patch_size, :patch_size, :patch_size]
    print(f"Patch 形狀: {patch.shape}")
    print(f"Patch 數值範圍: [{patch.min():.2f}, {patch.max():.2f}]")
    print(f"Patch 平均值: {patch.mean():.4f}")
else:
    print("❌ 影像太小，無法提取 32x32x32 patch")

print("\n" + "="*60)
print("除錯檢查完成")
print("="*60)
