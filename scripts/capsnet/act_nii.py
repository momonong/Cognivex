import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F

# Load activation
act = torch.load("output/capsnet/sub-14_conv3.pt")[0]  # shape: [C, D, H, W]

# 使用最強 channel
channel_idx = torch.norm(act, p=2, dim=(1,2,3)).argmax()
act = act[channel_idx]

# Resize to original shape
ref_img = nib.load("data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz")
affine = ref_img.affine
ref_shape = ref_img.shape[:3]
act_tensor = act.unsqueeze(0).unsqueeze(0)
resized = F.interpolate(act_tensor, size=(ref_shape[2], ref_shape[0], ref_shape[1]), mode="trilinear", align_corners=False)
resized = resized.squeeze().numpy()
nii_data = np.transpose(resized, (1, 2, 0))  # [X, Y, Z]

# Normalize + threshold top 2%
nii_data = (nii_data - nii_data.min()) / (nii_data.max() - nii_data.min() + 1e-8)
threshold = np.percentile(nii_data, 99)
nii_masked = np.where(nii_data >= threshold, nii_data, 0).astype(np.float32)

# Save NIfTI
nib.save(nib.Nifti1Image(nii_masked, affine), "output/capsnet/sub-14_conv3_strongest_masked.nii.gz")
print("✅ Saved filtered, strongest channel activation.")
print("path:", "output/capsnet/sub-14_conv3_strongest_masked.nii.gz")
