import torch
import nibabel as nib
import numpy as np
import os
from nilearn.image import resample_to_img


def select_strongest_channel(act: torch.Tensor, norm_type: str = "l2") -> int:
    """
    Select the channel index with the strongest activation in the activation tensor.
    - act shape: [C, D, H, W]
    - norm_type: "l1" or "l2"
    """
    if norm_type == "l2":
        norms = torch.norm(act, p=2, dim=(1, 2, 3))
    elif norm_type == "l1":
        norms = torch.norm(act, p=1, dim=(1, 2, 3))
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")
    return norms.argmax().item()


def activation_to_nifti(
    activation_path: str,
    reference_nii_path: str,
    output_path: str,
    norm_type: str = "l2",
    threshold_percentile: float = 95.0,
):
    """
    Convert activation tensor to NIfTI file aligned with original fMRI.
    - activation_path: Path to saved .pt tensor, shape [B, C, D, H, W]
    - reference_nii_path: Original NIfTI file for alignment, used to get affine and spatial dimensions
    - output_path: Path for the final output NIfTI file
    - norm_type: "l1" or "l2", determines how to select the strongest channel
    - threshold_percentile: Percentage of strongest activations to keep for masking
    """
    act = torch.load(activation_path)[0]  # shape: [C, D, H, W]
    # print(f"[Original activation] shape: {act.shape}")
    # Flatten if needed (e.g., CapsuleLayer: [32, 8, D, H, W] -> [256, D, H, W])
    if act.ndim > 4:
        act = act.reshape(act.shape[0] * act.shape[1], *act.shape[2:])
    # print(f"[Flattened activation] shape: {act.shape}")

    # Step 1: Select strongest channel
    channel_idx = select_strongest_channel(act, norm_type)
    act = act[channel_idx]  # shape: [D, H, W]
    # print(f"[Selected channel {channel_idx}] shape: {act.shape}")

    # Step 2: Load reference image
    ref_img = nib.load(reference_nii_path)
    high_res_affine = ref_img.affine
    high_res_shape = ref_img.shape[:3]
    # print(f"[Reference image shape] {ref_shape}")

    # Step 3: FIXED - Interpolate with correct dimension mapping
    # Activation is [D, H, W], we want to map it to reference [X, Y, Z]
    # Assume activation [D, H, W] corresponds to [X, Y, Z] respectively
    low_res_shape = act.shape
    scaling_factors = np.array(high_res_shape) / np.array(low_res_shape)
    
    activation_affine = high_res_affine.copy()
    for i in range(3):
        # 根據尺寸比例，放大 affine 的對角線元素 (代表 voxel 尺寸)
        activation_affine[i, i] *= scaling_factors[i]
        
    # --- 步驟 4 (新): 使用 nilearn 執行【專業的空間重採樣】 ---
    # 這一步取代了 F.interpolate 和 np.transpose
    source_nii = nib.Nifti1Image(act.cpu().numpy(), activation_affine)
    resampled_nii = resample_to_img(
        source_img=source_nii, 
        target_img=ref_img, 
        interpolation='continuous',
        force_resample=True,
        copy_header=True,
    )
    nii_data = resampled_nii.get_fdata()

    # Step 5: Normalize + threshold
    positive_values = nii_data[nii_data > 0]
    if positive_values.size > 0:
        threshold = np.percentile(positive_values, threshold_percentile)
        nii_data[nii_data < threshold] = 0

    # Step 6: Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib.Nifti1Image(nii_data.astype(np.float32), resampled_nii.affine), output_path)

if __name__ == "__main__":
    # Example usage
    activation_to_nifti(
        activation_path="output/capsnet/module_test_conv3.pt",
        reference_nii_path="data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz",
        output_path="output/capsnet/module_test.nii.gz",
        norm_type="l2",
        threshold_percentile=99.0,
    )
