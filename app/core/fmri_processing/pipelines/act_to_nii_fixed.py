import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import os


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


def activation_to_nifti_fixed(
    activation_path: str,
    reference_nii_path: str,
    output_path: str,
    norm_type: str = "l2",
    threshold_percentile: float = 99.0,
):
    """
    Convert activation tensor to NIfTI file aligned with original fMRI.
    FIXED VERSION with correct coordinate system handling.
    
    - activation_path: Path to saved .pt tensor, shape [B, C, D, H, W]
    - reference_nii_path: Original NIfTI file for alignment, used to get affine and spatial dimensions
    - output_path: Path for the final output NIfTI file
    - norm_type: "l1" or "l2", determines how to select the strongest channel
    - threshold_percentile: Percentage of strongest activations to keep for masking
    """
    print(f"🔧 [FIXED] Converting activation to NIfTI...")
    
    act = torch.load(activation_path)[0]  # shape: [C, D, H, W]
    print(f"[Original activation] shape: {act.shape}")
    
    # Flatten if needed (e.g., CapsuleLayer: [32, 8, D, H, W] -> [256, D, H, W])
    if act.ndim > 4:
        act = act.reshape(act.shape[0] * act.shape[1], *act.shape[2:])
    print(f"[Flattened activation] shape: {act.shape}")

    # Step 1: Select strongest channel
    channel_idx = select_strongest_channel(act, norm_type)
    act = act[channel_idx]  # shape: [D, H, W]
    print(f"[Selected channel {channel_idx}] shape: {act.shape}")

    # Step 2: Load reference image
    ref_img = nib.load(reference_nii_path)
    affine = ref_img.affine
    ref_shape = ref_img.shape[:3]  # (X, Y, Z)
    print(f"[Reference image shape] {ref_shape}")
    print(f"[Reference affine]\\n{affine}")

    # Step 3: FIXED - Interpolate with correct dimension mapping
    # Activation is [D, H, W], we want to map it to reference [X, Y, Z]
    # Let's assume activation [D, H, W] corresponds to [X, Y, Z] respectively
    act_tensor = act.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    print(f"[Pre-interpolation] shape: {act_tensor.shape}")
    
    # FIXED: Direct mapping D->X, H->Y, W->Z
    resized = F.interpolate(
        act_tensor,
        size=ref_shape,  # (X, Y, Z) - direct mapping
        mode="trilinear",
        align_corners=False,
    )  # [1, 1, X, Y, Z]
    print(f"[Post-interpolation] shape: {resized.shape}")

    # Step 4: FIXED - No transpose needed since we already have [X, Y, Z]
    nii_data = resized.squeeze().numpy()  # [X, Y, Z]
    print(f"[Final NIfTI data] shape: {nii_data.shape}")

    # Step 5: Normalize + threshold
    data_min, data_max = nii_data.min(), nii_data.max()
    nii_data = (nii_data - data_min) / (data_max - data_min + 1e-8)
    threshold = np.percentile(nii_data, threshold_percentile)
    nii_masked = np.where(nii_data >= threshold, nii_data, 0).astype(np.float32)
    
    print(f"[Normalization] min={data_min:.6f}, max={data_max:.6f}")
    print(f"[Threshold] {threshold_percentile}th percentile: {threshold:.6f}")
    print(f"[Masked] non-zero voxels: {(nii_masked > 0).sum()}")

    # Step 6: Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib.Nifti1Image(nii_masked, affine), output_path)
    print(f"✅ Saved fixed activation to: {output_path}")


if __name__ == "__main__":
    # Test with the same data that was causing problems
    activation_to_nifti_fixed(
        activation_path="output/langraph/sub-01_capsnet_conv2.pt",
        reference_nii_path="data/raw/CN/sub-01/dswausub-009_S_0751_task-rest_bold.nii.gz",
        output_path="output/test_fixed_activation.nii.gz",
        norm_type="l2",
        threshold_percentile=99.0,
    )
