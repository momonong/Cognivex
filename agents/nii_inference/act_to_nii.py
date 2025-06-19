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


def activation_to_nifti(
    activation_path: str,
    reference_nii_path: str,
    output_path: str,
    norm_type: str = "l2",
    threshold_percentile: float = 99.0,
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
    ref_shape = ref_img.shape[:3]
    print(f"[Reference image shape] {ref_shape}")

    # Step 3: Interpolate alignment
    act_tensor = act.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    resized = F.interpolate(
        act_tensor,
        size=(ref_shape[2], ref_shape[0], ref_shape[1]),  # Z, X, Y
        mode="trilinear",
        align_corners=False,
    )  # [1, 1, Z, X, Y]

    # Step 4: Reshape to [X, Y, Z]
    resized = resized.squeeze().numpy()
    nii_data = np.transpose(resized, (1, 2, 0))  # [X, Y, Z]

    # Step 5: Normalize + threshold
    nii_data = (nii_data - nii_data.min()) / (nii_data.max() - nii_data.min() + 1e-8)
    threshold = np.percentile(nii_data, threshold_percentile)
    nii_masked = np.where(nii_data >= threshold, nii_data, 0).astype(np.float32)

    # Step 6: Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib.Nifti1Image(nii_masked, affine), output_path)
    print("Saved filtered, strongest channel activation.")
    print("path:", output_path)


if __name__ == "__main__":
    # Example usage
    activation_to_nifti(
        activation_path="output/capsnet/module_test_conv3.pt",
        reference_nii_path="data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz",
        output_path="output/capsnet/module_test.nii.gz",
        norm_type="l2",
        threshold_percentile=99.0,
    )
