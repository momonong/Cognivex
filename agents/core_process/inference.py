import torch
import nibabel as nib
import numpy as np
import os
from scripts.capsnet.model import CapsNetRNN


def load_nifti_and_preprocess(path: str, window: int, stride: int) -> torch.Tensor:
    """
    Load a NIfTI fMRI file and preprocess into sliding window format.
    Returns: torch.Tensor with shape [B, 1, T, D, H, W]
    """
    nii = nib.load(path)
    data = nii.get_fdata()  # [X, Y, Z, T]
    data = np.transpose(data, (3, 2, 0, 1))  # ➜ [T, Z, H, W]

    # Normalize to [0, 1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    # Sliding window over time
    clips = []
    for i in range(0, data.shape[0] - window + 1, stride):
        clip = data[i:i + window]  # [T, D, H, W]
        clips.append(clip)

    arr = np.stack(clips)  # [B, T, D, H, W]
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)  # ➜ [B, 1, T, D, H, W]
    return tensor


def run_inference(
    model: torch.nn.Module,
    model_path: str,
    nii_path: str,
    save_dir: str,
    save_name: str,
    window: int = 5,
    stride: int = 3,
    device: str = "cpu",
):
    """
    Run inference using a prepared model with hooks attached.

    Parameters:
        model: A prepared model instance with hook-attached layers
        nii_path: Path to input NIfTI file
        save_dir: Directory to save outputs
        save_name: Prefix name for saving activation files
        window: Temporal window size
        stride: Temporal stride size
        device: Inference device (e.g., "cpu", "cuda", "mps")
    """
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu")) 
    model.eval()

    # Load input
    inputs = load_nifti_and_preprocess(nii_path, window, stride).to(device)
    print(f"Loaded input shape: {inputs.shape}")  # [B, 1, T, D, H, W]

    # Inference
    with torch.no_grad():
        outputs = model(inputs)
        preds = (outputs > 0.5).float().squeeze().cpu().numpy()

    # Classification result
    final_pred = int(np.round(preds.mean()))
    print(f"Inference Result: {final_pred} (1=AD, 0=CN)")

    # Save activations
    if hasattr(model, "activations") and isinstance(model.activations, dict):
        os.makedirs(save_dir, exist_ok=True)
        for layer_name, act in model.activations.items():
            filename = f"{save_name}_{layer_name.replace('.', '_')}.pt"
            save_path = os.path.join(save_dir, filename)
            torch.save(act.cpu(), save_path)
            print(f"Saved activation: {save_path}")


# Optional: run as script
if __name__ == "__main__":
    from scripts.capsnet.model import CapsNetRNN
    run_inference(
        model=CapsNetRNN(),
        nii_path="data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz",
        save_dir="output/capsnet",
        save_name="module_test",  
        window=5,
        stride=3,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )
