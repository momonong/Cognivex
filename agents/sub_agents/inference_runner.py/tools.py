import torch
import nibabel as nib
import numpy as np
import importlib
import os


def run_inference_and_save_activation(
    model_path: str,
    model_class: str,
    model_module: str,
    nii_path: str,
    layer_name: str,
    window: int = 5,
    stride: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
):
    # Import model dynamically
    model_mod = importlib.import_module(model_module)
    model_cls = getattr(model_mod, model_class)
    model = model_cls().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load NIfTI and preprocess
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    data = np.transpose(data, (3, 2, 0, 1))
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    clips = [
        data[i:i + window]
        for i in range(0, data.shape[0] - window + 1, stride)
    ]
    inputs = torch.tensor(np.stack(clips), dtype=torch.float32).unsqueeze(1).to(device)

    # Forward
    with torch.no_grad():
        outputs = model(inputs)
        preds = (outputs > 0.5).float().squeeze().cpu().numpy()
        final_pred = int(np.round(preds.mean()))

    # Get activation
    activation = model.activations.get(layer_name)
    if activation is None:
        raise ValueError(f"Activation not found for layer '{layer_name}'")

    # Save activation
    subject_id = os.path.basename(nii_path).split("_")[0]  # like 'sub-14'
    save_path = f"output/activations/{subject_id}_{layer_name}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(activation.cpu(), save_path)

    return {
        "pred": final_pred,
        "activation_path": save_path,
        "subject_id": subject_id,
        "shape": tuple(activation.shape)
    }
