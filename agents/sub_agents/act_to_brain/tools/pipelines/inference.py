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
        clip = data[i : i + window]  # [T, D, H, W]
        clips.append(clip)

    arr = np.stack(clips)  # [B, T, D, H, W]
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)  # ➜ [B, 1, T, D, H, W]
    return tensor


def run_nii_inference(
    model: torch.nn.Module,
    nii_path: str,
    save_dir: str,
    save_name: str,
    selected_layer_names: list[str],
    window: int = 5,
    stride: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
,
):
    model = model.to(device)
    model.eval()

    inputs = load_nifti_and_preprocess(nii_path, window, stride).to(device)
    # print(f"Loaded input shape: {inputs.shape}")

    with torch.no_grad():
        outputs = model(inputs)
        preds = (outputs > 0.5).float().squeeze().cpu().numpy()

    final_pred = int(np.round(preds.mean()))
    # print(f"Inference Result: {final_pred} (1=AD, 0=CN)")

    if hasattr(model, "activations") and isinstance(model.activations, dict):
        os.makedirs(save_dir, exist_ok=True)
        for layer_name in selected_layer_names:
            if layer_name in model.activations:
                act = model.activations[layer_name]
                filename = f"{save_name}_{layer_name.replace('.', '_')}.pt"
                save_path = os.path.join(save_dir, filename)
                torch.save(act.cpu(), save_path)
                # print(f"Saved activation: {save_path}")
    
    # Save final prediction result
    prediction_result = "AD" if final_pred == 1 else "CN"
    return prediction_result


# Optional: run as script
if __name__ == "__main__":
    import torch
    from scripts.capsnet.model import CapsNetRNN
    from agents.sub_agents.act_to_brain.tools.pipelines.attach_hook import attach_hooks

    # [1] 初始化模型並載入權重
    model = CapsNetRNN()
    model.load_state_dict(
        torch.load("model/capsnet/best_capsnet_rnn.pth", map_location="cpu")
    )

    # [2] 指定要 hook 的層名稱（module path）
    selected_layer_names = ["capsnet.conv3", "capsnet.caps2"]
    
    # [3] 建立 activation dict 並掛 hook
    activation_dict = {}
    attach_hooks(model, selected_layer_names, activation_dict)

    # [4] 將 activation dict 附加到 model（run_inference 會使用）
    model.activations = activation_dict

    # [5] 執行推論，並只儲存指定層 activation
    run_nii_inference(
        model=model,
        nii_path="data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz",
        save_dir="output/capsnet",
        save_name="module_test",
        selected_layer_names=selected_layer_names,  
        window=5,
        stride=3,
        device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
