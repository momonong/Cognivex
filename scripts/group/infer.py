import glob
import torch
import nibabel as nib
import numpy as np
from scripts.capsnet.model import CapsNetRNN
import os


# ---------- 參數 ----------
MODEL_PATH = "model/capsnet/best_capsnet_rnn.pth"
# NII_PATH = "data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz"  # 要推論的影像路徑
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
WINDOW = 5
STRIDE = 3

# ---------- 載入模型 ----------
model = CapsNetRNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

file_list = sorted(glob.glob("data/raw/AD/*/*.nii.gz"))


for nii_path in file_list:
    # ---------- 預處理 NIfTI ----------
    nii = nib.load(nii_path)
    data = nii.get_fdata()  # shape: [X, Y, Z, T]
    data = np.transpose(data, (3, 2, 0, 1))  # [T, Z, H, W] → 視窗取樣

    # Normalize to [0, 1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    # 滑動視窗：每個視窗大小 = WINDOW
    clips = []
    for i in range(0, data.shape[0] - WINDOW + 1, STRIDE):
        clip = data[i : i + WINDOW]  # shape: [window, Z, H, W]
        clips.append(clip)

    inputs = (
        torch.tensor(np.stack(clips), dtype=torch.float32).unsqueeze(1).to(DEVICE)
    )  # [N, 1, W, Z, H, W]
    print(f"🔍 Loaded input shape: {inputs.shape}")

    # ---------- 推論 ----------
    with torch.no_grad():
        outputs = model(inputs)
        preds = (outputs > 0.5).float().squeeze().cpu().numpy()

    # ---------- 結果 ----------
    final_pred = int(np.round(preds.mean()))
    print(f"🧠 Inference Result: {final_pred} (1=AD, 0=CN)")

    # 推論後儲存 activation
    act = model.activations["conv3"]  # [B, C, D, H, W]
    subject_id = os.path.basename(os.path.dirname(nii_path))  # sub-14
    save_path = f"output/group/activations/{subject_id}_conv3.pt"
    torch.save(act.cpu(), save_path)
    print(f"💾 Saved activation to {save_path}")
