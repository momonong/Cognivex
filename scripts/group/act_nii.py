import os
import glob
import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F

paths = sorted(glob.glob("output/group/activations/*.pt"))


for path in paths:
    subject_id = [os.path.basename(path).split("_")[0]][0]

    # === Step 1. 讀取 conv3 activation tensor ===
    act = torch.load(path)[0]  # shape: [C, D, H, W]
    print(f"[原始 activation] shape: {act.shape}  # [channels, depth, height, width]")

    # === Step 2. 選取最強 activation channel ===
    channel_idx = torch.norm(act, p=2, dim=(1, 2, 3)).argmax()
    act = act[channel_idx]  # shape: [D, H, W]
    print(f"[選擇 channel {channel_idx}] shape: {act.shape}  # [depth, height, width]")

    # === Step 3. 準備目標影像參考 shape（對齊原始 fMRI）===
    original_path = glob.glob(f"data/raw/AD/{subject_id}/*.nii.gz")[0]
    ref_img = nib.load(original_path)
    affine = ref_img.affine
    ref_shape = ref_img.shape[:3]
    print(f"[參考影像 NIfTI] path: {original_path}")
    print(f"[參考影像 NIfTI] shape: {ref_shape}  # [X, Y, Z]")

    # === Step 4. 將 activation tensor 改成 [1,1,D,H,W]，以便使用 F.interpolate ===
    act_tensor = act.unsqueeze(0).unsqueeze(0)  # shape: [1,1,D,H,W]
    print(f"[加維度以便 interpolate] shape: {act_tensor.shape}")

    # === Step 5. 使用 trilinear interpolation 調整到 ref_shape ===
    # 注意：順序必須是 Z, X, Y → PyTorch 內部是 D, H, W
    resized = F.interpolate(
        act_tensor,
        size=(ref_shape[2], ref_shape[0], ref_shape[1]),  # [Z, X, Y]
        mode="trilinear",
        align_corners=False,
    )  # shape: [1, 1, Z, X, Y]
    print(f"[resized activation] shape: {resized.shape}")

    # === Step 6. squeeze 回去 numpy array，然後轉回正確順序 ===
    resized = resized.squeeze().numpy()  # shape: [Z, X, Y]
    nii_data = np.transpose(resized, (1, 2, 0))  # shape: [X, Y, Z]
    print(f"[轉成 NIfTI] 最終 shape: {nii_data.shape}（X, Y, Z）")

    # === Step 7. Normalize 並選取 top percentile 當作 activation mask ===
    nii_data = (nii_data - nii_data.min()) / (nii_data.max() - nii_data.min() + 1e-8)
    threshold = np.percentile(nii_data, 99)
    nii_masked = np.where(nii_data >= threshold, nii_data, 0).astype(np.float32)

    # === Step 8. 存成 NIfTI 檔案 ===
    save_path = f"output/group/nifti/{subject_id}_masked.nii.gz"
    nib.save(
        nib.Nifti1Image(nii_masked, affine),
        save_path,
    )
    print("✅ Saved filtered, strongest channel activation.")
    print("path:", save_path)
