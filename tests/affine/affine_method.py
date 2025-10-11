import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
from nilearn import plotting, datasets
from nilearn.image import resample_to_img

# ===============================================================
#  第一部分：貼上您完整的模型定義
# ===============================================================
class Squash(nn.Module):
    def forward(self, s):
        norm = torch.norm(s, dim=-1, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (s / (norm + 1e-8))

class CapsuleLayer3D(nn.Module):
    def __init__(self, in_caps, out_caps, in_dim, out_dim, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.conv = nn.Conv3d(in_caps * in_dim, out_caps * out_dim, kernel_size, stride, padding)
        self.squash = Squash()
    def forward(self, x):
        B, in_caps, in_dim, D, H, W = x.size()
        x = x.view(B, in_caps * in_dim, D, H, W)
        x = self.conv(x)
        d, h, w = x.shape[2:]
        x = x.view(B, self.out_caps, self.out_dim, d, h, w)
        x = self.squash(x)
        return x

class CapsNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=5, stride=2, padding=0)
        self.caps1 = CapsuleLayer3D(1, 32, 64, 8)
        self.caps2 = CapsuleLayer3D(32, 32, 8, 8)
        self.final_caps = nn.Linear(32 * 8, 2 * 16)
        self.squash = Squash()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.unsqueeze(1)
        x = self.caps1(x)
        x = self.caps2(x)
        x = x.mean(dim=[3, 4, 5])
        x = x.view(x.size(0), -1)
        x = self.final_caps(x)
        x = x.view(x.size(0), 2, 16)
        x = self.squash(x)
        return x

class CapsNetRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.capsnet = CapsNet3D()
        self.rnn = nn.RNN(input_size=32, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.activations = {}
        self._register_hooks()
    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.activations['conv3'] = output.detach()
        self.capsnet.conv3.register_forward_hook(hook_fn)
    def forward(self, x):
        if x.dim() == 5: x = x.unsqueeze(2)
        B, C, T, D, H, W = x.size()
        feats = []
        for t in range(T):
            x_t = x[:, :, t, :, :, :]
            caps_out = self.capsnet(x_t)
            feats.append(caps_out.view(B, -1))
        feats = torch.stack(feats, dim=1)
        rnn_out, _ = self.rnn(feats)
        out = self.fc(rnn_out[:, -1, :])
        return torch.sigmoid(out).squeeze(1)

# ===============================================================
#  第二部分：貼上我們驗證過的輔助函式
# ===============================================================
def select_strongest_channel(act: torch.Tensor, norm_type: str = "l2") -> int:
    if act.ndim == 3: return 0
    if norm_type == "l2":
        norms = torch.norm(act, p=2, dim=(1, 2, 3))
    elif norm_type == "l1":
        norms = torch.norm(act, p=1, dim=(1, 2, 3))
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")
    print(f"-> 在 {act.shape[0]} 個頻道中，第 {norms.argmax().item()} 個頻道的 L2 Norm 最強。")
    return norms.argmax().item()

def activation_to_nifti_nilearn(
    activation_tensor: torch.Tensor,
    activation_affine: np.ndarray,
    reference_nii_path: str,
    output_path: str,
):
    source_nii = nib.Nifti1Image(activation_tensor.cpu().numpy(), activation_affine)
    reference_nii = nib.load(reference_nii_path)
    resampled_nii = resample_to_img(
        source_img=source_nii,
        target_img=reference_nii,
        interpolation='continuous'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(resampled_nii, output_path)

# ===============================================================
#  第三部分：主執行流程
# ===============================================================
if __name__ == '__main__':
    print("--- 開始執行模型激活圖可視化 ---")
    
    # --- 1. 準備工作 ---
    OUTPUT_DIR = "figures/final_visualization"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("-> 正在下載 MNI 標準大腦模板...")
    mni_template = datasets.load_mni152_template(resolution=2)
    REFERENCE_NII_PATH = os.path.join(OUTPUT_DIR, "mni_template.nii.gz")
    nib.save(mni_template, REFERENCE_NII_PATH)

    # --- 2. 載入模型並捕捉激活圖 ---
    print("-> 正在實例化模型...")
    model = CapsNetRNN()
    model.eval()

    # 建立一個符合模型輸入的假 fMRI 數據
    # 尺寸: [Batch, Channels, Time, Depth, Height, Width]
    # 我們使用一個常見的 fMRI 尺寸作為範例
    dummy_input = torch.randn(1, 1, 1, 91, 109, 91)
    
    print(f"-> 正在用假數據 ({dummy_input.shape}) 進行一次前向傳播...")
    with torch.no_grad():
        _ = model(dummy_input)
    
    # 從 hook 中取出我們捕捉到的激活圖
    full_activation = model.activations['conv3']
    print(f"-> 成功捕捉到 'conv3' 層的激活圖，原始形狀: {full_activation.shape}")

    # --- 3. 處理激活圖 ---
    # 去掉 Batch 維度，得到 [C, D, H, W]
    activation_channels = full_activation[0]
    
    # 選擇最強的那個頻道，得到 [D, H, W]
    channel_idx = select_strongest_channel(activation_channels)
    activation_map = activation_channels[channel_idx]
    print(f"-> 已選取最強頻道，最終激活圖形狀: {activation_map.shape}")

    # --- 4. 計算低解析度激活圖的 Affine ---
    print("-> 正在為低解析度激活圖計算正確的 Affine...")
    high_res_affine = mni_template.affine
    high_res_shape = mni_template.shape
    low_res_shape = activation_map.shape
    
    # 計算每個維度的縮放比例
    scaling_factors = np.array(high_res_shape) / np.array(low_res_shape)
    
    # 建立新的 affine
    activation_affine = high_res_affine.copy()
    for i in range(3):
        activation_affine[i, i] *= scaling_factors[i]
    
    # --- 5. 執行座標轉換 ---
    print("-> 正在執行我們驗證過的 nilearn 轉換函式...")
    FINAL_NII_PATH = os.path.join(OUTPUT_DIR, "model_activation_map.nii.gz")
    activation_to_nifti_nilearn(
        activation_tensor=activation_map,
        activation_affine=activation_affine,
        reference_nii_path=REFERENCE_NII_PATH,
        output_path=FINAL_NII_PATH
    )

    # --- 6. 產生最終的 PNG 疊圖 ---
    print("-> 正在產生最終的 PNG 預覽圖...")
    
    # 為了讓圖像更清晰，我們會過濾掉 95% 以下的低信號
    display = plotting.plot_stat_map(
        FINAL_NII_PATH,
        bg_img=REFERENCE_NII_PATH,
        title="Model Activation Map (conv3, strongest channel)",
        display_mode='ortho',
        colorbar=True,
        threshold=np.percentile(activation_map.numpy(), 95) # 動態計算閾值
    )
    
    FINAL_PNG_PATH = os.path.join(OUTPUT_DIR, "MODEL_ACTIVATION_PREVIEW.png")
    display.savefig(FINAL_PNG_PATH, dpi=300)

    print("\n" + "="*50)
    print("🎉 可視化流程全部完成！")
    print(f"最終的 NIfTI 檔案儲存於: {FINAL_NII_PATH}")
    print(f"最終的 PNG 預覽圖儲存於: {FINAL_PNG_PATH}")
    print("="*50)