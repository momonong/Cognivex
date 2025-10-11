import torch
from scripts.capsnet.model import CapsNetRNN
import torch.nn as nn
import torch.nn.functional as F
import torch

# (為了方便測試，我先把您的模型 class 貼在這裡)
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
        self.input_size = (1, 1, 91, 91, 109)
    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.activations['conv3'] = output.detach()
        self.capsnet.conv3.register_forward_hook(hook_fn)
    def forward(self, x):
        if x.dim() == 5:
            x = x.unsqueeze(2)
            B, C, T, D, H, W = x.size()
        elif x.dim() == 6:
            B, C, T, D, H, W = x.size()
        else:
            raise ValueError(f"Expected input shape [B, 1, T, D, H, W] or [B, 1, D, H, W], but got {x.size()}")
        feats = []
        for t in range(T):
            x_t = x[:, :, t, :, :, :]
            caps_out = self.capsnet(x_t)
            feats.append(caps_out.view(B, -1))
        feats = torch.stack(feats, dim=1)
        rnn_out, _ = self.rnn(feats)
        out = self.fc(rnn_out[:, -1, :])
        return torch.sigmoid(out).squeeze(1)


# --- 主程式 ---
if __name__ == '__main__':
    # 1. 實例化您的模型
    model = CapsNetRNN()
    model.eval() # 設定為評估模式

    # 2. 建立一個符合模型輸入尺寸的【假影像】
    # 尺寸為 [Batch, Channels, Time, Depth, Height, Width]
    # 我們用 T=1 來觸發一次 hook 即可
    # 注意：這裡的 D,H,W 尺寸需要是模型能夠接受的，我們根據 conv1 的註解推算
    # 假設原始影像是 91x91x109
    dummy_input = torch.randn(1, 1, 1, 91, 109, 91) # 假設 D,H,W 順序

    # 3. 執行一次前向傳播 (Forward Pass)
    print("正在執行一次前向傳播來觸發 hook...")
    with torch.no_grad():
        _ = model(dummy_input)

    # 4. 檢查儲存下來的激活圖形狀
    if 'conv3' in model.activations:
        conv3_activation = model.activations['conv3']
        activation_shape = conv3_activation.shape
        print(f"\n成功捕捉到 conv3 層的輸出！")
        print(f"完整形狀 (B, C, D, H, W): {activation_shape}")
        
        # 提取我們需要的空間維度
        spatial_dimensions = tuple(activation_shape[2:])
        print(f"\n✅ 您要找的 YOUR_MODEL_ACTIVATION_SHAPE 是: {spatial_dimensions}")
    else:
        print("\n錯誤：沒有捕捉到 conv3 的激活圖。")