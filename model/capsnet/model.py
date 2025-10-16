import torch
import torch.nn as nn
import torch.nn.functional as F


class Squash(nn.Module):
    def forward(self, s):
        norm = torch.norm(s, dim=-1, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (s / (norm + 1e-8))


class CapsuleLayer3D(nn.Module):
    def __init__(
        self, in_caps, out_caps, in_dim, out_dim, kernel_size=3, stride=2, padding=1
    ):
        super().__init__()
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.conv = nn.Conv3d(
            in_caps * in_dim, out_caps * out_dim, kernel_size, stride, padding
        )
        self.squash = Squash()

    def forward(self, x):
        # x: [B, in_caps, in_dim, D, H, W]
        B, in_caps, in_dim, D, H, W = x.size()
        x = x.view(B, in_caps * in_dim, D, H, W)
        x = self.conv(x)  # [B, out_caps*out_dim, d, h, w]
        d, h, w = x.shape[2:]
        x = x.view(B, self.out_caps, self.out_dim, d, h, w)
        x = self.squash(x)
        return x


class CapsNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        # 3D卷積
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=5, stride=2, padding=0)
        # 膠囊層
        self.caps1 = CapsuleLayer3D(1, 32, 64, 8)
        self.caps2 = CapsuleLayer3D(32, 32, 8, 8)
        # 最終膠囊層（非卷積）
        self.final_caps = nn.Linear(32 * 8, 2 * 16)
        self.squash = Squash()

    def forward(self, x):
        # x: [B, 1, 61, 73, 61]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 膠囊層
        x = x.unsqueeze(1)  # [B, 1, 64, d, h, w]
        x = self.caps1(x)
        x = self.caps2(x)
        # Global Average Pooling
        x = x.mean(dim=[3, 4, 5])  # [B, 32, 8]
        x = x.view(x.size(0), -1)  # [B, 32*8]
        x = self.final_caps(x)  # [B, 2*16]
        x = x.view(x.size(0), 2, 16)
        x = self.squash(x)
        return x  # [B, 2, 16]


class CapsNetRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.capsnet = CapsNet3D()
        self.rnn = nn.RNN(input_size=32, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.activations = {}
        self._register_hooks()
        self.input_size = (1, 1, 91, 91, 109)  # For reference

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.activations['conv3'] = output.detach()

        # 例如你想抓 Conv3d-3，對應的是 self.capsnet.conv3
        self.capsnet.conv3.register_forward_hook(hook_fn)

    def forward(self, x):
        # x: [B, 1, T, D, H, W] ← 4D fMRI 資料
        if x.dim() == 5: # Handle 5D input as a single time step
            x = x.unsqueeze(2) # Add time dimension T=1 at index 2
            B, C, T, D, H, W = x.size() # Recalculate dimensions
        elif x.dim() == 6:
            B, C, T, D, H, W = x.size()
        else:
            raise ValueError(f"Expected input shape [B, 1, T, D, H, W] or [B, 1, D, H, W], but got {x.size()}")

        feats = []
        for t in range(T):
            # 取出第 t 個時間點的 volume
            x_t = x[:, :, t, :, :, :].to(next(self.capsnet.parameters()).device)          # [B, 1, D, H, W]
            caps_out = self.capsnet(x_t)       # [B, 2, 16]
            feats.append(caps_out.view(B, -1)) # [B, 32]

        feats = torch.stack(feats, dim=1)  # [B, T, 32]
        rnn_out, _ = self.rnn(feats)       # [B, T, 64]
        out = self.fc(rnn_out[:, -1, :])   # [B, 1]
        return torch.sigmoid(out).squeeze(1)
