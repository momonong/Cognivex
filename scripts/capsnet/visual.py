import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_activation(act_path, out_dir='activation_images', slice_axis=2, max_slices=10):
    os.makedirs(out_dir, exist_ok=True)

    # 讀取 activation tensor
    act = torch.load(act_path)  # shape: [B, C, D, H, W]
    if act.dim() == 5:
        act = act[0]  # 取第 1 個樣本 [C, D, H, W]
    elif act.dim() == 4:
        pass  # 已經是 [C, D, H, W]
    else:
        raise ValueError(f"Unsupported activation shape: {act.shape}")

    print(f"Loaded activation shape: {act.shape}")
    C, D, H, W = act.shape

    # 選擇一個 channel 來畫圖（可自訂）
    for ch in range(min(C, 3)):  # 只畫前幾個 channel
        vol = act[ch].cpu().numpy()  # shape: [D, H, W]

        # 根據 axis 選切片方向
        if slice_axis == 0:
            slices = vol[:max_slices]
        elif slice_axis == 1:
            slices = vol.transpose(1, 0, 2)[:max_slices]
        elif slice_axis == 2:
            slices = vol.transpose(1, 2, 0)[:, :, :max_slices]
        else:
            raise ValueError("Invalid slice_axis. Must be 0, 1, or 2.")

        for i in range(slices.shape[-1]):
            plt.imshow(slices[:, :, i], cmap='hot')
            plt.title(f'Channel {ch} - Slice {i}')
            plt.axis('off')
            plt.savefig(os.path.join(out_dir, f'channel{ch}_slice{i}.png'), bbox_inches='tight')
            plt.close()
        print(f"✅ Channel {ch} visualized and saved.")

if __name__ == "__main__":
    # 路徑可依你推論儲存的位置修改
    act_path = "output/capsnet/sub-14_conv3.pt"
    out_dir = "figures/capsnet/sub-14"
    visualize_activation(act_path, out_dir, slice_axis=1, max_slices=10)
