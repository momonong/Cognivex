import torch
import torch.nn as nn

class Simple3DCNN_InstanceNorm(nn.Module):
    """
    模型定義必須與訓練時完全相同，才能載入權重。
    使用 InstanceNorm3d 替代 BatchNorm3d。
    """
    def __init__(self, in_channels=1, num_classes=2):
        super(Simple3DCNN_InstanceNorm, self).__init__()
        
        def create_conv_block(in_c, out_c, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=padding),
                nn.InstanceNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2)
            )
        
        # Patch 128 -> 64
        self.block1 = create_conv_block(in_channels, 16)
        # Patch 64 -> 32
        self.block2 = create_conv_block(16, 32)
        # Patch 32 -> 16
        self.block3 = create_conv_block(32, 64)
        # Patch 16 -> 8
        self.block4 = create_conv_block(64, 128)
        
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x