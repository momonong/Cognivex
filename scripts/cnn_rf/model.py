import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=3):
        super(Simple3DCNN, self).__init__()
        
        # 你的輸入是 (Batch, 3, D, H, W)
        # 3 個通道對應 GM, FA, MD
        
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling
        # 這層很重要，不管輸入影像尺寸多大，它都會把特徵壓縮成 1x1x1
        # 這樣我們就不怕 MNI 模板尺寸些微變動的問題
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5), # 防止過擬合的關鍵
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [Batch, 3, D, H, W]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.gap(x) # -> [Batch, 128, 1, 1, 1]
        x = x.view(x.size(0), -1) # Flatten -> [Batch, 128]
        
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # 測試模型結構
    # 假設 MNI 1mm 模板大約是 182x218x182，我們模擬一個輸入
    dummy_input = torch.randn(2, 3, 182, 218, 182)
    model = Simple3DCNN(num_classes=3)
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}") # 應該是 [2, 3]