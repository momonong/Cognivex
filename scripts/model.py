import torch
import torch.nn as nn
import torch.nn.functional as F

class MCADNNet(nn.Module):
    def __init__(self, num_classes=2, input_shape=(1, 64, 64)):
        super(MCADNNet, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 🧠 利用 dummy input 自動推算 flatten size
        self._flatten_dim = self._get_flatten_dim(input_shape)

        self.fc1 = nn.Linear(in_features=self._flatten_dim, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=num_classes)

    def _get_flatten_dim(self, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)  # e.g., [1, 1, 64, 64]
            x = self.pool0(F.relu(self.conv0(dummy)))
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool0(F.relu(self.conv0(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
