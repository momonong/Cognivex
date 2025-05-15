import torch
import torch.nn as nn
import torch.nn.functional as F

class MCADNNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MCADNNet, self).__init__()

        # conv0: input -> [batch, 1, H, W] => output -> [batch, 10, H-4, W-4]
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
        # pool0: [batch, 10, H-4, W-4] => [batch, 10, (H-4)//2, (W-4)//2]
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv1: [batch, 10, H', W'] => [batch, 20, H''-4, W''-4]
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2: [batch, 20, H''', W'''] => [batch, 50, ...]
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # NOTE: in_features depends on input image size, use adaptive layer or calculate manually later
        self.fc1 = nn.Linear(in_features=50*2*2, out_features=500)  # placeholder shape
        self.fc2 = nn.Linear(in_features=500, out_features=num_classes)

    def forward(self, x):
        x = self.pool0(F.relu(self.conv0(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)  # flatten except batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
