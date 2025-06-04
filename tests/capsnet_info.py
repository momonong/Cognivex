import torch
from torchsummary import summary
from scripts.capsnet.model import CapsNetRNN

# 模型實例
model = CapsNetRNN()

# 正確的輸入形狀：C=1, T=5, D=64, H=64, W=64 → 即 (1, 5, 64, 64, 64)
# 注意 torchsummary 的 input_size 是不含 batch 的，所以只給 [C, T, D, H, W]
summary(model, input_size=(1, 5, 64, 64, 64), device="cpu")
