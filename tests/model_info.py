import torch
from torchsummary import summary
from scripts.capsnet.model import CapsNetRNN

model = CapsNetRNN().to("cpu")  # ✅ 強制放回 CPU
summary(model, input_size=(1, 1, 91, 91, 109), device="cpu")
