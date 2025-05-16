import torch
from torchsummary import summary
from scripts.model import MCADNNet

model = MCADNNet().to("cpu")  # ✅ 強制放回 CPU
summary(model, input_size=(1, 64, 64), device="cpu")
