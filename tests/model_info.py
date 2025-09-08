import torch
from torchsummary import summary
from scripts.capsnet.model import CapsNetRNN
from scripts.macadnnet.model import MCADNNet

model = CapsNetRNN().to("cpu")  # ✅ 強制放回 CPU
summary(model, input_size=model.input_size, device="cpu")

model_1 = MCADNNet().to("cpu")  # ✅ 強制放回 CPU
summary(model_1, input_size=(1, 64, 64), device="cpu")
