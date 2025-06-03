# scripts/gradcam_utils.py

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from scripts.macadnnet.model import MCADNNet  # 請確認此路徑正確

# ---------- 參數 ----------
WEIGHTS_PATH = "model/macadnnet/best_overall_model.pth"
TARGET_LAYER = "conv2"
DEVICE = torch.device("mps" if torch.cuda.is_available() else "cpu")

# ---------- 載入模型 ----------
model = MCADNNet(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

# ---------- GradCAM 產生器 ----------
def generate_gradcam_for_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    activations = {}
    gradients = {}

    def save_activation(module, input, output):
        activations["value"] = output.detach()

    def save_gradient(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    target_module = getattr(model, TARGET_LAYER)
    target_module.register_forward_hook(save_activation)
    target_module.register_full_backward_hook(save_gradient)

    img = Image.open(image_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    output = model(img_tensor)
    pred = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred].backward()

    act = activations["value"].squeeze(0)
    grad = gradients["value"].squeeze(0)
    weights = grad.mean(dim=(1, 2))
    cam = torch.zeros(act.shape[1:], dtype=torch.float32).to(DEVICE)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    return cam.cpu().numpy()  # [H, W]
