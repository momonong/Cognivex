import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from scripts.macadnnet.model import MCADNNet  # ← 你的模型路徑

# ---------- 參數 ----------
IMAGE_PATH = "data/slices_from_nii/AD/sub-14/sub-14_z024_t156.png"
WEIGHTS_PATH = "model/macadnnet/best_overall_model.pth"
TARGET_LAYER = "conv2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 前處理 ----------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

original_image = Image.open(IMAGE_PATH).convert("L").resize((64, 64))
img_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

# ---------- 模型與 Hook ----------
model = MCADNNet(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

activations = {}
gradients = {}

def save_activation(module, input, output):
    activations["value"] = output.detach()

def save_gradient(module, grad_input, grad_output):
    gradients["value"] = grad_output[0].detach()

getattr(model, TARGET_LAYER).register_forward_hook(save_activation)
getattr(model, TARGET_LAYER).register_full_backward_hook(save_gradient)

# ---------- Forward + Backward ----------
output = model(img_tensor)
pred = output.argmax(dim=1).item()
score = output[0, pred]
model.zero_grad()
score.backward()

# ---------- GradCAM 熱圖 ----------
act = activations["value"].squeeze(0)  # [C, H, W]
grad = gradients["value"].squeeze(0)
weights = grad.mean(dim=(1, 2))
cam = torch.zeros(act.shape[1:], dtype=torch.float32).to(DEVICE)
for i, w in enumerate(weights):
    cam += w * act[i]
cam = F.relu(cam)
cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)
cam_np = cam.cpu().numpy()

# ---------- 上採樣 & 疊圖 ----------
heatmap_resized = cv2.resize(cam_np, original_image.size[::-1])  # (W, H)
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
original_np = np.array(original_image)
original_rgb = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
overlay = cv2.addWeighted(original_rgb, 0.5, heatmap_color, 0.5, 0)

# ---------- 畫圖 ----------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("GradCAM Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.savefig("figures/gradcam_overlay_aligned.png")
