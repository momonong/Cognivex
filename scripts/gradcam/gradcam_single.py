import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from scripts.macadnnet.model import MCADNNet  # ← 你的模型路徑

# ---------- 參數 ----------
IMAGE_PATH = "data/slices_from_nii/AD/sub-14/sub-14_z024_t156.png"  # ← 測試用切片
WEIGHTS_PATH = "model/macadnnet/best_overall_model.pth"
TARGET_LAYER = "conv2"  # 可以改成 conv0, conv1, conv2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 前處理 ----------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------- 建立模型與 hook ----------
model = MCADNNet(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

activations = {}
gradients = {}

def save_activation(module, input, output):
    activations["value"] = output.detach()

def save_gradient(module, grad_input, grad_output):
    gradients["value"] = grad_output[0].detach()

# 註冊 hook（針對 conv2）
target_module = getattr(model, TARGET_LAYER)
target_module.register_forward_hook(save_activation)
target_module.register_full_backward_hook(save_gradient)

# ---------- 讀入影像 ----------
img = Image.open(IMAGE_PATH).convert("L")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ---------- Forward + backward ----------
output = model(img_tensor)
class_idx = output.argmax(dim=1).item()
score = output[0, class_idx]
model.zero_grad()
score.backward()

# ---------- 產生 GradCAM heatmap ----------
act = activations["value"].squeeze(0)        # shape: [C, H, W]
grad = gradients["value"].squeeze(0)         # shape: [C, H, W]

weights = grad.mean(dim=(1, 2))              # Global average pooling
cam = torch.zeros(act.shape[1:], dtype=torch.float32).to(DEVICE)  # shape: [H, W]

for i, w in enumerate(weights):
    cam += w * act[i]

cam = F.relu(cam)                            # ReLU
cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)               # Normalize to 0~1
cam = cam.cpu().numpy()

# ---------- 畫圖 ----------
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("GradCAM Heatmap")
plt.imshow(img, cmap='gray')
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.axis("off")

plt.tight_layout()
plt.savefig("figures/gradcam_output.png")
