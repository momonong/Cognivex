import os
import numpy as np
import nibabel as nib
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from scripts.macadnnet.model import MCADNNet

# ---------- 參數 ----------
SUBJECT_ID = "sub-14"
Z = 24
T = 156
LABEL = "AD"
IMAGE_PATH = f"data/slices_from_nii/{LABEL}/{SUBJECT_ID}/{SUBJECT_ID}_z{Z:03d}_t{T:03d}.png"
NII_PATH = f"data/raw/{LABEL}/{SUBJECT_ID}/dswausub-098_S_6601_task-rest_bold.nii"
WEIGHTS_PATH = "model/macadnnet/best_overall_model.pth"
OUTPUT_PATH = f"output/nifti_gradcam/{SUBJECT_ID}_z{Z}_t{T}_gradcam.nii.gz"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_LAYER = "conv2"

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------- 模型與 hook ----------
model = MCADNNet(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

activations, gradients = {}, {}
def save_activation(module, input, output):
    activations["value"] = output.detach()

def save_gradient(module, grad_input, grad_output):
    gradients["value"] = grad_output[0].detach()

getattr(model, TARGET_LAYER).register_forward_hook(save_activation)
getattr(model, TARGET_LAYER).register_full_backward_hook(save_gradient)

# ---------- 載入圖片 ----------
img = Image.open(IMAGE_PATH).convert("L")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ---------- Forward & Backward ----------
output = model(img_tensor)
class_idx = output.argmax(dim=1).item()
score = output[0, class_idx]
model.zero_grad()
score.backward()

# ---------- GradCAM heatmap ----------
act = activations["value"].squeeze(0)  # [C, H, W]
grad = gradients["value"].squeeze(0)   # [C, H, W]
weights = grad.mean(dim=(1, 2))
cam = torch.zeros_like(act[0])

for i, w in enumerate(weights):
    cam += w * act[i]

cam = F.relu(cam)
cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)
cam_np = cam.cpu().numpy()  # shape: [H, W], 即 [9, 9]

# ---------- 疊圖到原始 NIfTI ----------
nii_img = nib.load(NII_PATH)
nii_data = nii_img.get_fdata()
affine = nii_img.affine
nii_shape = nii_data.shape  # [X, Y, Z, T]

# 建立空白 volume
gradcam_vol = np.zeros_like(nii_data, dtype=np.float32)

# 對應中心填入 cam
x_center, y_center = nii_shape[0] // 2, nii_shape[1] // 2
x_start, x_end = x_center - 4, x_center + 5
y_start, y_end = y_center - 4, y_center + 5

# 填入 GradCAM 到 [x, y, z, t]
gradcam_vol[x_start:x_end, y_start:y_end, Z, T] = cam_np

# ---------- 儲存 NIfTI ----------
nib.save(nib.Nifti1Image(gradcam_vol, affine), OUTPUT_PATH)
print(f"✅ GradCAM NIfTI saved: {OUTPUT_PATH}")
print(f"📦 Max value in saved NIfTI: {gradcam_vol.max():.6f}")
