import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from scripts.macadnnet.model import MCADNNet
import os
from pathlib import Path

# ----------- 設定參數 ----------
IMAGE_PATH = "data/slices/fold0/AD/AD_sub-07_z050_t003.png"
WEIGHTS_PATH = "model/mcadnnet_fold.pth"
OUTPUT_PREFIX = "conv2AD"
OUTPUT_DIR = "output"
EXTRACT_ACTIVATION = True

# ----------- 儲存 activation --------
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

# ----------- 模型載入 -------------
def load_model(weights_path, device, extract_activation=False):
    model = MCADNNet(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    if extract_activation:
        model.conv2.register_forward_hook(get_activation("conv2"))
        model.fc1.register_forward_hook(get_activation("fc1"))
    return model

# ----------- 圖像處理 -------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# ----------- 預測 ----------------
def predict(image_tensor, model, device):
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_class].item()
    return predicted_class, confidence

# ----------- 儲存 activation ------
def save_activations(output_prefix, subject_id, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    for layer_name, tensor in activations.items():
        save_path = os.path.join(output_dir, f"{output_prefix}_{layer_name}_{subject_id}.npy")
        np.save(save_path, tensor.numpy())
        print(f"✅ Saved activation to: {save_path}")

# ----------- 擷取 ID -------------
def extract_subject_id(image_path):
    filename = Path(image_path).stem
    parts = filename.split("_")
    for p in parts:
        if p.startswith("sub-"):
            return p
    return "unknown"

# ----------- 主流程 ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🧠 Device: {device}")

    model = load_model(WEIGHTS_PATH, device, extract_activation=EXTRACT_ACTIVATION)
    image_tensor = preprocess_image(IMAGE_PATH)
    predicted_class, confidence = predict(image_tensor, model, device)

    label_map = {0: "AD", 1: "CN"}
    print(f"🧠 Prediction: {label_map[predicted_class]} ({confidence * 100:.2f}%)")

    if EXTRACT_ACTIVATION:
        if not activations:
            print("⚠️ No activation captured.")
        else:
            subject_id = extract_subject_id(IMAGE_PATH)
            save_activations(OUTPUT_PREFIX, subject_id, output_dir=OUTPUT_DIR)
