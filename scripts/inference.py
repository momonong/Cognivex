import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from scripts.model import MCADNNet
import os

# 🧠 多層 activation 存放區
activations = {}

def get_activation(name):
    """建立 hook 函數，儲存指定層 activation"""
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

def load_model(weights_path, device, extract_activation=False):
    model = MCADNNet(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    if extract_activation:
        model.conv2.register_forward_hook(get_activation("conv2"))
        model.fc1.register_forward_hook(get_activation("fc1"))

    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # shape: [1, 1, 64, 64]

def predict(image_tensor, model, device):
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_class].item()
    return predicted_class, confidence

def save_activations(output_prefix, subject_id):
    for layer_name, tensor in activations.items():
        output_path = f"output/{output_prefix}_{layer_name}_{subject_id}.npy"
        np.save(output_path, tensor.numpy())
        print(f"Saved {layer_name} activation to: {output_path}")

def extract_subject_id(image_path):
    base = os.path.basename(image_path)
    subject_id = base.replace(".png", "").replace("sub-", "")
    return subject_id

def main():
    parser = argparse.ArgumentParser(description="Run inference and extract activations")
    parser.add_argument("--image", type=str, required=True, help="Path to the input PNG image")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--extract-activation", action="store_true", help="Extract and save intermediate activations")
    parser.add_argument("--activation-output", type=str, default="activation", help="Prefix for activation file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    model = load_model(args.weights, device, extract_activation=args.extract_activation)
    image_tensor = preprocess_image(args.image)
    predicted_class, confidence = predict(image_tensor, model, device)

    label_map = {0: "AD", 1: "NC"}
    print(f"Predicted class: {label_map[predicted_class]} ({confidence*100:.2f}%)")

    if args.extract_activation:
        if len(activations) == 0:
            print("⚠️ No activation captured. Check layer names or model structure.")
        else:
            subject_id = extract_subject_id(args.image)
            save_activations(args.activation_output, subject_id)

if __name__ == "__main__":
    main()