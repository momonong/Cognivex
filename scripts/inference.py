import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from scripts.model import MCADNNet


def load_model(weights_path, device):
    model = MCADNNet(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
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


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single fMRI slice")
    parser.add_argument("--image", type=str, required=True, help="Path to the input PNG image")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    args = parser.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    model = load_model(args.weights, device)
    image_tensor = preprocess_image(args.image)
    predicted_class, confidence = predict(image_tensor, model, device)

    label_map = {0: "AD", 1: "NC"}  # 或根據你的實際順序反過來
    print(f"Predicted class: {label_map[predicted_class]} ({confidence*100:.2f}%)")


if __name__ == "__main__":
    main()
