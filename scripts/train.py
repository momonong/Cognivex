import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from scripts.model import MCADNNet  # 確保你 model.py 中的類別名稱正確

# ----- 1. Device Setting -----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ----- 2. Hyperparameters -----
batch_size = 32
epochs = 10
learning_rate = 0.001
image_size = (64, 64)
num_classes = 2

data_dir = "data/images"  # 確保這個資料夾存在，且子資料夾為 CN/AD/MCI

# ----- 3. Data Loading -----
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----- 4. Model / Loss / Optimizer -----
model = MCADNNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----- 5. Training Loop -----
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

# ----- 6. Save Model -----
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "model/mcadnnet_mps.pth")
print("Model saved to model/mcadnnet_mps.pth")
