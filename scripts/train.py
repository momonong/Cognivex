import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scripts.model import MCADNNet

# ---------- 1. Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ---------- 2. Config ----------
image_size = (64, 64)
batch_size = 32
epochs = 5
learning_rate = 0.001
num_classes = 2
meta_csv = "data/slices_metadata.csv"
final_model_path = "model/best_overall_model.pth"
os.makedirs("model", exist_ok=True)

# ---------- 3. Transform ----------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------- 4. Custom Dataset ----------
class SliceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {"CN": 0, "AD": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("L")
        label = self.label_map[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------- 5. Load Metadata ----------
df = pd.read_csv(meta_csv)

best_overall_acc = 0.0
best_overall_state_dict = None
best_overall_fold = -1

# ---------- 6. Cross-validation ----------
for val_fold in range(5):
    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]

    train_dataset = SliceDataset(train_df, transform)
    val_dataset = SliceDataset(val_df, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MCADNNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    best_state_dict = None

    print(f"\n🚀 Training Fold {val_fold}...")

    for epoch in range(epochs):
        # ----- Training -----
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100. * correct / total

        # ----- Validation -----
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_acc = accuracy_score(val_labels, val_preds) * 100
        epoch_val_precision = precision_score(val_labels, val_preds, zero_division=0)
        epoch_val_recall = recall_score(val_labels, val_preds, zero_division=0)
        epoch_val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        print(f"[Fold {val_fold} | Epoch {epoch+1}] "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {epoch_val_acc:.2f}% | "
              f"P: {epoch_val_precision:.2f} | R: {epoch_val_recall:.2f} | F1: {epoch_val_f1:.2f}")

        # ----- 儲存最佳模型 -----
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state_dict = model.state_dict()

    print(f"✅ Fold {val_fold} Best Val Acc: {best_val_acc:.2f}%")
    if best_val_acc > best_overall_acc:
        best_overall_acc = best_val_acc
        best_overall_state_dict = best_state_dict
        best_overall_fold = val_fold

# ---------- 7. Save best model overall ----------
if best_overall_state_dict:
    torch.save(best_overall_state_dict, final_model_path)
    print(f"\n🏆 最佳模型來自 Fold {best_overall_fold}，驗證準確率為 {best_overall_acc:.2f}%")
    print(f"✅ 已儲存最佳模型到：{final_model_path}")
else:
    print("❌ 沒有模型被儲存，請檢查資料或訓練過程")
