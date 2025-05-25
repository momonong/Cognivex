import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scripts.macadnnet.model import MCADNNet

# ---------- 1. Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ---------- 2. Config ----------
image_size = (64, 64)
batch_size = 32
epochs = 50
learning_rate = 0.001
num_classes = 2
meta_csv = "data/slices_metadata_z30-60.csv"
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
early_stop_epoch_record = {}

# ---------- 6. Cross-validation ----------
for val_fold in range(5):
    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]

    train_dataset = SliceDataset(train_df, transform)
    val_dataset = SliceDataset(val_df, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MCADNNet(num_classes=num_classes, dropout_p=0.5).to(device)

    # Freeze CNN layers
    for name, param in model.named_parameters():
        if "conv" in name:
            param.requires_grad = False

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state_dict = None
    best_metrics = {}
    patience = 3
    no_improve_count = 0

    print(f"\n🚀 Training Fold {val_fold} (Frozen CNN + EarlyStopping)...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = accuracy_score(val_labels, val_preds) * 100
        epoch_val_precision = precision_score(val_labels, val_preds, zero_division=0)
        epoch_val_recall = recall_score(val_labels, val_preds, zero_division=0)
        epoch_val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        print(f"[Fold {val_fold} | Epoch {epoch+1}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}% | "
              f"P: {epoch_val_precision:.2f} | R: {epoch_val_recall:.2f} | F1: {epoch_val_f1:.2f}")

        scheduler.step(avg_val_loss)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state_dict = model.state_dict()
            best_metrics = {
                "fold": val_fold,
                "acc": epoch_val_acc,
                "precision": epoch_val_precision,
                "recall": epoch_val_recall,
                "f1": epoch_val_f1,
                "epoch": epoch + 1
            }
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"⏹️ Early stopping at Epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    torch.save(best_metrics, f"model/fold{val_fold}_metrics.pt")
    early_stop_epoch_record[val_fold] = best_metrics.get("epoch", "N/A")
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

# ---------- 8. Summary Reporting ----------
print("\n📊 每個 Fold 最佳結果統整：")
print("=" * 50)
fold_metrics = []
for i in range(5):
    log_file = f"model/fold{i}_metrics.pt"
    if os.path.exists(log_file):
        metric = torch.load(log_file)
        fold_metrics.append(metric)
        print(f"Fold {i} → "
              f"Val Acc: {metric['acc']:.2f}% | "
              f"P: {metric['precision']:.2f} | "
              f"R: {metric['recall']:.2f} | "
              f"F1: {metric['f1']:.2f} | "
              f"Stopped at Epoch: {metric['epoch']}")
    else:
        print(f"Fold {i} → ❌ 無儲存紀錄")

print("=" * 50)
if fold_metrics:
    best = max(fold_metrics, key=lambda x: x["acc"])
    print(f"\n🏆 最佳模型來自 Fold {best['fold']}：")
    print(f"🔹 Accuracy: {best['acc']:.2f}%")
    print(f"🔹 Precision: {best['precision']:.2f}")
    print(f"🔹 Recall: {best['recall']:.2f}")
    print(f"🔹 F1 Score: {best['f1']:.2f}")
    print(f"🔹 Epoch: {best['epoch']}")
else:
    print("❌ 無法讀取最佳模型評估結果")
