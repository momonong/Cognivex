import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from scripts.capsnet.preprocess import FMRI4DDataset
from scripts.capsnet.model import CapsNetRNN

def train(model, train_loader, val_loader, epochs=20, lr=1e-6, weight_decay=0.1, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    best_val_acc = 0
    os.makedirs("checkpoints", exist_ok=True)

    # Optional: Cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{epochs}] Training", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            pred = (out > 0.5).float()
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = train_correct / train_total
        train_loss /= train_total

        # ---------- 驗證 ----------
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"[Epoch {epoch}] Validation", leave=False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                pred = (out > 0.5).float()
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total

        print(f"📊 Epoch {epoch}: "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} || "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # Save model if best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"checkpoints/best_capsnet_rnn.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved to: {save_path}")

        scheduler.step()  # optional

    print(f"🎯 Best Val Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    # ---------- 準備資料 ----------
    dataset = FMRI4DDataset('data/raw', window=5, stride=3)
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=2)

    # ---------- 模型訓練 ----------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🧠 Using device: {device}")
    model = CapsNetRNN()
    train(model, train_loader, val_loader, epochs=20, lr=1e-6, weight_decay=0.1, device=device)
