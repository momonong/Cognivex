import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
import numpy as np
import nibabel as nib
import os
import glob
import time

# ====================================================================
# 【1. 設定與配置】(🚨 核心修正點)
# ====================================================================
DATA_ROOT = "E:/fMRI/Model/sMRI_data_3Class/" 
MODEL_OUTPUT_DIR = "model/cnn_3d_3class/"     

NUM_CLASSES = 3
NUM_FOLDS = 5
NUM_EPOCHS = 50 
LEARNING_RATE = 0.0001 # 保持 0.0001 (穩定)

# 🚨 核心修正點 V11 (根據您的建議)：
# 1. 物理 BATCH_SIZE 設為 2 (VRAM 安全, 預期 ~11-17GB VRAM)
BATCH_SIZE = 2
# 2. 累積步數設為 8
ACCUMULATION_STEPS = 8
# 3. 虛擬批次大小 (Effective Batch Size) = 2 * 8 = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


# ====================================================================
# 【2. 模型定義 (ResNet 架構)】(保持不變)
# ====================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False) 
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels)
            )
            
    def forward(self, x):
        main_path_output = self.relu(self.norm1(self.conv1(x)))
        main_path_output = self.norm2(self.conv2(main_path_output))
        output = main_path_output + self.shortcut(x)
        output = self.relu(output)
        return output

class ResNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
        super(ResNet3D, self).__init__()
        
        self.initial_conv = nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.initial_norm = nn.InstanceNorm3d(16)
        self.initial_relu = nn.ReLU(inplace=False)
        
        self.block1 = nn.Sequential(
            ResidualBlock(16, 16),
            nn.MaxPool3d(kernel_size=2, stride=2) 
        )
        self.block2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2), 
            ResidualBlock(32, 32)
        )
        self.block3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2), 
            ResidualBlock(64, 64)
        )
        self.block4 = nn.Sequential(
            ResidualBlock(64, 128, stride=2), 
            ResidualBlock(128, 128)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.initial_relu(self.initial_norm(self.initial_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x) 
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ====================================================================
# 【3. PyTorch 資料集載入器 (已修正正規化)】(保持不變)
# ====================================================================
class NiftiDataset(Dataset):
    def __init__(self, data_root):
        self.file_list = []
        self.label_list = []
        self.label_map = {"NC": 0, "MCI": 1, "AD": 2}
        
        print(f"正在掃描資料集: {data_root}")
        
        for label_name, label_id in self.label_map.items():
            class_path = os.path.join(data_root, label_name)
            if not os.path.isdir(class_path):
                print(f"⚠️ 警告：找不到 '{class_path}' 資料夾。")
                continue
            files = glob.glob(os.path.join(class_path, "*.nii.gz")) + \
                    glob.glob(os.path.join(class_path, "*.nii"))
            for file_path in files:
                self.file_list.append(file_path)
                self.label_list.append(label_id)
        
        labels_np = np.array(self.label_list)
        nc_count = np.sum(labels_np == 0)
        mci_count = np.sum(labels_np == 1)
        ad_count = np.sum(labels_np == 2)
        print(f"資料集掃描完成。總共找到 {len(self.file_list)} 筆資料。")
        print(f"   - NC (0): {nc_count} 筆")
        print(f"   - MCI (1): {mci_count} 筆")
        print(f"   - AD (2): {ad_count} 筆")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]; label = self.label_list[idx]
        try:
            img = nib.load(file_path); data_np = img.get_fdata(dtype=np.float32)
            min_val = np.min(data_np); max_val = np.max(data_np)
            if max_val - min_val > 1e-6: data_norm = (data_np - min_val) / (max_val - min_val)
            else: data_norm = data_np
            data_tensor = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0) 
            return data_tensor, label
        except Exception as e:
            print(f"🚨 錯誤：載入或處理檔案 {file_path} 失敗。錯誤: {e}")
            return None, None 

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# ====================================================================
# 【4. 訓練 & 驗證 輔助函數 (梯度累積)】(保持不變)
# ====================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    optimizer.zero_grad() 
    
    for i, (inputs, labels) in enumerate(dataloader):
        if inputs.shape[0] == 0: continue
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps 
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * inputs.size(0) * accumulation_steps 
        _, predicted = torch.max(outputs.data, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / total_preds
    epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval(); running_loss = 0.0; correct_preds = 0; total_preds = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            if inputs.shape[0] == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0); _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0); correct_preds += (predicted == labels).sum().item()
    epoch_loss = running_loss / total_preds; epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc

# ====================================================================
# 【5. 主執行迴圈 (已修改)】
# ====================================================================

def main():
    print(f"--- Cognivex 3-Class 模型訓練啟動 (V11 - ResNet + 梯度累積優化) ---")
    print(f"使用設備: {DEVICE}")
    print(f"物理批次大小 (Batch Size): {BATCH_SIZE}")
    print(f"梯度累積步數: {ACCUMULATION_STEPS}")
    print(f"==> 虛擬批次大小 (Effective Batch Size): {BATCH_SIZE * ACCUMULATION_STEPS}")

    dataset = NiftiDataset(DATA_ROOT)
    if len(dataset) == 0:
        print("🚨 致命錯誤：資料集中沒有找到任何檔案。")
        return
        
    print("... 正在計算類別權重...")
    labels_np = np.array(dataset.label_list)
    nc_count = np.sum(labels_np == 0); mci_count = np.sum(labels_np == 1); ad_count = np.sum(labels_np == 2)
    total = len(labels_np)
    weight_nc = total / (NUM_CLASSES * nc_count) if nc_count > 0 else 0
    weight_mci = total / (NUM_CLASSES * mci_count) if mci_count > 0 else 0
    weight_ad = total / (NUM_CLASSES * ad_count) if ad_count > 0 else 0
    class_weights = torch.tensor([weight_nc, weight_mci, weight_ad], dtype=torch.float32).to(DEVICE)
    print(f"✅ 類別權重計算完畢: NC={weight_nc:.2f}, MCI={weight_mci:.2f}, AD={weight_ad:.2f}")

    dataset_indices = np.arange(len(dataset))
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    print(f"\n--- 開始 {NUM_FOLDS}-Fold 交叉驗證 ---")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_indices)):
        print(f"\n==================== FOLD {fold + 1} / {NUM_FOLDS} ====================")
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        # 🚨 修正點：num_workers=2 (使用 BS=2 時是安全的)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler, collate_fn=collate_fn, num_workers=2, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler, collate_fn=collate_fn, num_workers=2, pin_memory=True)
        
        # 🚨 修正點：實例化新的 ResNet3D 模型
        model = ResNet3D(num_classes=NUM_CLASSES).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(weight=class_weights) 
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        best_val_loss = float('inf')
        fold_model_path = os.path.join(MODEL_OUTPUT_DIR, f"cnn_3d_3class_fold_{fold + 1}_best.pth")

        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # 🚨 修正點：傳入 ACCUMULATION_STEPS
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, ACCUMULATION_STEPS)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
            
            scheduler.step(val_loss)
            
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                  f"耗時: {epoch_duration:.2f}s | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), fold_model_path)
                print(f"   ...Validation Loss 改善 ({best_val_loss:.4f})。模型已儲存至 {fold_model_path}")

    print("\n--- 所有 Folds 訓練完成 ---")

if __name__ == "__main__":
    main()