"""
V41: 使用你之前成功的 Simple3DCNN 架構
直接對整個 3D volume 分類，不用 ROI 或 MIL
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
import numpy as np
import nibabel as nib
import os
import glob
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import scipy.ndimage

# ====================================================================
# 【1. 配置】
# ====================================================================
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

DATA_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/"
MODEL_OUTPUT_DIR = "model/v41_simple_3dcnn/"

NUM_CLASSES = 3
NUM_FOLDS = 5
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 24  # 🚀 增加到 24 來充分利用 VRAM (目前只用 10GB/24GB)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 影像參數
PATCH_SIZE = (128, 128, 128)
TARGET_VOXEL_SIZE = (1.0, 1.0, 1.0)

# 🚨 修正：根據 EDA 結果設定正確的數值範圍
# T1: [-50, 1500], T2: [-150, 1400], DWI: [-400, 4600]
INTENSITY_RANGES = {
    'T1': (-50.0, 1500.0),
    'T2': (-150.0, 1400.0),
    'DWI': (-400.0, 4600.0)
}

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


# ====================================================================
# 【2. 模型 - 使用你成功的架構】
# ====================================================================

class Simple3DCNN_MultiClass(nn.Module):
    """基於你成功的 binary classification 模型"""
    
    def __init__(self, in_channels=3, num_classes=3):  # 3 modalities
        super(Simple3DCNN_MultiClass, self).__init__()
        
        def create_conv_block(in_c, out_c, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=padding),
                nn.InstanceNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2)
            )
        
        # 128 -> 64
        self.block1 = create_conv_block(in_channels, 16)
        # 64 -> 32
        self.block2 = create_conv_block(16, 32)
        # 32 -> 16
        self.block3 = create_conv_block(32, 64)
        # 16 -> 8
        self.block4 = create_conv_block(64, 128)
        
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ====================================================================
# 【3. 資料集】
# ====================================================================

class MultiModalDataset(Dataset):
    """載入 T1, T2, DWI 三個模態"""
    
    def __init__(self, data_root, verbose=True):
        self.data_root = data_root
        self.label_map = {"NC": 0, "MCI": 1, "AD": 2}
        self.subjects = []
        self.verbose = verbose
        
        if self.verbose:
            print(f"掃描資料集...")
        for label_name, label_id in self.label_map.items():
            class_path = os.path.join(data_root, label_name)
            if not os.path.isdir(class_path):
                continue
            
            t1_files = glob.glob(os.path.join(class_path, "*_T1.nii.gz"))
            for t1_path in t1_files:
                base_name = t1_path.replace("_T1.nii.gz", "")
                subject_id = os.path.basename(base_name)
                t2_path = base_name + "_T2_FLAIR.nii.gz"
                fa_path = base_name + "_DWI.nii.gz"
                
                if os.path.exists(t2_path) and os.path.exists(fa_path):
                    self.subjects.append({
                        "t1": t1_path, "t2": t2_path, "fa": fa_path,
                        "label": label_id, "subject_id": subject_id
                    })
        
        if self.verbose:
            print(f"找到 {len(self.subjects)} 位病患")
    
    def __len__(self):
        return len(self.subjects)
    
    def _load_and_preprocess(self, nifti_path, modality):
        """載入並預處理單個 NIfTI 檔案"""
        try:
            # 1. 載入
            img = nib.load(nifti_path)
            img_ras = nib.as_closest_canonical(img)
            data_ras = img_ras.get_fdata()
            
            # 2. Resample
            current_voxel_size = img_ras.header.get_zooms()[:3]
            zoom_factors = [c / t for c, t in zip(current_voxel_size, TARGET_VOXEL_SIZE)]
            data_resampled = scipy.ndimage.zoom(data_ras, zoom_factors, order=1)
            
            # 3. 🚨 修正：使用正確的數值範圍標準化
            intensity_min, intensity_max = INTENSITY_RANGES[modality]
            data_scaled = (data_resampled - intensity_min) / (intensity_max - intensity_min + 1e-6)
            data_scaled = np.clip(data_scaled, 0.0, 1.0)
            
            # 4. Center crop
            (h, w, d) = data_scaled.shape
            (ch, cw, cd) = PATCH_SIZE
            h_start = max(0, (h // 2) - (ch // 2))
            w_start = max(0, (w // 2) - (cw // 2))
            d_start = max(0, (d // 2) - (cd // 2))
            
            data_cropped = data_scaled[
                h_start : h_start + ch,
                w_start : w_start + cw,
                d_start : d_start + cd
            ]
            
            # 5. Pad if needed
            if data_cropped.shape != PATCH_SIZE:
                padded = np.zeros(PATCH_SIZE, dtype=np.float32)
                padded[:data_cropped.shape[0], :data_cropped.shape[1], :data_cropped.shape[2]] = data_cropped
                data_cropped = padded
            
            return data_cropped
            
        except Exception as e:
            if self.verbose:
                print(f"錯誤：載入 {os.path.basename(nifti_path)} 失敗: {e}")
            return np.zeros(PATCH_SIZE, dtype=np.float32)
    
    def __getitem__(self, idx):
        subject = self.subjects[idx]
        
        try:
            # 載入三個模態（傳入模態名稱以使用正確的數值範圍）
            t1_data = self._load_and_preprocess(subject["t1"], 'T1')
            t2_data = self._load_and_preprocess(subject["t2"], 'T2')
            fa_data = self._load_and_preprocess(subject["fa"], 'DWI')
            
            # Stack 成 (3, H, W, D)
            volume = np.stack([t1_data, t2_data, fa_data], axis=0)
            
            return torch.tensor(volume, dtype=torch.float32), subject["label"], subject["subject_id"]
        
        except Exception as e:
            if self.verbose:
                print(f"錯誤：處理 {subject['subject_id']} 失敗: {e}")
            return torch.zeros((3, *PATCH_SIZE), dtype=torch.float32), subject["label"], subject["subject_id"]


# ====================================================================
# 【4. 訓練和驗證】
# ====================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch_num):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    first_batch_logits = None
    
    for batch_idx, (volumes, labels, _) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        volumes, labels = volumes.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # 🚀 更快的梯度清零
        
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(volumes)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * volumes.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        
        # 記錄第一個 batch 的 logits
        if batch_idx == 0:
            first_batch_logits = outputs[0].detach().cpu().numpy()
    
    # 計算預測分布
    pred_dist = np.bincount(all_preds, minlength=NUM_CLASSES)
    label_dist = np.bincount(all_labels, minlength=NUM_CLASSES)
    
    return running_loss / total, correct / total, pred_dist, label_dist, first_batch_logits


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    first_batch_logits = None
    
    with torch.no_grad():
        for batch_idx, (volumes, labels, _) in enumerate(dataloader):
            volumes, labels = volumes.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(volumes)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * volumes.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            # 記錄第一個 batch 的 logits
            if batch_idx == 0:
                first_batch_logits = outputs[0].detach().cpu().numpy()
    
    # 計算預測分布
    pred_dist = np.bincount(all_preds, minlength=NUM_CLASSES)
    label_dist = np.bincount(all_labels, minlength=NUM_CLASSES)
    
    return running_loss / total, correct / total, pred_dist, label_dist, first_batch_logits


# ====================================================================
# 【5. 主程式】
# ====================================================================

def main():
    # 🚀 啟用 cudnn benchmark 來自動優化卷積算法
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    print("="*60)
    print("V41: Simple 3D CNN (基於你成功的架構)")
    print("="*60)
    print(f"使用裝置: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Patch Size: {PATCH_SIZE}")
    print(f"學習率: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("="*60)
    
    # 1. 建立資料集
    dataset = MultiModalDataset(DATA_ROOT, verbose=True)
    
    if len(dataset) == 0:
        print("錯誤：沒有找到資料")
        return
    
    # 測試載入
    test_volume, test_label, test_id = dataset[0]
    print(f"\n測試載入: {test_id}")
    print(f"Volume 形狀: {test_volume.shape}")
    print(f"數值範圍: [{test_volume.min():.4f}, {test_volume.max():.4f}]")
    print(f"標籤: {test_label}")
    
    # 2. 計算類別權重
    labels_np = np.array([s['label'] for s in dataset.subjects])
    class_counts = np.bincount(labels_np, minlength=NUM_CLASSES)
    class_weights = torch.tensor(
        [len(labels_np) / (NUM_CLASSES * c) if c > 0 else 0 for c in class_counts],
        dtype=torch.float32
    ).to(DEVICE)
    print(f"\n類別分布: NC={class_counts[0]}, MCI={class_counts[1]}, AD={class_counts[2]}")
    print(f"類別權重: {class_weights.cpu().numpy()}")
    
    # 3. K-Fold 交叉驗證
    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    dataset_indices = np.arange(len(dataset))
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_indices, labels_np)):
        fold_num = fold + 1
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num}/{NUM_FOLDS}")
        print(f"{'='*60}")
        
        # 建立 DataLoader
        train_subset = torch.utils.data.Subset(dataset, train_ids)
        val_subset = torch.utils.data.Subset(dataset, val_ids)
        
        # 🚀 使用多個 workers 來加速資料載入
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                                 num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, 
                               num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        
        # 建立模型
        model = Simple3DCNN_MultiClass(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
        
        # 🚀 使用 torch.compile 加速（PyTorch 2.0+ 且有 Triton）
        # 注意：Windows 上 Triton 支援有限，如果失敗就跳過
        # try:
        #     model = torch.compile(model, mode='reduce-overhead')
        #     print(f"   ✅ 使用 torch.compile 加速")
        # except Exception as e:
        #     print(f"   ⚠️ torch.compile 不可用，使用標準模式")
        #     pass
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # 使用新的 GradScaler API
        try:
            scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        except TypeError:
            scaler = GradScaler()
        
        # 訓練
        best_val_acc = 0.0
        patience_counter = 0
        patience = 20
        
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc, train_pred_dist, train_label_dist, train_logits = train_epoch(
                model, train_loader, criterion, optimizer, DEVICE, scaler, epoch+1
            )
            val_loss, val_acc, val_pred_dist, val_label_dist, val_logits = validate_epoch(
                model, val_loader, criterion, DEVICE
            )
            
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 詳細輸出
            print(f"\nEpoch {epoch+1:3d}/{NUM_EPOCHS} | LR: {current_lr:.6f}")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Labels={train_label_dist}, Preds={train_pred_dist}")
            if train_logits is not None:
                print(f"    Train Logits 範例: [{train_logits[0]:.3f}, {train_logits[1]:.3f}, {train_logits[2]:.3f}]")
            
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f} | "
                  f"Labels={val_label_dist}, Preds={val_pred_dist}")
            if val_logits is not None:
                print(f"    Val Logits 範例: [{val_logits[0]:.3f}, {val_logits[1]:.3f}, {val_logits[2]:.3f}]")
            
            # 儲存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                model_path = os.path.join(MODEL_OUTPUT_DIR, f"fold_{fold_num}_best.pth")
                torch.save(model.state_dict(), model_path)
                print(f"    ✅ 最佳 Val Acc: {best_val_acc:.4f} (已儲存)")
            else:
                patience_counter += 1
                print(f"    ⏳ 沒有改善 ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"\n  ⚠️ Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\nFold {fold_num} 完成。最佳 Val Acc: {best_val_acc:.4f}")
    
    print("\n" + "="*60)
    print("所有 Folds 訓練完成！")
    print("="*60)


if __name__ == "__main__":
    main()
