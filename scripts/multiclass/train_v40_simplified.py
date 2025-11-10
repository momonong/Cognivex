"""
V40: 簡化版 - 只使用與 AD 相關的重要 ROI
使用 Nilearn 提取 ROI 特徵，然後用簡單的 MLP 分類
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import os
import glob
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pandas as pd

from nilearn import datasets
from nilearn import image as nimg
from nilearn.maskers import NiftiLabelsMasker

# ====================================================================
# 【1. 配置】
# ====================================================================
DATA_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/"
MODEL_OUTPUT_DIR = "model/v40_simplified_roi/"

NUM_CLASSES = 3
NUM_FOLDS = 5
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32  # 可以用更大的 batch size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# 🎯 關鍵：只使用與 AD 相關的重要 ROI
# AAL atlas 中與 AD 相關的腦區索引
# 參考文獻：海馬迴、杏仁核、顳葉、頂葉等
IMPORTANT_ROIS = {
    # 海馬迴 (Hippocampus) - 最重要！
    'Hippocampus_L': 37,
    'Hippocampus_R': 38,
    'ParaHippocampal_L': 39,
    'ParaHippocampal_R': 40,
    
    # 杏仁核 (Amygdala)
    'Amygdala_L': 41,
    'Amygdala_R': 42,
    
    # 顳葉 (Temporal)
    'Temporal_Sup_L': 79,
    'Temporal_Sup_R': 80,
    'Temporal_Mid_L': 85,
    'Temporal_Mid_R': 86,
    'Temporal_Inf_L': 89,
    'Temporal_Inf_R': 90,
    
    # 頂葉 (Parietal)
    'Parietal_Sup_L': 59,
    'Parietal_Sup_R': 60,
    'Parietal_Inf_L': 61,
    'Parietal_Inf_R': 62,
    
    # 扣帶迴 (Cingulate)
    'Cingulum_Ant_L': 31,
    'Cingulum_Ant_R': 32,
    'Cingulum_Post_L': 35,
    'Cingulum_Post_R': 36,
    
    # 前額葉 (Frontal)
    'Frontal_Sup_L': 1,
    'Frontal_Sup_R': 2,
    'Frontal_Mid_L': 7,
    'Frontal_Mid_R': 8,
}

print(f"使用 {len(IMPORTANT_ROIS)} 個重要 ROI (從 116 個中篩選)")


# ====================================================================
# 【2. 資料集 - 使用 Nilearn 提取特徵】
# ====================================================================

class SimplifiedROIDataset(Dataset):
    """使用 Nilearn 提取 ROI 平均值作為特徵"""
    
    def __init__(self, data_root, aal_img, important_roi_indices):
        self.data_root = data_root
        self.important_roi_indices = list(important_roi_indices.values())
        self.label_map = {"NC": 0, "MCI": 1, "AD": 2}
        self.subjects = []
        
        # 建立 Nilearn masker (只提取重要的 ROI)
        self.masker = NiftiLabelsMasker(
            labels_img=aal_img,
            labels=self.important_roi_indices,
            standardize=True,  # 標準化
            strategy='mean'  # 計算每個 ROI 的平均值
        )
        
        print(f"正在掃描資料集...")
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
        
        print(f"找到 {len(self.subjects)} 位病患")
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject = self.subjects[idx]
        
        try:
            # 載入三個模態
            t1_img = nimg.load_img(subject["t1"])
            t2_img = nimg.load_img(subject["t2"])
            fa_img = nimg.load_img(subject["fa"])
            
            # 使用 Nilearn 提取 ROI 特徵 (每個 ROI 的平均值)
            t1_features = self.masker.fit_transform(t1_img).flatten()  # (n_rois,)
            t2_features = self.masker.fit_transform(t2_img).flatten()
            fa_features = self.masker.fit_transform(fa_img).flatten()
            
            # 合併三個模態的特徵
            features = np.concatenate([t1_features, t2_features, fa_features])
            
            return torch.tensor(features, dtype=torch.float32), subject["label"], subject["subject_id"]
        
        except Exception as e:
            print(f"錯誤：處理 {subject['subject_id']} 失敗: {e}")
            # 返回零特徵
            n_features = len(self.important_roi_indices) * 3
            return torch.zeros(n_features, dtype=torch.float32), subject["label"], subject["subject_id"]


# ====================================================================
# 【3. 簡單的 MLP 模型】
# ====================================================================

class SimpleMLPClassifier(nn.Module):
    """簡單但有效的 MLP 分類器"""
    
    def __init__(self, input_dim, num_classes=3):
        super(SimpleMLPClassifier, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


# ====================================================================
# 【4. 訓練和驗證】
# ====================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels, _ in tqdm(dataloader, desc="Training", leave=False):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / total, correct / total


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for features, labels, subject_ids in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(subject_ids)
    
    return running_loss / total, correct / total, all_preds, all_labels, all_ids


# ====================================================================
# 【5. 主程式】
# ====================================================================

def main():
    print("=" * 60)
    print("V40: 簡化版 ROI 分類器")
    print("=" * 60)
    
    # 1. 載入 AAL atlas
    print("\n載入 AAL atlas...")
    aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
    aal_img = nimg.load_img(aal_atlas.maps)
    
    # 2. 建立資料集
    dataset = SimplifiedROIDataset(DATA_ROOT, aal_img, IMPORTANT_ROIS)
    
    if len(dataset) == 0:
        print("錯誤：沒有找到資料")
        return
    
    # 測試載入
    test_features, test_label, test_id = dataset[0]
    print(f"\n特徵維度: {test_features.shape[0]} ({len(IMPORTANT_ROIS)} ROIs × 3 modalities)")
    
    # 3. 計算類別權重
    labels_np = np.array([s['label'] for s in dataset.subjects])
    class_counts = np.bincount(labels_np, minlength=NUM_CLASSES)
    class_weights = torch.tensor(
        [len(labels_np) / (NUM_CLASSES * c) if c > 0 else 0 for c in class_counts],
        dtype=torch.float32
    ).to(DEVICE)
    print(f"類別權重: {class_weights.cpu().numpy()}")
    
    # 4. K-Fold 交叉驗證
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
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 建立模型
        input_dim = len(IMPORTANT_ROIS) * 3  # 3 modalities
        model = SimpleMLPClassifier(input_dim, NUM_CLASSES).to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # 訓練
        best_val_acc = 0.0
        patience_counter = 0
        patience = 15
        
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, val_preds, val_labels, val_ids = validate_epoch(model, val_loader, criterion, DEVICE)
            
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # 儲存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                model_path = os.path.join(MODEL_OUTPUT_DIR, f"fold_{fold_num}_best.pth")
                torch.save(model.state_dict(), model_path)
                
                # 儲存預測結果
                results_df = pd.DataFrame({
                    'subject_id': val_ids,
                    'true_label': val_labels,
                    'predicted_label': val_preds
                })
                results_path = os.path.join(MODEL_OUTPUT_DIR, f"fold_{fold_num}_predictions.csv")
                results_df.to_csv(results_path, index=False)
                
                print(f"  ✅ 最佳 Val Acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  ⚠️ Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\nFold {fold_num} 完成。最佳 Val Acc: {best_val_acc:.4f}")
    
    print("\n" + "="*60)
    print("所有 Folds 訓練完成！")
    print("="*60)


if __name__ == "__main__":
    main()
