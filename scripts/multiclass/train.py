import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler 
import torch.nn.functional as F
# 🚨 V39 修正：
from sklearn.model_selection import StratifiedKFold # <--- 不再使用 KFold
import numpy as np
import nibabel as nib 
import os
import glob
import time
import re
import warnings 
from tqdm import tqdm
import pandas as pd
import random 

# (Dipy/Nilearn/Warning imports remain unchanged)
try:
    from nilearn import datasets
    from nilearn import image as nimg
    from dipy.align.reslice import reslice 
except ImportError:
    pass 

warnings.filterwarnings("ignore", category=UserWarning, module='nilearn')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Casting data from int16 to float32")

# ====================================================================
# 【1. 設定與配置 (V39 - Stratified)】
# ====================================================================
DATA_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI/" 
MODEL_OUTPUT_DIR = "model/atlas_mil_stratified/"  # 更新輸出目錄 (v39)

NUM_CLASSES = 3
NUM_FOLDS = 5
NUM_EPOCHS = 50  # 🚨 減少 epochs，先快速測試

LEARNING_RATE = 1e-3        # 🚨 提高學習率來加速收斂
WEIGHT_DECAY = 1e-4         
BATCH_SIZE = 8              # 🚨 減少 batch size 來加速 (116 個 ROI 已經很大了)
ACCUMULATION_STEPS = 2      # 🚨 用 gradient accumulation 來模擬 batch_size=16
GRAD_CLIP_NORM = 1.0       

PATCH_SIZE = (32, 32, 32)
NUM_ROIS = 116 
FEATURE_DIM = 256           # 🚨 減少特徵維度來加速 (從 512 → 256)          

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() is not None and os.cpu_count() >= 2 else 0

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


# (FocalLoss, download_and_check_atlas, AtlasPatchDataset, AtlasMILNet 保持 V36 不變)
# ... (此處省略 V36 已有的類別和函數) ...
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing  # 🚨 新增 label smoothing
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                self.alpha = torch.tensor(alpha)
    def forward(self, input, target):
        # 🚨 使用 label_smoothing 來避免過度自信
        ce_loss = F.cross_entropy(input, target, weight=None, reduction='none', 
                                   label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss
        if self.alpha is not None:
            if self.alpha.device != input.device:
                 self.alpha = self.alpha.to(input.device)
            alpha_t = self.alpha[target]
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
def download_and_check_atlas(target_mni_template_img):
    print("正在下載/載入 AAL 大腦圖譜...")
    try:
        aal_atlas = datasets.fetch_atlas_aal(version='SPM12') 
        aal_img = nimg.load_img(aal_atlas.maps)
        if aal_img.shape != target_mni_template_img.shape or not np.allclose(aal_img.affine, target_mni_template_img.affine):
            print("警告：AAL 圖譜與您的 MNI 資料不符。正在將 AAL 圖譜重採樣至 MNI 空間...")
            aal_img = nimg.resample_to_img(aal_img, target_mni_template_img, interpolation='nearest')
            print(f"✅ AAL 圖譜已成功對齊至 Shape: {aal_img.shape}")
        else:
            print(f"✅ AAL 圖譜已對齊。Shape: {aal_img.shape}")
        return aal_img
    except Exception as e:
        print(f"🚨 致命錯誤：無法下載或處理 AAL 圖譜。錯誤訊息: {e}")
        exit()
class AtlasPatchDataset(Dataset):
    def __init__(self, data_root, aal_img):
        self.data_root = data_root
        self.aal_img = aal_img
        self.aal_data = aal_img.get_fdata().astype(np.int16) 
        self.patch_size_3d = (PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2])
        self.subjects = []
        self.label_map = {"NC": 0, "MCI": 1, "AD": 2}
        self.training = False 
        print(f"正在掃描 *已對齊* 的資料集 (V39)...")
        # 🚨 V39 修正：我們不再依賴 Excel，確認使用資料夾結構
        for label_name, label_id in self.label_map.items():
            class_path = os.path.join(data_root, label_name)
            if not os.path.isdir(class_path): 
                print(f"警告：找不到資料夾 {class_path}，跳過。")
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
        print(f"掃描完成。找到 {len(self.subjects)} 位已對齊的病患。")
    def __len__(self):
        return len(self.subjects)
    
    def _rotate_and_resample(self, img_data, img_affine):
        if self.training and hasattr(random, 'random') and random.random() < 0.5 and 'dipy' in globals():
            angles = np.deg2rad(np.random.uniform(-5, 5, 3)) 
            Rx = np.array([[1, 0, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0]), 0], [0, np.sin(angles[0]), np.cos(angles[0]), 0], [0, 0, 0, 1]])
            Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1]), 0], [0, 1, 0, 0], [-np.sin(angles[1]), 0, np.cos(angles[1]), 0], [0, 0, 0, 1]])
            Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0, 0], [np.sin(angles[2]), np.cos(angles[2]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            new_affine = img_affine @ Rz @ Ry @ Rx
            new_data, new_affine_out = reslice(img_data, img_affine, img_affine, new_affine)
            return new_data, new_affine_out
        return img_data, img_affine 

    def _normalize(self, data):
        min_val, max_val = data.min(), data.max()
        if max_val - min_val > 1e-6:
            return (data - min_val) / (max_val - min_val)
        return data

    def __getitem__(self, idx):
        subject_files = self.subjects[idx]
        label = subject_files["label"]
        subject_id = subject_files["subject_id"]
        
        try:
            t1_nimg = nimg.load_img(subject_files["t1"])
            t2_nimg = nimg.load_img(subject_files["t2"])
            fa_nimg = nimg.load_img(subject_files["fa"])
            
            t1_data = t1_nimg.get_fdata(dtype=np.float32)
            t2_data = t2_nimg.get_fdata(dtype=np.float32)
            fa_data = fa_nimg.get_fdata(dtype=np.float32)
            
            # 🚨 除錯：檢查原始資料
            if idx == 0:  # 只在第一個樣本時輸出
                print(f"\n[DEBUG] 第一個樣本 ({subject_id}) 的原始資料:")
                print(f"  T1 範圍: [{t1_data.min():.2f}, {t1_data.max():.2f}], 平均: {t1_data.mean():.2f}")
                print(f"  T2 範圍: [{t2_data.min():.2f}, {t2_data.max():.2f}], 平均: {t2_data.mean():.2f}")
                print(f"  FA 範圍: [{fa_data.min():.2f}, {fa_data.max():.2f}], 平均: {fa_data.mean():.2f}")
            
            if self.training:
                 t1_data, t1_affine = self._rotate_and_resample(t1_data, t1_nimg.affine)
                 t2_data, _ = self._rotate_and_resample(t2_data, t2_nimg.affine)
                 fa_data, _ = self._rotate_and_resample(fa_data, fa_nimg.affine)

            bag_of_patches = []
            for i in range(1, NUM_ROIS + 1):
                roi_indices = np.where(self.aal_data == i)
                if roi_indices[0].size == 0:
                    roi_patch = torch.zeros((3, *self.patch_size_3d), dtype=torch.float32)
                    bag_of_patches.append(roi_patch)
                    continue
                    
                x1, x2 = roi_indices[0].min(), roi_indices[0].max()
                y1, y2 = roi_indices[1].min(), roi_indices[1].max()
                z1, z2 = roi_indices[2].min(), roi_indices[2].max()

                t1_patch = self._normalize(t1_data[x1:x2+1, y1:y2+1, z1:z2+1])
                t2_patch = self._normalize(t2_data[x1:x2+1, y1:y2+1, z1:z2+1])
                fa_patch = self._normalize(fa_data[x1:x2+1, y1:y2+1, z1:z2+1])
                
                # 🚨 除錯：檢查標準化後的資料
                if idx == 0 and i == 37:  # 第一個樣本的海馬迴
                    print(f"\n[DEBUG] ROI {i} (海馬迴) 標準化後:")
                    print(f"  T1 範圍: [{t1_patch.min():.4f}, {t1_patch.max():.4f}]")
                    print(f"  T2 範圍: [{t2_patch.min():.4f}, {t2_patch.max():.4f}]")
                    print(f"  FA 範圍: [{fa_patch.min():.4f}, {fa_patch.max():.4f}]")

                t1_tensor = torch.tensor(t1_patch, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) 
                t2_tensor = torch.tensor(t2_patch, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                fa_tensor = torch.tensor(fa_patch, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                
                t1_resized = F.interpolate(t1_tensor.unsqueeze(0), size=self.patch_size_3d, mode='trilinear', align_corners=False).squeeze(0) 
                t2_resized = F.interpolate(t2_tensor.unsqueeze(0), size=self.patch_size_3d, mode='trilinear', align_corners=False).squeeze(0)
                fa_resized = F.interpolate(fa_tensor.unsqueeze(0), size=self.patch_size_3d, mode='trilinear', align_corners=False).squeeze(0)
                
                roi_patch = torch.cat([t1_resized, t2_resized, fa_resized], dim=0)
                bag_of_patches.append(roi_patch)
            
            final_tensor = torch.stack(bag_of_patches)
            
            # 🚨 除錯：檢查最終輸出
            if idx == 0:
                print(f"\n[DEBUG] 最終輸出張量:")
                print(f"  形狀: {final_tensor.shape}")
                print(f"  範圍: [{final_tensor.min():.4f}, {final_tensor.max():.4f}]")
                print(f"  平均: {final_tensor.mean():.4f}")
                print(f"  標準差: {final_tensor.std():.4f}")
                print(f"  是否有 NaN: {torch.isnan(final_tensor).any()}")
                print(f"  是否有 Inf: {torch.isinf(final_tensor).any()}")

            return final_tensor, label, subject_id
        
        except Exception as e:
            # 🚨 修正：無論 training 或 validation 都要顯示錯誤
            print(f"\n🚨 錯誤：處理病患 {subject_id} (idx={idx}) 時失敗: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
            
def mil_collate_fn(batch):
    original_batch_size = len(batch)
    batch = list(filter(lambda x: x[0] is not None, batch))
    filtered_batch_size = len(batch)
    
    # 🚨 除錯：顯示過濾掉多少資料
    if original_batch_size != filtered_batch_size:
        print(f"\n⚠️ 警告：Batch 中有 {original_batch_size - filtered_batch_size}/{original_batch_size} 個樣本載入失敗")
    
    if not batch:
        return torch.tensor([]), torch.tensor([]), []
    patches = torch.stack([item[0] for item in batch])    
    labels = torch.tensor([item[1] for item in batch])    
    subject_ids = [item[2] for item in batch]              
    return patches, labels, subject_ids
class AtlasMILNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM, num_rois=NUM_ROIS):
        super(AtlasMILNet, self).__init__()
        self.num_rois = num_rois
        self.feature_dim = feature_dim
        
        # 🚨 簡化：更快的 encoder (減少層數但增加通道)
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),  # 32→16, 更快
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # 16→8
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # 8→4
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1)), 
            nn.Flatten(), 
            nn.Linear(256, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # 🚨 簡化：更快的 attention (移除 BN 來避免 reshape 問題)
        self.attention_net = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 🚨 簡化：更快的 classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # 🚨 修正：更好的初始化
        self._init_weights() 
    def _init_weights(self):
        """初始化模型權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, N, C, D, H, W = x.shape
        x = x.view(B * N, C, D, H, W)
        
        # 特徵提取
        patch_features = self.encoder(x)  # (B*N, feature_dim)
        patch_features = patch_features.view(B, N, -1)  # (B, N, feature_dim)
        
        # 🚨 簡化：直接計算 attention (不需要複雜的 reshape)
        patch_features_flat = patch_features.view(B * N, -1)
        A_scores_flat = self.attention_net(patch_features_flat)  # (B*N, 1)
        A_scores = A_scores_flat.view(B, N, 1)  # (B, N, 1)
        A_weights = F.softmax(A_scores, dim=1)  # (B, N, 1)
        
        # 加權聚合
        patient_vector = torch.sum(patch_features * A_weights, dim=1)  # (B, feature_dim)
        
        # 分類
        logits = self.classifier(patient_vector)  # (B, num_classes)
        
        return logits, A_weights.squeeze(-1) 

# (train_epoch 和 validate_epoch 保持 V36 不變)
def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps, fold_num, scaler):
    model.train()
    if hasattr(dataloader.dataset, 'training'):
        dataloader.dataset.training = True 
    running_loss = 0.0; correct_preds = 0; total_preds = 0
    all_labels = []; all_preds = []
    optimizer.zero_grad() 
    progress_bar = tqdm(dataloader, desc=f"Fold {fold_num} Train", leave=False, dynamic_ncols=True)
    for i, (patches, labels, _) in enumerate(progress_bar):
        if patches.shape[0] == 0: continue
        patches, labels = patches.to(device), labels.to(device)
        
        # 🚨 除錯：檢查輸入
        if i == 0:
            print(f"\n[DEBUG] 第一個 batch 的輸入:")
            print(f"  Patches 形狀: {patches.shape}")
            print(f"  Patches 範圍: [{patches.min():.4f}, {patches.max():.4f}]")
            print(f"  Patches 平均: {patches.mean():.4f}")
            print(f"  Labels: {labels.cpu().numpy()}")
        
        with autocast(): 
            logits, _ = model(patches)
            
            # 🚨 除錯：檢查模型輸出
            if i == 0:
                print(f"\n[DEBUG] 第一個 batch 的輸出:")
                print(f"  Logits 形狀: {logits.shape}")
                print(f"  Logits 範圍: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"  Logits 平均: {logits.mean():.4f}")
                print(f"  Logits 樣本: {logits[0].detach().cpu().numpy()}")
            
            loss = criterion(logits, labels)
            
            # 🚨 除錯：檢查 loss
            if i == 0:
                print(f"  Loss: {loss.item():.4f}")
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        running_loss += loss.item() * patches.size(0) 
        _, predicted = torch.max(logits.data, 1); total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        progress_bar.set_postfix(loss=f"{(running_loss / (total_preds + 1e-6)):.4f}")
    epoch_loss = running_loss / total_preds; epoch_acc = correct_preds / total_preds
    
    # 🚨 除錯：顯示 training 的預測分布
    print(f"\n[Train] 標籤分布: {np.bincount(all_labels, minlength=NUM_CLASSES)}, 預測分布: {np.bincount(all_preds, minlength=NUM_CLASSES)}")
    
    return epoch_loss, epoch_acc
def validate_epoch(model, dataloader, criterion, device, fold_num):
    model.eval()
    if hasattr(dataloader.dataset, 'training'):
        dataloader.dataset.training = False 
    running_loss = 0.0; correct_preds = 0; total_preds = 0
    all_subject_ids = []; all_labels = []; all_preds = []; all_weights = []
    
    # 🚨 除錯：記錄每個 batch 的預測
    batch_details = []
    
    progress_bar = tqdm(dataloader, desc=f"Fold {fold_num} Val  ", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, (patches, labels, subject_ids) in enumerate(progress_bar):
            if patches.shape[0] == 0: continue
            patches, labels = patches.to(device), labels.to(device)
            with autocast():
                logits, weights = model(patches) 
                loss = criterion(logits, labels)
            running_loss += loss.item() * patches.size(0)
            _, predicted = torch.max(logits.data, 1)
            
            # 🚨 除錯：記錄這個 batch 的詳細資訊
            batch_correct = (predicted == labels).sum().item()
            batch_details.append({
                'batch_idx': batch_idx,
                'batch_size': labels.size(0),
                'labels': labels.cpu().numpy().tolist(),
                'predictions': predicted.cpu().numpy().tolist(),
                'correct': batch_correct,
                'logits_sample': logits[0].cpu().numpy().tolist() if logits.size(0) > 0 else []
            })
            
            total_preds += labels.size(0)
            correct_preds += batch_correct
            all_subject_ids.extend(subject_ids)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_weights.append(weights.cpu().numpy())
    
    # 🚨 修正：檢查是否有有效的預測
    if total_preds == 0:
        print(f"\n⚠️ 警告：Fold {fold_num} 的 validation set 沒有任何有效資料！")
        return 0.0, 0.0, None
    
    epoch_loss = running_loss / total_preds; epoch_acc = correct_preds / total_preds
    
    # 🚨 除錯：顯示詳細的預測資訊
    print(f"\n--- Validation 詳細資訊 (Fold {fold_num}) ---")
    print(f"總樣本數: {total_preds}, 正確預測: {correct_preds}, Accuracy: {epoch_acc:.4f}")
    print(f"標籤分布: {np.bincount(all_labels, minlength=NUM_CLASSES)}")
    print(f"預測分布: {np.bincount(all_preds, minlength=NUM_CLASSES)}")
    
    # 顯示前 3 個 batch 的詳細資訊
    for detail in batch_details[:3]:
        print(f"\nBatch {detail['batch_idx']}: Size={detail['batch_size']}, Correct={detail['correct']}/{detail['batch_size']}")
        print(f"  真實標籤: {detail['labels']}")
        print(f"  預測標籤: {detail['predictions']}")
        if detail['logits_sample']:
            print(f"  Logits 範例: [{detail['logits_sample'][0]:.3f}, {detail['logits_sample'][1]:.3f}, {detail['logits_sample'][2]:.3f}]")
    
    if len(batch_details) > 3:
        print(f"... (還有 {len(batch_details) - 3} 個 batches)")
    print("---" * 20)
    try:
        xai_results = {"subject_id": all_subject_ids, "label": all_labels, "prediction": all_preds}
        weights_np = np.concatenate(all_weights, axis=0)
        for i in range(NUM_ROIS):
            xai_results[f"ROI_{i+1}_Weight"] = weights_np[:, i]
        xai_df = pd.DataFrame(xai_results)
    except Exception as e:
        if total_preds > 0:
            print(f"警告：建立 XAI DataFrame 失敗: {e}")
        xai_df = None
    return epoch_loss, epoch_acc, xai_df

# ====================================================================
# 【5. 主執行迴圈 (V39 - StratifiedKFold)】
# ====================================================================

def main():
    print(f"--- Cognivex 計畫 (V39 - Atlas-MIL (StratifiedKFold)) 訓練啟動 ---")
    print(f"使用 *已對齊* 的資料: {DATA_ROOT}")
    print(f"標籤來源：資料夾名稱 (AD/MCI/NC)")
    print(f"Batch Size (Phys/Accum/Virt): {BATCH_SIZE}/{ACCUMULATION_STEPS}/{BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"==========================================================")

    # 1. 取得並驗證 AAL 圖譜
    print("正在取得參考影像 (用於對齊 Atlas)...")
    try:
        ref_t1_path = glob.glob(os.path.join(DATA_ROOT, "*", "*_T1.nii.gz"))[0]
        ref_img = nimg.load_img(ref_t1_path)
    except Exception as e:
        print(f"🚨 致命錯誤：無法從 {DATA_ROOT} 載入參考 T1 影像。錯誤: {e}")
        return
        
    aal_img = download_and_check_atlas(ref_img)

    # 2. 建立資料集
    dataset = AtlasPatchDataset(DATA_ROOT, aal_img)
    
    if len(dataset) == 0:
        print("🚨 致命錯誤：資料集中沒有任何有效的病患。")
        return
    
    # 🚨 除錯：測試載入第一個樣本
    print("\n--- 測試載入第一個樣本 ---")
    try:
        test_patches, test_label, test_id = dataset[0]
        if test_patches is not None:
            print(f"✅ 成功載入樣本: {test_id}, Label: {test_label}, Shape: {test_patches.shape}")
        else:
            print(f"❌ 第一個樣本載入失敗！")
    except Exception as e:
        print(f"❌ 測試載入失敗: {e}")
        import traceback
        traceback.print_exc()
        
    print("... 正在計算類別權重...")
    # 🚨 V39 修正：
    # 我們需要標籤 (y) 來進行 StratifiedKFold
    labels_np = np.array([s['label'] for s in dataset.subjects])
    
    nc_count = np.sum(labels_np == 0); mci_count = np.sum(labels_np == 1); ad_count = np.sum(labels_np == 2)
    total = len(labels_np)
    
    if nc_count == 0 or mci_count == 0 or ad_count == 0:
        print(f"🚨 警告：一個或多個類別在資料集中缺失！NC={nc_count}, MCI={mci_count}, AD={ad_count}")
        
    weight_nc = total / (NUM_CLASSES * nc_count) if nc_count > 0 else 0
    weight_mci = total / (NUM_CLASSES * mci_count) if mci_count > 0 else 0
    weight_ad = total / (NUM_CLASSES * ad_count) if ad_count > 0 else 0
    
    class_weights = torch.tensor([weight_nc, weight_mci, weight_ad], dtype=torch.float32).to(DEVICE)
    print(f"✅ 類別權重計算完畢: NC={weight_nc:.2f}, MCI={weight_mci:.2f}, AD={weight_ad:.2f}")
    
    # 🚨 除錯：顯示實際的 loss 權重效果
    print(f"   權重比例: NC:MCI:AD = {weight_nc/weight_mci:.2f}:{1.0:.2f}:{weight_ad/weight_mci:.2f}")
    print(f"   這意味著 AD 的錯誤會被放大 {weight_ad/weight_nc:.2f}x 相對於 NC")

    # 3. K-Fold 交叉驗證 (V39 修正)
    dataset_indices = np.arange(len(dataset))
    
    # 🚨 V39 修正：使用 StratifiedKFold
    kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    print(f"\n--- 開始 {NUM_FOLDS}-Fold *分層* 交叉驗證 (Stratified) ---")

    # 🚨 V39 修正：.split() 現在需要 y (labels_np)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_indices, labels_np)):
        fold_num = fold + 1
        print(f"\n==================== FOLD {fold_num} / {NUM_FOLDS} ====================")
        
        # 檢查分層結果 (可選)
        train_labels = labels_np[train_ids]
        val_labels = labels_np[val_ids]
        print(f"   Train: {len(train_ids)} (NC={np.sum(train_labels==0)}, MCI={np.sum(train_labels==1)}, AD={np.sum(train_labels==2)})")
        print(f"   Val:   {len(val_ids)} (NC={np.sum(val_labels==0)}, MCI={np.sum(val_labels==1)}, AD={np.sum(val_labels==2)})")
        
        # 🚨 修正：使用 Subset 而不是 Sampler，確保索引正確對應
        train_subset = torch.utils.data.Subset(dataset, train_ids)
        val_subset = torch.utils.data.Subset(dataset, val_ids)
        
        print(f"... 正在啟動多執行緒資料載入器 (num_workers={NUM_WORKERS})...")
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=mil_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=mil_collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
        
        # 🚨 除錯：顯示 dataloader 資訊
        print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"   預期 Val 樣本數: {len(val_ids)}, 預期 Val batches: ~{len(val_ids) // BATCH_SIZE + (1 if len(val_ids) % BATCH_SIZE else 0)}")
        
        # 🚨 驗證：檢查 validation subset 的標籤分布
        val_subset_labels = [dataset.subjects[idx]['label'] for idx in val_ids]
        val_label_counts = np.bincount(val_subset_labels, minlength=NUM_CLASSES)
        print(f"   驗證 Val subset 標籤分布: NC={val_label_counts[0]}, MCI={val_label_counts[1]}, AD={val_label_counts[2]}")
        if not np.array_equal(val_label_counts, np.bincount(val_labels, minlength=NUM_CLASSES)):
            print(f"   ⚠️ 警告：Val subset 標籤與預期不符！")
        
        model = AtlasMILNet(num_classes=NUM_CLASSES).to(DEVICE)
        print(f"   ✅ 模型已建立並初始化")
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # 🚨 改用 CrossEntropyLoss + 類別權重 (更穩定，梯度更大)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print(f"   ✅ 使用 CrossEntropyLoss (label_smoothing=0.1, class_weights)") 
        
        # 🚨 改善：使用 CosineAnnealingWarmRestarts 來避免卡在局部最小值
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        # 備用：如果想用 ReduceLROnPlateau，可以改回來
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True) 
        
        scaler = GradScaler()
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10  # 🚨 Early stopping: 10 epochs 沒改善就停止
        
        fold_model_path = os.path.join(MODEL_OUTPUT_DIR, f"v39_mil_fold_{fold_num}_best.pth")
        xai_output_path = os.path.join(MODEL_OUTPUT_DIR, f"v39_mil_fold_{fold_num}_xai_weights.csv")

        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, ACCUMULATION_STEPS, fold_num, scaler)
            val_loss, val_acc, xai_df = validate_epoch(model, val_loader, criterion, DEVICE, fold_num)
            
            # 🚨 修正：CosineAnnealingWarmRestarts 不需要參數
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                  f"耗時: {epoch_duration:.2f}s | LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # 🚨 改善：基於 accuracy 來儲存模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), fold_model_path)
                print(f"   ✅ Val Acc 改善 ({best_val_acc:.4f})。模型已儲存。")
                
                if xai_df is not None:
                    xai_df.to_csv(xai_output_path, index=False)
                    print(f"   ✅ XAI 權重已儲存")
            else:
                patience_counter += 1
                print(f"   ⏳ 沒有改善 ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"\n⚠️ Early stopping! {patience} epochs 沒有改善。")
                    break

    print("\n--- 所有 Folds 訓練完成 ---")

if __name__ == "__main__":
    main()