import os
import glob
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import math
import warnings
from tqdm import tqdm  # 用於顯示進度條

# --- 0. 全局配置 (Global Configuration) ---

# --- 請根據您的環境修改以下路徑 ---
# 您的 .nii 檔案所在的基礎資料夾
DATA_DIR = r"C:\阿茲海默\sMRI_data"
# 您的兩個類別的資料夾名稱
CLASS_A_NAME = "AD"  # 標籤為 1
CLASS_B_NAME = "NC"  # 標籤為 0
# --- --------------------------- ---

# 論文 3.1 節: "10 consecutive slices are selected from the center"
NUM_SLICES_PER_SUBJECT = 10
# 論文 3.1 節: "with a slice size of 128x128"
SLICE_IMG_SIZE = 128

# 論文 3.3 節: 訓練參數
# "batch size is 100" (如果您的 VRAM 不足, 請調低此數值, 例如 16, 32)
BATCH_SIZE = 32
# "trained for 100 epochs"
NUM_EPOCHS = 100
# "initial learning rate is 0.01"
LEARNING_RATE = 0.01
# "weight-decay is 0.01"
WEIGHT_DECAY = 0.01
# "five-fold cross-validation"
N_SPLITS = 5
# "dropout value of 0.2"
DROPOUT_RATE = 0.2
# 論文 2.3 節: Triplet Loss 的 margin
TRIPLET_MARGIN = 1.0

# 設定 torch 設備
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"將在 {DEVICE} 設備上運行")

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)


# --- 1. 資料處理 (Preprocessing & Loading) ---

def preprocess_nii_to_slices(nii_path):
    """
    載入一個 .nii 檔案, 並執行論文中的切片預處理。
    1. 載入 3D 影像。
    2. 選取矢状面 (sagittal plane)。
    3. 找到中央 10 張切片。
    4. 旋轉、標準化 (0-255) 並縮放至 128x128。

    返回:
        Numpy array, shape (NUM_SLICES, 1, SLICE_IMG_SIZE, SLICE_IMG_SIZE)
    """
    try:
        # 1. 載入 NIfTI 影像 (假設已由 CAT12 處理過)
        img = nib.load(nii_path)
        data = img.get_fdata()

        # 2. 選取矢状面 (sagittal plane)
        # NIfTI 儲存通常是 (Sagittal, Coronal, Axial) -> (X, Y, Z)
        # 我們假設第 0 軸是矢状面 (X 軸)
        sagittal_dim = 0
        num_total_slices = data.shape[sagittal_dim]

        if num_total_slices < NUM_SLICES_PER_SUBJECT:
            print(f"警告：檔案 {nii_path} 矢狀面切片數 ({num_total_slices}) 少於 {NUM_SLICES_PER_SUBJECT}。將跳過此檔案。")
            return None

        # 3. 找到中央 10 張切片
        center_slice_index = num_total_slices // 2
        start_index = center_slice_index - (NUM_SLICES_PER_SUBJECT // 2)
        end_index = start_index + NUM_SLICES_PER_SUBJECT

        # 選取切片 [start_index:end_index, :, :]
        selected_slices_data = data[start_index:end_index, :, :]

        processed_slices = []
        for i in range(NUM_SLICES_PER_SUBJECT):
            slice_2d = selected_slices_data[i, :, :]

            # (重要) 旋轉影像使其方向正確
            slice_2d = np.rot90(slice_2d)

            # (重要) 將體素強度標準化到 0-255 (灰階圖片)
            if np.max(slice_2d) > 0:
                slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))
            slice_2d_uint8 = (slice_2d * 255).astype(np.uint8)

            # 4. 縮放到 128x128
            resized_slice = cv2.resize(slice_2d_uint8, (SLICE_IMG_SIZE, SLICE_IMG_SIZE),
                                       interpolation=cv2.INTER_CUBIC)

            processed_slices.append(resized_slice)

        # 堆疊成 (10, 128, 128)
        stacked_slices = np.stack(processed_slices)

        # 增加通道維度 -> (10, 1, 128, 128)
        # 10 張切片, 1 個灰階通道, 128x128 像素
        return stacked_slices[:, np.newaxis, :, :]

    except Exception as e:
        print(f"錯誤：處理檔案 {nii_path} 失敗: {e}")
        return None


class AlzheimerDataset(Dataset):
    """
    自定義 PyTorch Dataset
    """

    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        nii_path = self.file_paths[idx]
        label = self.labels[idx]

        # 載入並預處理 NIfTI 檔
        # 返回 shape: (10, 1, 128, 128)
        slices_array = preprocess_nii_to_slices(nii_path)

        if slices_array is None:
            # 如果處理失敗 (例如檔案損壞), 返回一個空值 (稍後在 collate_fn 中過濾)
            return None

        # 將 numpy array 轉換為 PyTorch Tensor
        # 並將強度從 0-255 標準化到 0.0-1.0
        slices_tensor = torch.tensor(slices_array, dtype=torch.float32) / 255.0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return slices_tensor, label_tensor


def collate_fn_skip_corrupted(batch):
    """
    過濾掉在 __getitem__ 中返回 None 的損壞樣本
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)


def find_nii_files(base_dir, class_a_name, class_b_name):
    """
    根據用戶的資料夾結構遞迴搜尋 .nii 檔案
    """
    base_path = Path(base_dir)

    # 根據您的路徑 "C:\阿茲海默\sMRI_data\AD\[Subject_ID]\[file].nii"
    # 使用 rglob 遞迴搜尋所有 .nii 檔案
    # **/*.nii 匹配 [Subject_ID]\[file].nii
    files_a = glob.glob(str(base_path / class_a_name / "**" / "*.nii"), recursive=True)
    files_b = glob.glob(str(base_path / class_b_name / "**" / "*.nii"), recursive=True)

    # 處理 .nii.gz
    files_a.extend(glob.glob(str(base_path / class_a_name / "**" / "*.nii.gz"), recursive=True))
    files_b.extend(glob.glob(str(base_path / class_b_name / "**" / "*.nii.gz"), recursive=True))

    print(f"找到 {len(files_a)} 個 {class_a_name} 檔案")
    print(f"找到 {len(files_b)} 個 {class_b_name} 檔案")

    # 建立檔案列表和標籤 (1 = class_a, 0 = class_b)
    all_files = files_a + files_b
    all_labels = [1] * len(files_a) + [0] * len(files_b)

    if len(all_files) == 0:
        print(f"錯誤：在 {base_dir} 中找不到任何 .nii 或 .nii.gz 檔案。")
        print("請檢查 DATA_DIR, CLASS_A_NAME, 和 CLASS_B_NAME 變數是否設定正確。")
        print(f"預期路徑結構範例: {base_dir}\\{class_a_name}\\[Subject_Folder]\\[file].nii")

    return all_files, all_labels


# --- 2. 模型架構 (Model Architecture) ---

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) 模組 (論文 2.2 節)
    """

    def __init__(self, channels, k_size=3):
        # 論文 "on groups of 3 adjacent channels", "k also represents the coverage of 3"
        # 這暗示 k_size=3
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        y = self.avg_pool(x)  # (B, C, 1, 1)

        # (B, C, 1) -> (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        # (B, 1, C) -> (B, 1, C)
        y = self.conv(y)
        # (B, 1, C) -> (B, C, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # (B, C, 1, 1)
        y = self.sigmoid(y)

        # (B, C, H, W) * (B, C, 1, 1) -> (B, C, H, W)
        return x * y.expand_as(x)


def channel_shuffle(x, groups):
    """
    ShuffleNet 中的 Channel Shuffle 操作
    """
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # transpose
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class ShuffleUnit(nn.Module):
    """
    ShuffleNet V1 Base Unit (論文 Fig. 2, right)
    使用殘差連接 (Add)
    """

    def __init__(self, in_channels, out_channels, groups):
        super(ShuffleUnit, self).__init__()
        self.groups = groups
        # 論文 Fig 2 中 Base unit 的 in_channels 和 out_channels 相同
        assert in_channels == out_channels
        bottleneck_channels = out_channels // 4

        self.gconv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.dwconv = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1,
                                groups=bottleneck_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.gconv2 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.gconv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = channel_shuffle(out, self.groups)

        out = self.dwconv(out)
        out = self.bn2(out)

        out = self.gconv2(out)
        out = self.bn3(out)

        out += residual  # Add (殘差連接)
        return F.relu(out, inplace=True)


class ShuffleUnitDownsample(nn.Module):
    """
    ShuffleNet V1 Downsampling Unit (論文 Fig. 2, left)
    使用 Concat 連接
    """

    def __init__(self, in_channels, out_channels, groups):
        super(ShuffleUnitDownsample, self).__init__()
        self.groups = groups
        # 輸出通道數由主分支和捷徑分支合併而成
        out_channels_branch = out_channels - in_channels
        bottleneck_channels = out_channels_branch // 4

        # 左側分支 (Shortcut)
        self.shortcut_avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # 右側分支 (Main)
        self.gconv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        # 3x3DWConv (stride=2)
        self.dwconv = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=2, padding=1,
                                groups=bottleneck_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.gconv2 = nn.Conv2d(bottleneck_channels, out_channels_branch, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels_branch)

    def forward(self, x):
        # 左側分支
        shortcut = self.shortcut_avgpool(x)

        # 右側分支
        out = self.gconv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = channel_shuffle(out, self.groups)

        out = self.dwconv(out)
        out = self.bn2(out)

        out = self.gconv2(out)
        out = self.bn3(out)
        out = F.relu(out, inplace=True)

        # Concat (論文 Fig 2)
        out = torch.cat([shortcut, out], dim=1)
        return out


class ShuffleNetV1Backbone(nn.Module):
    """
    實現論文 (Fig. 2) 中修改的 ShuffleNet V1 骨幹
    論文 2.1 節 "modified to (2, 4, 2)"
    這代表 Stage 2/3/4 的總 unit 數
    Stage 2: 1 Downsampling Unit + 1 Base Unit (共 2)
    Stage 3: 1 Downsampling Unit + 3 Base Units (共 4)
    Stage 4: 1 Downsampling Unit + 1 Base Unit (共 2)
    """

    def __init__(self, in_channels=1, groups=3):
        super(ShuffleNetV1Backbone, self).__init__()
        self.groups = groups

        # 論文的輸入是 128x128
        # 初始 Conv + MaxPool
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 根據 g=3, 1x 的標準 (ShuffleNet v1 論文), 輸出通道數
        # 論文並未指定, 我們採用標準 ShuffleNet v1 的配置
        stage_out_channels = {1: [144, 288, 576], 2: [200, 400, 800], 3: [240, 480, 960]}
        # 論文的 Fig 2 中 gconv1/gconv2 使用了 'groups'，這暗示 g=3 (或 8)
        # 讓我們遵循 g=3, 1x 的標準 (240, 480, 960)
        channels_g3 = [240, 480, 960]

        # Stage 2 (2 units total)
        self.stage2 = self._make_stage(24, channels_g3[0], num_base_units=1)

        # Stage 3 (4 units total)
        self.stage3 = self._make_stage(channels_g3[0], channels_g3[1], num_base_units=3)

        # Stage 4 (2 units total)
        self.stage4 = self._make_stage(channels_g3[1], channels_g3[2], num_base_units=1)

        # 骨幹的最終輸出通道數
        self.final_out_channels = channels_g3[2]  # 960

    def _make_stage(self, in_channels, out_channels, num_base_units):
        layers = []
        # 1. Downsampling Unit
        layers.append(ShuffleUnitDownsample(in_channels, out_channels, self.groups))
        # 2. Base Units
        for _ in range(num_base_units):
            layers.append(ShuffleUnit(out_channels, out_channels, self.groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (B, 1, 128, 128)

        out = self.conv1(x)  # -> (B, 24, 64, 64)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.maxpool(out)  # -> (B, 24, 32, 32)

        out = self.stage2(out)  # -> (B, 240, 16, 16)
        out = self.stage3(out)  # -> (B, 480, 8, 8)
        out = self.stage4(out)  # -> (B, 960, 4, 4)

        # 骨幹只返回 feature map, global pool 在主模型中進行
        return out


class PaperModel(nn.Module):
    """
    實現論文 Fig. 1 的完整架構
    """

    def __init__(self, num_classes=2, groups=3, dropout_p=DROPOUT_RATE):
        super(PaperModel, self).__init__()

        # 1. 特徵提取模組 (骨幹)
        self.backbone = ShuffleNetV1Backbone(in_channels=1, groups=groups)

        # 獲取骨幹的輸出特徵數 (例如 960)
        backbone_out_features = self.backbone.final_out_channels  # 960

        # 2. Attention 模組 (ECA)
        # ECA 在 Fig. 1 中應用於 HxWxC 的 feature map 上
        self.eca = ECA(channels=backbone_out_features, k_size=3)

        # 3. 全局池化 (在兩個分支中都會用到)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 4. Loss 模組

        # 4a. 分類分支 (Cross Entropy)
        self.dropout = nn.Dropout(p=dropout_p)  # 論文 3.3 節 "dropout value of 0.2"
        self.fc_classify = nn.Linear(backbone_out_features, num_classes)

        # 4b. Triplet 分支 (論文 2.3 節 "three sequential 1x1 convolution layers")
        # 論文未指定維度, 我們自定義 (e.g., 960 -> 512 -> 256 -> 128)
        triplet_embed_dim = 128
        self.triplet_branch = nn.Sequential(
            nn.Conv2d(backbone_out_features, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, triplet_embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(triplet_embed_dim),
        )

    def forward(self, x):
        # x shape: (B, 10, 1, 128, 128)

        B, N_slices, C, H, W = x.shape

        # 將 batch 和 slices 維度合併, 以便一次性處理所有 2D 圖片
        # (B, 10, 1, 128, 128) -> (B * 10, 1, 128, 128)
        x_flat = x.view(B * N_slices, C, H, W)

        # 1. 骨幹提取特徵
        # (B * 10, 1, 128, 128) -> (B * 10, 960, 4, 4)
        feature_maps = self.backbone(x_flat)

        # 2. 將 batch 和 slices 維度分離
        # (B * 10, 960, 4, 4) -> (B, 10, 960, 4, 4)
        feature_maps_grouped = feature_maps.view(B, N_slices, -1,
                                                 feature_maps.shape[2],
                                                 feature_maps.shape[3])

        # 3. Tensor Stitching (TS) - 論文 Fig. 1
        # 我們將 "TS" 解讀為對 10 張切片的特徵圖譜取平均
        # (B, 10, 960, 4, 4) -> (B, 960, 4, 4)
        stitched_maps = torch.mean(feature_maps_grouped, dim=1)

        # --- 分支 1: 分類 (Classification) ---
        # 4a. Attention 模組 (ECA)
        att_maps = self.eca(stitched_maps)  # (B, 960, 4, 4)

        # 5a. 全局池化
        pooled_vec = self.global_pool(att_maps)  # (B, 960, 1, 1)
        pooled_vec = pooled_vec.view(B, -1)  # (B, 960)

        # 6a. Dropout & FC
        dropped_vec = self.dropout(pooled_vec)
        logits = self.fc_classify(dropped_vec)  # (B, 2)

        # --- 分支 2: Triplet ---
        # 4b. 1x1 卷積
        triplet_maps = self.triplet_branch(stitched_maps)  # (B, 128, 4, 4)

        # 5b. 全局池化
        triplet_vec = self.global_pool(triplet_maps)  # (B, 128, 1, 1)
        triplet_vec = triplet_vec.view(B, -1)  # (B, 128)

        # 6b. L2 正規化 (論文 2.3 節)
        triplet_embedding = F.normalize(triplet_vec, p=2, dim=1)

        # 返回分類的 logits 和 Triplet 的 embedding
        return logits, triplet_embedding


# --- 3. 損失函數 (Loss Function) ---

class BatchHardTripletLoss(nn.Module):
    """
    實現 Batch-Hard Triplet Loss (在一個 batch 內部挖掘困難樣本)
    """

    def __init__(self, margin=TRIPLET_MARGIN):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, embeddings, labels):
        # embeddings: (B, EmbedDim), labels: (B)

        # 1. 計算 pairwise L2 距離矩陣
        # (B, EmbedDim) -> (B, 1, EmbedDim) 和 (1, B, EmbedDim)
        dist_mat = torch.cdist(embeddings.unsqueeze(0), embeddings.unsqueeze(0)).squeeze(0)

        # 2. 挖掘 hard positive 和 hard negative
        B = labels.size(0)

        # 建立 mask
        # labels.unsqueeze(1) == labels.unsqueeze(0)
        # (B, 1) == (1, B) -> (B, B)
        is_pos = labels.expand(B, B) == labels.expand(B, B).t()
        is_neg = ~is_pos

        # 找到 "最遠" 的 positive (hardest positive)
        # 我們要最大化 dist(a, p)
        # 為了安全, 將負樣本的距離設為 -inf
        dist_pos = dist_mat.clone()
        dist_pos[is_neg] = -torch.inf
        hardest_positive = torch.max(dist_pos, dim=1)[0]

        # 找到 "最近" 的 negative (hardest negative)
        # 我們要最小化 dist(a, n)
        # 為了安全, 將正樣本的距離設為 +inf
        dist_neg = dist_mat.clone()
        dist_neg[is_pos] = torch.inf
        hardest_negative = torch.min(dist_neg, dim=1)[0]

        # 3. 計算 Triplet Loss
        # L_T = max(0, margin + d(a, p) - d(a, n))
        triplet_loss = self.relu(self.margin + hardest_positive - hardest_negative)

        # 只對有意義的樣本 (triplet_loss > 0) 取平均
        num_non_zero_triplets = (triplet_loss > 0).sum()
        if num_non_zero_triplets == 0:
            return torch.tensor(0.0, device=DEVICE)

        return triplet_loss.sum() / num_non_zero_triplets


class CombinedLoss(nn.Module):
    """
    論文 2.3 節: L = L_C + L_T
    """

    def __init__(self, margin=TRIPLET_MARGIN):
        super(CombinedLoss, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_triplet = BatchHardTripletLoss(margin=margin)

    def forward(self, logits, embeddings, labels):
        L_C = self.loss_ce(logits, labels)
        L_T = self.loss_triplet(embeddings, labels)

        L_total = L_C + L_T

        return L_total, L_C, L_T


# --- 4. 訓練與評估 (Training & Evaluation) ---

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    訓練一個 Epoch
    """
    model.train()
    total_loss = 0.0
    total_loss_c = 0.0
    total_loss_t = 0.0

    # 使用 tqdm 顯示進度條
    for batch in tqdm(dataloader, desc="訓練中", leave=False):
        if not batch[0].numel(): continue  # 跳過空 batch

        slices_tensor, labels_tensor = batch
        slices_tensor, labels_tensor = slices_tensor.to(device), labels_tensor.to(device)

        # 1. 前向傳播
        logits, embeddings = model(slices_tensor)

        # 2. 計算損失
        loss, loss_c, loss_t = criterion(logits, embeddings, labels_tensor)

        # 3. 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * slices_tensor.size(0)
        total_loss_c += loss_c.item() * slices_tensor.size(0)
        total_loss_t += loss_t.item() * slices_tensor.size(0)

    # 調整學習率
    if scheduler:
        scheduler.step()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_loss_c = total_loss_c / len(dataloader.dataset)
    avg_loss_t = total_loss_t / len(dataloader.dataset)

    return avg_loss, avg_loss_c, avg_loss_t


def evaluate_epoch(model, dataloader, criterion, device):
    """
    評估一個 Epoch
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="評估中", leave=False):
            if not batch[0].numel(): continue  # 跳過空 batch

            slices_tensor, labels_tensor = batch
            slices_tensor, labels_tensor = slices_tensor.to(device), labels_tensor.to(device)

            # 1. 前向傳播
            logits, embeddings = model(slices_tensor)

            # 2. 計算損失
            loss, _, _ = criterion(logits, embeddings, labels_tensor)

            # 3. 收集預測結果
            preds = torch.argmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)[:, 1]  # 取類別 1 (AD) 的機率

            all_labels.extend(labels_tensor.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total_loss += loss.item() * slices_tensor.size(0)

    avg_loss = total_loss / len(dataloader.dataset)

    # 論文 3.4 節: 計算 ACC, SEN, SPE, F1, AUC
    if not all_labels:  # 如果資料集為空
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5  # 如果標籤中只有一個類別 (例如 batch size 太小)

    # TN, FP, FN, TP
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity (SEN)
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity (SPE)
    else:  # 預測全為 0 或全為 1
        sen = 0.0
        spe = 0.0

    return avg_loss, acc, sen, spe, f1, auc


# --- 5. 主執行函數 (Main Execution) ---

def main():
    print(f"--- 步驟 1: 載入資料 ---")
    all_files, all_labels = find_nii_files(DATA_DIR, CLASS_A_NAME, CLASS_B_NAME)

    if not all_files:
        return  # 找不到檔案, 提前終止

    # 將資料轉換為 numpy array 以便 KFold 切分
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)

    print(f"\n--- 步驟 2: 開始 {N_SPLITS} 折交叉驗證 (K-Fold Cross-Validation) ---")

    # 論文 3.3 節: "five-fold cross-validation is adopted"
    # 我們使用 StratifiedKFold 確保每折中的 AD/NC 比例相似
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_files, all_labels)):
        print(f"\n--- 第 {fold + 1} / {N_SPLITS} 折 ---")

        # 1. 建立資料集
        train_files, val_files = all_files[train_idx], all_files[val_idx]
        train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

        print(f"訓練集樣本數: {len(train_files)} (AD: {sum(train_labels == 1)}, NC: {sum(train_labels == 0)})")
        print(f"驗證集樣本數: {len(val_files)} (AD: {sum(val_labels == 1)}, NC: {sum(val_labels == 0)})")

        train_dataset = AlzheimerDataset(train_files, train_labels)
        val_dataset = AlzheimerDataset(val_files, val_labels)

        # 2. 建立 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn_skip_corrupted  # 使用自定義 collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn_skip_corrupted
        )

        # 3. 初始化模型、損失函數、優化器
        model = PaperModel(num_classes=2).to(DEVICE)
        criterion = CombinedLoss(margin=TRIPLET_MARGIN).to(DEVICE)

        # 論文 3.3 節: "SGD optimizer"
        optimizer = optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            momentum=0.9  # SGD 通常搭配 momentum
        )

        # 論文 3.3 節: "warmup cosine learning rate"
        # 這裡簡化為 Cosine Annealing, 效果接近
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_val_acc = 0.0
        best_fold_results = {}

        for epoch in range(NUM_EPOCHS):
            # 訓練
            train_loss, train_loss_c, train_loss_t = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, DEVICE
            )

            # 評估
            val_loss, val_acc, val_sen, val_spe, val_f1, val_auc = evaluate_epoch(
                model, val_loader, criterion, DEVICE
            )

            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} (C: {train_loss_c:.4f}, T: {train_loss_t:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val ACC: {val_acc:.4f}")

            # 儲存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_fold_results = {
                    "ACC": val_acc,
                    "SEN": val_sen,
                    "SPE": val_spe,
                    "F1": val_f1,
                    "AUC": val_auc
                }
                # (可選) 儲存模型權重
                torch.save(model.state_dict(), f"fold_{fold+1}_best_model.pth")

        print(f"--- 第 {fold + 1} 折 最佳驗證結果 ---")
        print(f"ACC: {best_fold_results['ACC']:.4f}")
        print(f"SEN: {best_fold_results['SEN']:.4f}")
        print(f"SPE: {best_fold_results['SPE']:.4f}")
        print(f"F1:  {best_fold_results['F1']:.4f}")
        print(f"AUC: {best_fold_results['AUC']:.4f}")

        fold_results.append(best_fold_results)

    # --- 步驟 3: 匯總 K-Fold 結果 ---
    print("\n--- 交叉驗證 (Cross-Validation) 最終匯總 ---")

    # 計算平均值
    avg_acc = np.mean([res["ACC"] for res in fold_results])
    avg_sen = np.mean([res["SEN"] for res in fold_results])
    avg_spe = np.mean([res["SPE"] for res in fold_results])
    avg_f1 = np.mean([res["F1"] for res in fold_results])
    avg_auc = np.mean([res["AUC"] for res in fold_results])

    # 計算標準差
    std_acc = np.std([res["ACC"] for res in fold_results])
    std_sen = np.std([res["SEN"] for res in fold_results])
    std_spe = np.std([res["SPE"] for res in fold_results])
    std_f1 = np.std([res["F1"] for res in fold_results])
    std_auc = np.std([res["AUC"] for res in fold_results])

    print(f"平均 ACC: {avg_acc:.4f} \u00B1 {std_acc:.4f}")
    print(f"平均 SEN: {avg_sen:.4f} \u00B1 {std_sen:.4f}")
    print(f"平均 SPE: {avg_spe:.4f} \u00B1 {std_spe:.4f}")
    print(f"平均 F1:  {avg_f1:.4f} \u00B1 {std_f1:.4f}")
    print(f"平均 AUC: {avg_auc:.4f} \u00B1 {std_auc:.4f}")


if __name__ == "__main__":
    # 設置 PyTorch 使用 CUDA
    torch.backends.cudnn.benchmark = True
    main()
