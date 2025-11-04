# app/core/papermodel_pipeline/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 來自 train.py 的模型配置 ---
DROPOUT_RATE = 0.2

# --- 來自 train.py 的模型架構 ---

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA) 模組 (論文 2.2 節)
    """
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x) 
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def channel_shuffle(x, groups):
    """
    ShuffleNet 中的 Channel Shuffle 操作
    """
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class ShuffleUnit(nn.Module):
    """
    ShuffleNet V1 Base Unit (論文 Fig. 2, right)
    """
    def __init__(self, in_channels, out_channels, groups):
        super(ShuffleUnit, self).__init__()
        self.groups = groups
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
        out += residual
        return F.relu(out, inplace=True)


class ShuffleUnitDownsample(nn.Module):
    """
    ShuffleNet V1 Downsampling Unit (論文 Fig. 2, left)
    """
    def __init__(self, in_channels, out_channels, groups):
        super(ShuffleUnitDownsample, self).__init__()
        self.groups = groups
        out_channels_branch = out_channels - in_channels
        bottleneck_channels = out_channels_branch // 4

        self.shortcut_avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.gconv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.dwconv = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=2, padding=1,
                                groups=bottleneck_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.gconv2 = nn.Conv2d(bottleneck_channels, out_channels_branch, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels_branch)

    def forward(self, x):
        shortcut = self.shortcut_avgpool(x)
        out = self.gconv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = channel_shuffle(out, self.groups)
        out = self.dwconv(out)
        out = self.bn2(out)
        out = self.gconv2(out)
        out = self.bn3(out)
        out = F.relu(out, inplace=True)
        out = torch.cat([shortcut, out], dim=1)
        return out


class ShuffleNetV1Backbone(nn.Module):
    """
    實現論文 (Fig. 2) 中修改的 ShuffleNet V1 骨幹 (2, 4, 2)
    """
    def __init__(self, in_channels=1, groups=3):
        super(ShuffleNetV1Backbone, self).__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        channels_g3 = [240, 480, 960] # g=3, 1x
        self.stage2 = self._make_stage(24, channels_g3[0], num_base_units=1)
        self.stage3 = self._make_stage(channels_g3[0], channels_g3[1], num_base_units=3)
        self.stage4 = self._make_stage(channels_g3[1], channels_g3[2], num_base_units=1)
        self.final_out_channels = channels_g3[2] # 960

    def _make_stage(self, in_channels, out_channels, num_base_units):
        layers = [ShuffleUnitDownsample(in_channels, out_channels, self.groups)]
        for _ in range(num_base_units):
            layers.append(ShuffleUnit(out_channels, out_channels, self.groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.maxpool(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out) # -> (B * 10, 960, 4, 4)
        return out


class PaperModel(nn.Module):
    """
    實現論文 Fig. 1 的完整架構 (推論模式)
    """
    def __init__(self, num_classes=2, groups=3, dropout_p=DROPOUT_RATE):
        super(PaperModel, self).__init__()
        self.backbone = ShuffleNetV1Backbone(in_channels=1, groups=groups)
        backbone_out_features = self.backbone.final_out_channels
        self.eca = ECA(channels=backbone_out_features, k_size=3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_classify = nn.Linear(backbone_out_features, num_classes)
        
        # Triplet 分支在推論時不需要，但為了載入權重，我們保留它
        # 這裡我們用 Identity 簡化，或者您可以複製 train.py 中的完整定義
        # self.triplet_branch = nn.Identity() # 簡化版
        
        # 為了 100% 匹配 train.py 的權重，我們複製完整定義
        triplet_embed_dim = 128
        self.triplet_branch = nn.Sequential(
            nn.Conv2d(backbone_out_features, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, triplet_embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(triplet_embed_dim),
        )

    def forward(self, x, return_stitched_maps: bool = False):
        # x shape: (B, 10, 1, 128, 128)
        B, N_slices, C, H, W = x.shape
        x_flat = x.view(B * N_slices, C, H, W)

        # 1. 骨幹提取特徵
        # feature_maps shape: (B * 10, 960, 4, 4)
        feature_maps = self.backbone(x_flat)

        # 2. 將 batch 和 slices 維度分離
        feature_maps_grouped = feature_maps.view(B, N_slices, -1, 
                                                 feature_maps.shape[2], 
                                                 feature_maps.shape[3])

        # 3. Tensor Stitching (TS) - 取平均
        # stitched_maps shape: (B, 960, 4, 4)
        stitched_maps = torch.mean(feature_maps_grouped, dim=1)

        # 4a. 分類分支
        att_maps = self.eca(stitched_maps)
        pooled_vec = self.global_pool(att_maps)
        pooled_vec = pooled_vec.view(B, -1)
        dropped_vec = self.dropout(pooled_vec)
        logits = self.fc_classify(dropped_vec)

        # 4b. Triplet 分支 (在推論時我們不需要它的輸出, 但它需要運行)
        triplet_maps = self.triplet_branch(stitched_maps)
        triplet_vec = self.global_pool(triplet_maps)
        triplet_vec = triplet_vec.view(B, -1)
        triplet_embedding = F.normalize(triplet_vec, p=2, dim=1)

        if return_stitched_maps:
            # XAI 需要 feature_maps 和 stitched_maps
            # feature_maps: (B * 10, 960, 4, 4) -> 每個切片各自的特徵
            # stitched_maps: (B, 960, 4, 4) -> 縫合後的特徵
            return logits, triplet_embedding, feature_maps, stitched_maps
        
        return logits, triplet_embedding