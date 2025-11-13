"""
多模態 ROI 特徵提取配置
Configuration for Multimodal ROI Feature Extraction Pipeline
"""

from pathlib import Path

# ====================================================================
# 數據路徑配置 (Data Paths)
# ====================================================================
DATA_ROOT = Path("E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI")
AAL_ATLAS_PATH = Path("data/aal3/AAL3v1_1mm.nii.gz")
MNI_TEMPLATE_PATH = Path("data/affine/mni152_template.nii.gz")

# ====================================================================
# 模型配置 (Model Configuration)
# ====================================================================
# 3D ResNet-10 Mini-CNN 配置
RESNET_CONFIG = {
    "in_channels": 1,           # 單模態輸入
    "num_classes": 64,          # 輸出 64 維特徵向量
    "block_config": [1, 1, 1, 1],  # ResNet-10: 4 個 block，每個 1 層
    "initial_filters": 32,      # 初始濾波器數量
}

# XGBoost 配置
XGBOOST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multi:softmax",  # NC vs MCI vs AD
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}

# ====================================================================
# ROI 配置 (ROI Configuration)
# ====================================================================
NUM_ROIS = 116  # AAL-116 圖譜
MODALITIES = ["T1", "T2_FLAIR", "DWI"]  # 三種模態
FEATURE_DIM_PER_ROI = 64  # 每個 ROI 每個模態的特徵維度
TOTAL_FEATURE_DIM = NUM_ROIS * len(MODALITIES) * FEATURE_DIM_PER_ROI  # 22,104

# ====================================================================
# 訓練配置 (Training Configuration)
# ====================================================================
# 高效能配置 (針對 24GB VRAM 優化)
BATCH_SIZE = 16  # 從 4 增加到 16 (4x 速度提升)
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15

# 如果 VRAM 不足，可以降低 BATCH_SIZE:
# BATCH_SIZE = 8  # 中等配置
# BATCH_SIZE = 4  # 保守配置

# 類別標籤
LABEL_MAP = {
    "NC": 0,
    "MCI": 1,
    "AD": 2
}

# ====================================================================
# 輸出路徑 (Output Paths)
# ====================================================================
OUTPUT_DIR = Path("output/multimodal_roi")
MODEL_DIR = Path("model/multimodal_roi")
CACHE_DIR = Path("cache/multimodal_roi")

# 創建必要目錄
for dir_path in [OUTPUT_DIR, MODEL_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ====================================================================
# 3D Patch 提取配置 (3D Patch Extraction)
# ====================================================================
PATCH_CONFIG = {
    "padding": 2,  # ROI 周圍的 padding (voxels)
    "min_patch_size": (8, 8, 8),  # 最小 patch 尺寸
    "target_patch_size": (32, 32, 32),  # 目標 patch 尺寸（會進行 resize）
    "interpolation": "trilinear",  # 插值方法
}

# 高效能配置選項 (如果想要更快的速度，可以增加 patch size)
# PATCH_CONFIG["target_patch_size"] = (40, 40, 40)  # 更大的 patch，更多細節
# 注意: 增加 patch size 會增加 VRAM 使用和計算時間

# ====================================================================
# 交叉驗證配置 (Cross-Validation)
# ====================================================================
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# ====================================================================
# 設備配置 (Device Configuration)
# ====================================================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 高效能配置 (針對多核 CPU 優化)
NUM_WORKERS = 8  # 從 4 增加到 8 (加速數據加載)

# 啟用 CUDA 優化
if torch.cuda.is_available():
    # 啟用 cuDNN 自動調優
    torch.backends.cudnn.benchmark = True
    # 啟用 TF32 (Ampere GPU 及以上)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ====================================================================
# 輔助函數 (Helper Functions)
# ====================================================================
def print_config():
    """打印配置信息"""
    print("="*80)
    print("Multi-modal ROI Pipeline Configuration")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Number of ROIs: {NUM_ROIS}")
    print(f"Modalities: {', '.join(MODALITIES)}")
    print(f"Feature dim per ROI: {FEATURE_DIM_PER_ROI}")
    print(f"Total feature dim: {TOTAL_FEATURE_DIM} = {NUM_ROIS} x {len(MODALITIES)} x {FEATURE_DIM_PER_ROI}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print("="*80)

# 只在直接運行 config.py 時顯示配置信息
if __name__ == "__main__":
    print_config()
