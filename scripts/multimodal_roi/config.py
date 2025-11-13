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
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15

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
NUM_WORKERS = 4

print(f"配置載入完成 | Device: {DEVICE}")
print(f"總特徵維度: {TOTAL_FEATURE_DIM} = {NUM_ROIS} ROIs × {len(MODALITIES)} modalities × {FEATURE_DIM_PER_ROI} features")
