"""
CNN-RF 模型配置
Configuration for CNN-RF Model
"""

from pathlib import Path

# ====================================================================
# 數據路徑配置 (Data Paths)
# ====================================================================
DATA_ROOT = Path("data/MRI_processed")  # 處理後的 MRI 數據
ROI_FEATURES_CSV = Path("data/roi_features.csv")  # ROI 特徵 CSV

# ====================================================================
# 圖譜配置 (Atlas Configuration)
# ====================================================================
ATLAS_PATH = Path("data/aal3/AAL3v1_1mm.nii.gz")
ATLAS_LABELS_PATH = Path("data/aal3/AAL3v1.json")
MNI_TEMPLATE_PATH = Path("data/templates/MNI152_T1_1mm_brain.nii.gz")

# ====================================================================
# 模型配置 (Model Configuration)
# ====================================================================
MODEL_DIR = Path("model/cnn_rf")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 可用的模型
MODELS = {
    "NC_vs_AD": {
        "path": MODEL_DIR / "rf_model_NC_vs_AD.joblib",
        "classes": ['AD', 'NC'],  # 按字母排序
        "description": "二分類：正常控制組 vs 阿茲海默症（全特徵）",
        "features": "All modalities (GM, FA, MD)",
        "selected_features": 30,
        "note": "⚠️ 特徵選擇過於激進，AD 生物標記未被選中"
    },
    "NC_vs_AD_GM": {
        "path": MODEL_DIR / "rf_model_NC_vs_AD_GM_only.joblib",
        "classes": ['AD', 'NC'],  # 按字母排序
        "description": "二分類：正常控制組 vs 阿茲海默症（僅 GM 特徵）⭐ 推薦",
        "features": "GM only (Gray Matter)",
        "selected_features": 83,
        "note": "✓ Hippocampus 和 Amygdala 被選中，生物學可解釋性高"
    },
    "NC_MCI_AD": {
        "path": MODEL_DIR / "rf_model_NC_MCI_AD.joblib",
        "classes": ['AD', 'MCI', 'NC'],  # 按字母排序
        "description": "三分類：正常控制組 vs 輕度認知障礙 vs 阿茲海默症",
        "features": "All modalities (GM, FA, MD)",
        "selected_features": "TBD",
        "note": "尚未訓練"
    }
}

# 默認使用的模型 - 改用 GM-only 模型
DEFAULT_MODEL = "NC_vs_AD_GM"  # ⭐ 推薦使用 GM-only 模型

# ====================================================================
# 特徵提取配置 (Feature Extraction)
# ====================================================================
# 影像模態
MODALITIES = {
    "GM": "sub-{subject_id}_GM_to_MNI.nii.gz",  # 灰質
    "FA": "sub-{subject_id}_FA_to_MNI.nii.gz",  # 分數各向異性
    "MD": "sub-{subject_id}_MD_to_MNI.nii.gz"   # 平均擴散率
}

# ====================================================================
# 訓練配置 (Training Configuration)
# ====================================================================
TRAINING_CONFIG = {
    "n_features_to_select": 30,  # 選擇最重要的 30 個特徵
    "n_splits": 5,                # 5 折交叉驗證
    "random_state": 42,
    
    # RandomForest 參數
    "rf_params": {
        "n_estimators": 200,
        "random_state": 42,
        "class_weight": 'balanced',
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
}

# ====================================================================
# 輸出配置 (Output Configuration)
# ====================================================================
OUTPUT_DIR = Path("output/cnn_rf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 可視化輸出
VISUALIZATION_CONFIG = {
    "brain_map_output": OUTPUT_DIR / "important_rois_map.nii.gz",
    "feature_importance_plot": OUTPUT_DIR / "feature_importance.png",
    "confusion_matrix_plot": OUTPUT_DIR / "confusion_matrix.png"
}

# ====================================================================
# 類別標籤映射 (Label Mapping)
# ====================================================================
LABEL_MAP = {
    "NC": 0,   # 正常控制組
    "MCI": 1,  # 輕度認知障礙
    "AD": 2    # 阿茲海默症
}

LABEL_NAMES = {
    0: "NC (Normal Control)",
    1: "MCI (Mild Cognitive Impairment)",
    2: "AD (Alzheimer's Disease)"
}

# ====================================================================
# 重要腦區 (Important ROIs)
# ====================================================================
# 這些是從訓練中發現的最重要腦區
TOP_ROIS_NC_VS_AD = [
    "Olfactory_L",          # 左側嗅覺皮層
    "OFCant_L",             # 左側前眶額皮層
    "OFCant_R",             # 右側前眶額皮層
    "Cingulate_Post_R",     # 右側後扣帶回
    "ParaHippocampal_R",    # 右側海馬旁回
    "Calcarine_R",          # 右側距狀裂
    "Lingual_R",            # 右側舌回
    "Fusiform_R",           # 右側梭狀回
    "Caudate_L"             # 左側尾狀核
]

# ====================================================================
# 輔助函數 (Helper Functions)
# ====================================================================
def get_model_config(model_name: str = None):
    """獲取模型配置"""
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    return MODELS[model_name]


def print_config():
    """打印配置信息"""
    print("="*80)
    print("CNN-RF Model Configuration")
    print("="*80)
    print(f"Data root: {DATA_ROOT}")
    print(f"ROI features: {ROI_FEATURES_CSV}")
    print(f"\nAvailable models:")
    for name, config in MODELS.items():
        status = "✓" if config['path'].exists() else "✗"
        print(f"  [{status}] {name}: {config['description']}")
    print(f"\nDefault model: {DEFAULT_MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    print_config()
