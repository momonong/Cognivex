from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CenterSpatialCropd,
    Orientationd,
    Spacingd,
    EnsureTyped
)

# --- 來自 cnn_3d 訓練配置的常數 ---
PATCH_SIZE = (128, 128, 128)
A_MIN = 0.0
A_MAX = 1000.0

# 這些是 XAI 和 Prediction 所需的標準驗證/測試轉換
test_transforms = Compose([
    LoadImaged(keys=["image"], meta_keys="image_meta_dict"),  # <--- 修正後的
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=A_MIN, a_max=A_MAX, b_min=0.0, b_max=1.0, clip=True),
    CenterSpatialCropd(keys=["image"], roi_size=PATCH_SIZE),
    EnsureTyped(keys=["image"])
])