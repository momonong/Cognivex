import os
import json
import ants
import numpy as np
import nibabel as nib
from tqdm import tqdm

# --- 1. 設定 ---
ATLAS_NII_PATH = r"data/aal3/AAL3v1_1mm.nii.gz" 
ATLAS_LABELS_PATH = r"data/aal3/AAL3v1.json"

# MNI 模板 (用於對齊檔頭資訊)
TEMPLATE_PATH = r"data/templates/MNI152_T1_1mm_brain.nii.gz"

OUTPUT_NIFTI = r"output/cnn_rf/NC_vs_AD_top_features_map.nii.gz"

# 這是從你 train_feat 日誌中複製的 Top 10 特徵
# (我們只需要腦區名稱，GM/FA/MD 後綴已移除)
TOP_FEATURES_NAMES = [
    "Olfactory_L",
    "OFCant_L",
    "OFCant_R",
    "Cingulate_Post_R",
    "ParaHippocampal_R",
    "Calcarine_R",
    "Lingual_R",      # Lingual_R_FA 和 Lingual_R_MD 都很重要
    "Fusiform_R",
    "Caudate_L"
]

def create_feature_map():
    print(f"[*] 載入 AAL3 模板: {ATLAS_NII_PATH}")
    print(f"[*] 載入 AAL3 標籤: {ATLAS_LABELS_PATH}")

    # --- 2. 載入 AAL3 圖譜和標籤 ---
    try:
        atlas_img = nib.load(ATLAS_NII_PATH)
        atlas_data = atlas_img.get_fdata().astype(int)
        
        # 載入 MNI 模板以獲取正確的檔頭 (affine)
        template_img = nib.load(TEMPLATE_PATH)
        
        with open(ATLAS_LABELS_PATH, 'r', encoding='utf-8') as f:
            labels_raw = json.load(f)
        
        # 建立一個「名稱 -> 索引」的反向對照表
        # e.g., {"Precentral_L": 1, "Precentral_R": 2, ...}
        name_to_index_map = {name: int(idx) for idx, name in labels_raw.items()}
        
    except FileNotFoundError as e:
        print(f"[!] 錯誤: 找不到檔案 {e.filename}")
        return
    except Exception as e:
        print(f"[!] 讀取檔案時發生錯誤: {e}")
        return

    # --- 3. 建立新的 3D 影像 (初始化為 0) ---
    # 我們使用 MNI 模板的維度 (193, 229, 193)，因為我們的 AAL3 模板可能還沒重新採樣
    # 為了安全起見，我們在這裡也做一次重新採樣
    print(f"[*] 正在將 AAL3 圖譜重新採樣至 MNI 空間 (這可能需要幾秒)...")
    try:
        atlas_img_ants = ants.image_read(ATLAS_NII_PATH)
        template_img_ants = ants.image_read(TEMPLATE_PATH)
        
        atlas_resampled_ants = ants.resample_image_to_target(
            atlas_img_ants, 
            template_img_ants, 
            interp_type='nearestNeighbor'
        )
        atlas_data = atlas_resampled_ants.numpy().astype(int)
        print(f"    -> 重新採樣完成。 維度: {atlas_data.shape}")
        
    except Exception as e:
        print(f"[!] 錯誤: 重新採樣 AAL3 失敗: {e}。請確認 'antspyx' 已安裝。")
        return

    # 建立一個全零的陣列
    feature_map_data = np.zeros(atlas_data.shape, dtype=np.int16)

    # --- 4. 標記重要的腦區 ---
    print(f"[*] 正在標記 {len(TOP_FEATURES_NAMES)} 個重要腦區...")
    
    count = 0
    for i, feature_name in enumerate(tqdm(TOP_FEATURES_NAMES, desc="標記腦區")):
        if feature_name in name_to_index_map:
            roi_index = name_to_index_map[feature_name]
            
            # 將地圖中所有等於 10 (Caudate_L) 的 voxel 設為 1 (或 i+1)
            feature_map_data[atlas_data == roi_index] = i + 1 # 依序標記 1, 2, 3...
            count += 1
        else:
            print(f"  [!] 警告: 在 AAL3 標籤中找不到 '{feature_name}'")
            
    # --- 5. 儲存新的 NIfTI 檔案 ---
    if count > 0:
        # 使用 MNI 模板的檔頭 (affine) 和標頭 (header) 來儲存
        output_img = nib.Nifti1Image(feature_map_data, template_img.affine, template_img.header)
        output_img.set_data_dtype(np.int16) # 設為整數
        
        nib.save(output_img, OUTPUT_NIFTI)
        
        print(f"\n[SUCCESS] 成功建立特徵地圖！")
        print(f"  -> 檔案儲存於: {OUTPUT_NIFTI}")
        print(f"\n[*] 下一步：")
        print(f"    1. 打開影像查看器 (例如 ITK-SNAP 或 3D Slicer)。")
        print(f"    2. 載入 MNI 模板: {TEMPLATE_PATH}")
        print(f"    3. 在模板上疊加 (Overlay) 你剛剛產生的: {OUTPUT_NIFTI}")
        print(f"    4. 你現在可以看到模型認為最重要的腦區了！")
    else:
        print("[!] 錯誤: 沒有標記任何腦區。")

if __name__ == "__main__":
    # pip install nibabel numpy tqdm antspyx
    # (我們也需要 antspyx 來做重新採樣)
    create_feature_map()
