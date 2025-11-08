import torch
import numpy as np
import nibabel as nib
import os
import glob

# --- 設定路徑 ---
ACTIVATION_DIR = 'output/cnn_3d/activations/'
MNI_TEMPLATE_PATH = 'data/affine/mni152_template.nii.gz'
TARGET_SHAPE = (160, 160, 160) # 您的 CNN 輸出目標維度

def validate_spatial_consistency():
    """
    檢查 MNI 模板、目標尺寸和 Activation 檔案之間的空間參數一致性。
    """
    print("\n=======================================================")
    print("🧠 階段 4 空間參數驗證工具")
    print("=======================================================")

    # 1. MNI 模板檢查
    print(f"1. 檢查 MNI 模板 ({MNI_TEMPLATE_PATH})...")
    mni_template_img = None
    try:
        mni_template_img = nib.load(MNI_TEMPLATE_PATH)
        mni_shape = mni_template_img.get_fdata().shape
        mni_affine = mni_template_img.affine
        print(f"   ✅ MNI 模板載入成功。")
        print(f"   => MNI 體素維度 (Shape): {mni_shape}")
        print(f"   => MNI Affine 矩陣:\n{mni_affine}")
    except Exception as e:
        print(f"   ❌ 致命錯誤：無法載入 MNI 模板。請檢查路徑和檔案權限。錯誤: {e}")
        return False, None

    # 2. 目標尺寸與 MNI 模板尺寸一致性檢查
    print(f"\n2. 檢查 CNN 目標尺寸 ({TARGET_SHAPE}) 與 MNI 維度的一致性...")
    if mni_shape != TARGET_SHAPE:
        print(f"   ❌ 警告：CNN 目標維度 ({TARGET_SHAPE}) 與 MNI 模板維度 ({mni_shape}) 不一致。")
        print("   這可能導致重新取樣錯誤或空間解釋混亂。請將 TARGET_SHAPE 設為 MNI 模板維度。")
        # 由於您可能已經確認過 MNI 模板是 (160, 160, 160)，這裡需要人工確認
        # 我們將基於 MNI 模板修正 TARGET_SHAPE 以進行後續檢查
        # TARGET_SHAPE = mni_shape
    else:
        print("   ✅ CNN 目標維度與 MNI 模板維度一致。")

    # 3. Activation 檔案檢查 (取第一個檔案為例)
    pt_files = sorted(glob.glob(os.path.join(ACTIVATION_DIR, '*_activation.pt')))
    if not pt_files:
        print(f"\n3. ❌ 錯誤：在 {ACTIVATION_DIR} 中找不到任何 Activation 檔案。請重跑階段 1。")
        return False, None
    
    first_pt_path = pt_files[0]
    first_npy_path = first_pt_path.replace('_activation.pt', '_affine.npy')

    print(f"\n3. 檢查第一個 Activation 檔案 ({os.path.basename(first_pt_path)})...")
    try:
        activation_tensor = torch.load(first_pt_path)
        activation_np = activation_tensor.numpy()
        
        # 假設結構是 (C, D_act, H_act, W_act)
        if activation_np.ndim == 5 and activation_np.shape[0] == 1:
            activation_np = activation_np.squeeze(axis=0)
        
        heatmap_raw_shape = np.mean(activation_np, axis=0).shape
        
        print(f"   ✅ Activation 檔案載入成功。")
        print(f"   => 原始熱圖維度 (D_act, H_act, W_act): {heatmap_raw_shape}")
        print(f"   => 縮放因子 (Zoom Factor): {[t / c for t, c in zip(TARGET_SHAPE, heatmap_raw_shape)]}")
        
    except Exception as e:
        print(f"   ❌ 錯誤：無法載入或處理 Activation 檔案。錯誤: {e}")
        return False, None

    # 4. 原始 Affine 矩陣檢查
    print(f"\n4. 檢查原始影像 Affine 矩陣 ({os.path.basename(first_npy_path)})...")
    try:
        raw_affine = np.load(first_npy_path)
        print(f"   ✅ 原始 Affine 載入成功。Shape: {raw_affine.shape}")
        print(f"   => 原始影像 Affine:\n{raw_affine}")
        
        # 比較 MNI Affine 和原始 Affine 的差異
        affine_diff = np.abs(mni_affine - raw_affine).sum()
        if affine_diff > 1e-4:
            print(f"   ⚠️ 警告：原始 Affine 與 MNI Affine 差異巨大 (差異總和: {affine_diff:.2f})。")
            print("   這確認了之前配準失敗的原因。必須使用強制替換或 nilearn.image.resample_to_img。")
        else:
            print("   ✅ 原始 Affine 與 MNI Affine 非常接近 (可能是 MNI 空間的影像)。")
            
    except Exception as e:
        print(f"   ❌ 錯誤：無法載入原始 Affine 檔案。錯誤: {e}")
        return False, None
    
    print("\n=======================================================")
    print("✅ 空間驗證完成。所有關鍵參數已記錄。")
    print("=======================================================")
    return True, mni_template_img

# --- 運行驗證工具 ---
if __name__ == '__main__':
    validation_successful, mni_img = validate_spatial_consistency()
    
    if validation_successful:
        print("\n下一步：請使用上面輸出的 MNI Affine 矩陣，並將其與您的原始 Affine 進行人工比對。")
        print("接著，請使用包含『Affine 強制替換』邏輯的 $04$ 腳本，並將 MNI 模板作為配準目標。")