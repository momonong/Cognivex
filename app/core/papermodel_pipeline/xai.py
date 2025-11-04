# app/core/papermodel_pipeline/xai.py
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import cv2
import ants
import pandas as pd
import json
import os
import traceback
from nilearn.image import resample_to_img
from typing import List, Dict, Any, Optional

# 匯入此 pipeline 的模型和預處理
from .model import PaperModel
from .preprocessing import NUM_SLICES_PER_SUBJECT, SLICE_IMG_SIZE

# --- XAI 輔助函式 ---

def _get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """ 輔助函式：按名稱路徑獲取 PyTorch 模組 """
    parts = path.split('.')
    mod = model
    for part in parts:
        mod = getattr(mod, part)
    return mod

def _calculate_grad_cam(activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """ 輔助函式：計算 Grad-CAM (來自舊管線) """
    # activation: [1, C, H, W], gradient: [1, C, H, W]
    pooled_gradients = torch.mean(gradient, dim=[0, 2, 3]) # [C]
    for i in range(activation.shape[1]): # C
        activation[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activation, dim=1).squeeze() # [H, W]
    heatmap = torch.relu(heatmap)
    if torch.max(heatmap) > 0:
        heatmap /= torch.max(heatmap)
    return heatmap

# --- XAI 核心函式 ---

def run_grad_cam_on_stitched_map(
    model: PaperModel,
    model_input_slices: torch.Tensor,
    target_class_idx: int,
    target_layer_name: str = "backbone.stage4"
) -> Optional[np.ndarray]:
    """
    [XAI 亮點]
    對「縫合後 (Stitched)」的 3D 特徵圖計算 Grad-CAM。
    這將告訴我們模型在「平均的 3D 大腦」上看到了什麼。
    """
    print(f"--- XAI: 運行 Stitched Grad-CAM (目標層: {target_layer_name}) ---")
    model.eval()
    
    # 設置儲存 activation 和 gradient 的容器
    activations = {}
    gradients = {}
    handles = []

    try:
        # --- 1. 掛載 Hooks ---
        target_layer = _get_module_by_path(model, target_layer_name)
        
        def forward_hook(module, input, output):
            # output shape: (B * 10, 960, 4, 4)
            # 我們需要的是「縫合後」的 map，這在 forward 函式中計算
            # 為了簡化，我們直接抓取 stitched_maps
            pass 

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] shape: (B, 960, 4, 4)
            gradients['stitched_map'] = grad_output[0].detach()

        # **** 重大修改：我們 hook 的是 stitched_map ****
        # 我們需要 hook `model.eca` 的 *輸入*，也就是 `stitched_maps`
        
        def pre_eca_hook(module, input):
            # input[0] shape 是 (B, 960, 4, 4) - 這就是 stitched_maps
            activations['stitched_map'] = input[0].detach()
            
        def pre_eca_backward_hook(module, grad_input, grad_output):
            # grad_input[0] shape (B, 960, 4, 4) - 這是流向 stitched_maps 的梯度
            gradients['stitched_map'] = grad_input[0].detach()

        # 我們 hook eca 層
        handles.append(model.eca.register_forward_pre_hook(pre_eca_hook))
        handles.append(model.eca.register_full_backward_hook(pre_eca_backward_hook))

        # --- 2. 執行前向和反向傳播 ---
        model.zero_grad()
        # 請求 'stitched_maps'
        logits, _, _, _ = model(model_input_slices, return_stitched_maps=True)
        
        # 選擇目標分數並反向傳播
        target_score = logits[0, target_class_idx]
        target_score.backward()

        # --- 3. 檢查 Hooks 結果 ---
        if 'stitched_map' not in activations or 'stitched_map' not in gradients:
            print("  錯誤: 未能從 hooks 捕獲 stitched_map 的 activation 或 gradient。")
            return None
        
        act = activations['stitched_map']  # (1, 960, 4, 4)
        grad = gradients['stitched_map'] # (1, 960, 4, 4)
        
        # --- 4. 計算 Grad-CAM ---
        stitched_heatmap = _calculate_grad_cam(act, grad) # (4, 4)
        
        print(f"  成功計算 Stitched Grad-CAM (shape: {stitched_heatmap.shape})")
        return stitched_heatmap.cpu().numpy()

    except Exception as e:
        print(f"  錯誤: 運行 Grad-CAM 失敗: {e}")
        traceback.print_exc()
        return None
    finally:
        for handle in handles:
            handle.remove()


def reproject_heatmap_to_3d(
    heatmap_2d: np.ndarray, # (4, 4) 或 (128, 128)
    original_nii_path: str
) -> Optional[nib.Nifti1Image]:
    """
    [XAI 亮點]
    將單張 2D 熱圖 (代表 "stitched" 結果) 重新投影回 3D NIfTI 空間。
    """
    print(f"--- XAI: 重新投影 2D 熱圖至 3D NIfTI 空間 ---")
    try:
        # 1. 載入原始 NIfTI 檔案以獲取 metadata
        ref_nii = nib.load(original_nii_path)
        ref_shape = ref_nii.shape # (X, Y, Z) - e.g., (192, 256, 256)
        ref_affine = ref_nii.affine
        ref_header = ref_nii.header

        # 2. 建立一個空的 3D 陣列
        final_3d_heatmap = np.zeros(ref_shape, dtype=np.float32)
        
        # 3. 獲取原始 2D 切片的形狀 (Y, Z)
        # 注意：preprocess_nii_to_slices 旋轉了 90 度
        # 原始切片 shape (Y, Z), 旋轉後 (Z, Y), resize 成 (128, 128)
        # 我們反向操作：
        # GradCAM (4,4) -> (128, 128) -> (Y, Z) -> rot-90
        
        slice_shape_2d_orig = (ref_shape[1], ref_shape[2]) # (Y, Z)
        
        # 4. Resize 熱圖
        # (4, 4) -> (128, 128)
        heatmap_128 = cv2.resize(heatmap_2d, (SLICE_IMG_SIZE, SLICE_IMG_SIZE), 
                                 interpolation=cv2.INTER_LINEAR)
        # (128, 128) -> (Y, Z) (cv2 resize 是 W, H -> Z, Y)
        heatmap_orig_res = cv2.resize(heatmap_128, (slice_shape_2d_orig[1], slice_shape_2d_orig[0]), 
                                      interpolation=cv2.INTER_LINEAR)
        
        # 5. 反向旋轉 (rot90 的反向是 rot -90, 或 3 次 rot90)
        heatmap_orig_oriented = np.rot90(heatmap_orig_res, k=-1)
        
        # 6. 找到中央 10 張切片的位置
        start_index = (ref_shape[0] // 2) - (NUM_SLICES_PER_SUBJECT // 2)

        # 7. 將這張 2D 熱圖「廣播」到所有 10 個切片位置
        print(f"  將 2D 熱圖 (shape {heatmap_orig_oriented.shape}) 廣播到 3D 體積的 {start_index} 至 {start_index + NUM_SLICES_PER_SUBJECT} 矢狀切面")
        for i in range(NUM_SLICES_PER_SUBJECT):
            final_3d_heatmap[start_index + i, :, :] = heatmap_orig_oriented
            
        # 8. 創建 NIfTI 影像
        heatmap_nii = nib.Nifti1Image(final_3d_heatmap, ref_affine, ref_header)
        print("  成功建立 Native Space 3D 熱圖 NIfTI 影像。")
        return heatmap_nii

    except Exception as e:
        print(f"  錯誤: 2D->3D 投影失敗: {e}")
        traceback.print_exc()
        return None


def normalize_native_to_mni(
    native_t1_path: str,
    native_heatmap_nii: nib.Nifti1Image,
    mni_template_path: str,
    output_dir: str # [新增] 我們需要一個地方來儲存 QC 檔案
) -> Optional[ants.ANTsImage]:
    """
    [XAI 步驟 - V2 with QC]
    使用 ANTs 將 Native Space 的熱圖標準化到 MNI 空間。
    並儲存一個 warped T1 影像用於 QC (Quality Control)。
    """
    print(f"--- XAI: 運行 ANTs 空間標準化 (SyN) ---")
    try:
        # [新增] 確保 QC 目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 載入影像
        print("  載入 ANTs 影像...")
        fixed_mni = ants.image_read(mni_template_path)
        moving_t1 = ants.image_read(native_t1_path)
        moving_heatmap = ants.from_nibabel(native_heatmap_nii)
        
        # 2. 執行 Brain Extraction
        print("  執行 T1 腦部提取 (Brain Extraction)...")
        moving_t1_mask = ants.get_mask(moving_t1, low_thresh=moving_t1.mean() * 0.3, cleanup=2)
        moving_t1_brain = ants.mask_image(moving_t1, moving_t1_mask)
        fixed_mni_brain = fixed_mni 

        # 3. 計算 T1 -> MNI 的轉換
        print("  計算 T1 -> MNI 的 SyN 轉換 (這可能需要幾分鐘)...")
        transform = ants.registration(
             fixed=fixed_mni_brain,
             moving=moving_t1_brain,
             type_of_transform='SyN',
             verbose=False 
        )
        
        # 4. 將轉換應用到 Heatmap
        print("  將轉換應用到 3D 熱圖...")
        heatmap_normalized = ants.apply_transforms(
             fixed=fixed_mni_brain,
             moving=moving_heatmap,
             transformlist=transform['fwdtransforms'],
             interpolator='linear',
             verbose=False
        )
        
        # --- [新增] 儲存 QC 影像 ---
        qc_warped_t1_path = os.path.join(output_dir, "qc_t1_warped_to_mni.nii.gz")
        print(f"  儲存 QC (Warped T1) 影像到: {qc_warped_t1_path}")
        ants.apply_transforms(
            fixed=fixed_mni_brain,
            moving=moving_t1, # 使用原始 T1 (帶頭骨)
            transformlist=transform['fwdtransforms'],
            interpolator='linear',
            output_filename=qc_warped_t1_path
        )
        # --- [結束新增] ---
        
        print("  ANTs 標準化成功。")
        return heatmap_normalized

    except Exception as e:
        print(f"  錯誤: ANTs 標準化失敗: {e}")
        traceback.print_exc()
        return None

def resample_to_atlas(
    mni_heatmap_ants: ants.ANTsImage,
    atlas_nii_path: str
) -> Optional[nib.Nifti1Image]:
    """
    [XAI 步驟]
    將 MNI 熱圖重採樣以匹配 AAL 圖譜的網格。
    """
    print(f"--- XAI: 重採樣熱圖至 Atlas 網格 ---")
    try:
        # 將 ANTsImage 轉回 NIfTI (Nilearn 需要 NIfTI object)
        mni_heatmap_nii = mni_heatmap_ants.to_nibabel()
        
        resampled_img = resample_to_img(
            source_img=mni_heatmap_nii,
            target_img=atlas_nii_path,
            interpolation="linear",
        )
        print("  重採樣成功。")
        return resampled_img
    except Exception as e:
        print(f"  錯誤: 重採樣失敗: {e}")
        traceback.print_exc()
        return None

def analyze_brain_regions(
    resampled_heatmap_nii: nib.Nifti1Image,
    atlas_nii_path: str,
    atlas_label_path: str, # JSON 檔案路徑
    threshold_percentile: float = 95.0
) -> Optional[pd.DataFrame]:
    """
    [XAI 步驟]
    分析每個腦區的激活強度。
    """
    print(f"--- XAI: 分析腦區激活 (閾值: {threshold_percentile}th percentile) ---")
    try:
        act_data = resampled_heatmap_nii.get_fdata()
        
        atlas_nii = nib.load(atlas_nii_path)
        atlas_data = atlas_nii.get_fdata().astype(int)
        
        if act_data.shape != atlas_data.shape:
            print(f"  錯誤: 熱圖 shape {act_data.shape} 與 Atlas shape {atlas_data.shape} 不匹配。")
            return None
            
        with open(atlas_label_path, 'r') as f:
            # 假設 JSON 格式為 {"1": "RegionName1", "2": "RegionName2", ...}
            label_map = json.load(f)

        # 決定閾值
        threshold_val = np.percentile(act_data[act_data > 0], threshold_percentile)
        print(f"  計算激活閾值 ({threshold_percentile}th): {threshold_val:.4f}")
        
        unique_regions = np.unique(atlas_data)
        results = []
        
        for region_id in unique_regions:
            if region_id == 0: continue # 跳過背景
            
            region_mask = (atlas_data == region_id)
            region_activations = act_data[region_mask]
            
            # 只考慮高於閾值的激活
            significant_activations = region_activations[region_activations > threshold_val]
            
            if significant_activations.size > 0:
                region_name = label_map.get(str(region_id), f"Unknown_ID_{region_id}")
                results.append({
                    "region_name": region_name,
                    "region_id": int(region_id),
                    "activation_score": float(np.mean(significant_activations)),
                    "voxel_count": int(significant_activations.size),
                })
                
        if not results:
            print("  警告: 沒有腦區的激活高於閾值。")
            return pd.DataFrame(columns=["region_name", "region_id", "activation_score", "voxel_count"])
            
        df = pd.DataFrame(results)
        df = df.sort_values(by="activation_score", ascending=False)
        print(f"  分析完成，找到 {len(df)} 個顯著激活的腦區。")
        return df
        
    except Exception as e:
        print(f"  錯誤: 腦區分析失敗: {e}")
        traceback.print_exc()
        return None