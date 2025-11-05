# app/core/papermodel_pipeline/xai.py
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import cv2
import pandas as pd
import json
import os
import traceback
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

# 匯入此 pipeline 的模型和預處理
from .model import PaperModel
from .preprocessing import NUM_SLICES_PER_SUBJECT, SLICE_IMG_SIZE, preprocess_nii_to_slices

# --- XAI 輔助函式 (V10) ---

def _get_2d_original_slices(original_nii_path: str) -> Optional[np.ndarray]:
    """
    [XAI V10 - 來自 V9]
    模擬 preprocessing.py 的流程，從 3D T1 影像中提取 10 張 2D 切片。
    返回: (10, 128, 128) 的 numpy 陣列 (原始 MRI)
    """
    print(f"--- XAI: 正在從 3D T1 提取 2D 原始切片 ---")
    try:
        ref_nii = nib.load(original_nii_path)
        ref_shape = ref_nii.shape
        data = ref_nii.get_fdata().astype(np.float32)

        start_index = (ref_shape[0] // 2) - (NUM_SLICES_PER_SUBJECT // 2)
        end_index = start_index + NUM_SLICES_PER_SUBJECT
        
        t1_slices_3d = data[start_index:end_index, :, :]

        processed_t1_slices = []
        for i in range(NUM_SLICES_PER_SUBJECT):
            slice_2d = t1_slices_3d[i, :, :]
            slice_2d_rotated = np.rot90(slice_2d)
            
            if np.max(slice_2d_rotated) > 0:
                slice_2d_rotated = (slice_2d_rotated - np.min(slice_2d_rotated)) / (np.max(slice_2d_rotated) - np.min(slice_2d_rotated))
            slice_2d_uint8 = (slice_2d_rotated * 255).astype(np.uint8)
            
            resized_slice = cv2.resize(
                slice_2d_uint8, 
                (SLICE_IMG_SIZE, SLICE_IMG_SIZE), # (128, 128)
                interpolation=cv2.INTER_CUBIC
            )
            processed_t1_slices.append(resized_slice)
            
        final_t1_slices = np.stack(processed_t1_slices) # (10, 128, 128)
        print(f"  成功提取 10 張 2D T1 切片 (shape: {final_t1_slices.shape})")
        return final_t1_slices

    except Exception as e:
        print(f"  錯誤: 提取 2D T1 切片失敗: {e}")
        traceback.print_exc()
        return None

# --- XAI 核心函式 (Plan V10.1) ---

def calculate_integrated_gradients(
    model: PaperModel,
    model_input_slices: torch.Tensor, # Shape (1, 10, 1, 128, 128)
    target_class_idx: int,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 50
) -> Optional[np.ndarray]:
    """
    [XAI V10.1 - BugFix]
    計算 10 張輸入切片 的「整合梯度」(Integrated Gradients)。
    """
    print(f"--- XAI: 運行 Integrated Gradients (Steps: {steps}) ---")
    model.eval()
    
    if baseline is None:
        baseline = torch.zeros_like(model_input_slices)

    # 1. 產生 Alphas (積分路徑)
    alphas = torch.linspace(0.0, 1.0, steps).to(model_input_slices.device)
    alphas = alphas.view(steps, 1, 1, 1, 1)

    # 2. 產生插值影像 (Interpolated Images)
    input_minus_baseline = model_input_slices - baseline
    interpolated_inputs = baseline + alphas * input_minus_baseline
    
    all_gradients = []
    
    print(f"  正在計算 {steps} 步的梯度...")
    try:
        # ---------------- [BUG FIX] ----------------
        # 錯誤: 'you can only change requires_grad flags of leaf variables.'
        # 我們不能在 'input_step' (一個 slice) 上設定 requires_grad。
        # 修正：我們在 *迴圈外部* 設定 'interpolated_inputs' 的 flag。
        
        # interpolated_inputs.requires_grad = True # (舊的 V10.0 修正 - 依然有問題)

        # ---------------- [V10.1 修正] ----------------
        # 真正的修正是：我們必須在 *迴圈內部*，
        # 將 'input_step' 從計算圖中 .detach()，
        # .clone() 它來建立一個 *新的* leaf variable，
        # 然後才能在它上面設定 .requires_grad
        
        for i in range(steps):
            model.zero_grad()
            
            # 1. 將 input_step 從計算圖中分離
            input_step_data = interpolated_inputs[i:i+1].detach()
            # 2. 複製 (Clone) 它來建立一個新的 Leaf
            input_step = input_step_data.clone()
            # 3. 現在 'input_step' 是一個 Leaf，我們可以設定 flag
            input_step.requires_grad = True
            
            # ---------------- [END V10.1 修正] ----------------
            
            # 3. 前向傳播
            logits, _ = model(input_step) # (1, 2)
            
            # 4. 獲取目標分數
            target_score = logits[0, target_class_idx]
            
            # 5. 反向傳播 (計算梯度)
            target_score.backward()
            
            # 6. 儲存 *輸入* 的梯度
            #    (現在 input_step.grad 會是有效的)
            all_gradients.append(input_step.grad.clone()) # (1, 10, 1, 128, 128)
        
        # 7. 堆疊所有梯度
        gradients_tensor = torch.cat(all_gradients, dim=0)

        # 8. 積分 (取平均)
        avg_gradients = torch.mean(gradients_tensor, dim=0)
        
        # 9. 計算 IG = (input - baseline) * avg_gradients
        integrated_gradients = input_minus_baseline * avg_gradients
        
        # 10. 清理 & 返回
        ig_attributions = integrated_gradients.squeeze().cpu().detach().numpy()
        
        print(f"  成功計算 Integrated Gradients (shape: {ig_attributions.shape})")
        return ig_attributions
        
        print(f"  成功計算 Integrated Gradients (shape: {ig_attributions.shape})")
        return ig_attributions

    except Exception as e:
        print(f"  錯誤: 運行 Integrated Gradients 失敗: {e}")
        traceback.print_exc()
        return None


def save_2d_overlay_visualizations(
    attributions_128: np.ndarray, # [修改] (10, 128, 128) 的 IG 歸因
    original_nii_path: str,
    output_dir: str
) -> Optional[List[str]]:
    """
    [XAI V10 - 不變]
    將 10 張 2D IG 歸因 疊加到 10 張 2D T1 切片上，並儲存為 PNG。
    """
    print(f"--- XAI: 正在儲存 10 張 2D 疊加 (Overlay) PNG 影像 ---")
    try:
        t1_slices = _get_2d_original_slices(original_nii_path)
        if t1_slices is None:
            raise RuntimeError("無法獲取 2D T1 切片")
            
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        cmap = plt.get_cmap('bwr')
        vmax = np.percentile(np.abs(attributions_128), 95)
        vmin = -vmax

        for i in range(NUM_SLICES_PER_SUBJECT):
            t1_slice = t1_slices[i] 
            attr = attributions_128[i]
            
            t1_rgb = cv2.cvtColor(t1_slice, cv2.COLOR_GRAY2RGB)
            attr_normalized = (attr - vmin) / (vmax - vmin + 1e-8)
            heatmap_colored = (cmap(attr_normalized)[:, :, :3] * 255).astype(np.uint8)
            alpha_mask = (np.abs(attr) > (vmax * 0.3)).astype(np.uint8) * 255
            alpha_mask_rgb = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2RGB)
            heatmap_masked = cv2.bitwise_and(heatmap_colored, alpha_mask_rgb)
            overlay = cv2.addWeighted(t1_rgb, 0.7, heatmap_masked, 0.3, 0)
            
            output_path = os.path.join(output_dir, f"slice_{i+1:02d}_ig_overlay.png")
            cv2.imwrite(output_path, overlay)
            saved_paths.append(output_path)
            
        print(f"  成功儲存 {len(saved_paths)} 張 PNG 影像到: {output_dir}")
        return saved_paths

    except Exception as e:
        print(f"  錯誤: 儲存 2D Overlays 失敗: {e}")
        traceback.print_exc()
        return None