# app/core/papermodel_pipeline/nodes.py
import torch
import numpy as np
import os
import traceback
import cv2
import nibabel as nib
from typing import Dict, Any

# 匯入此 pipeline 的所有組件
from . import model
from . import preprocessing
from . import xai
from .preprocessing import SLICE_IMG_SIZE, NUM_SLICES_PER_SUBJECT

# --- [V10] 全局服務加載 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
print(f"專案根目錄 (PROJECT_ROOT) 設定為: {PROJECT_ROOT}")

def _get_absolute_path(relative_path: str) -> str:
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)

MODEL_WEIGHTS_PATH = _get_absolute_path(os.getenv("MODEL_WEIGHTS_PATH", "model/shufflenet/shufflenet_best_model.pth"))
# [V10] 我們不再需要 Atlas 或 MNI 模板

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def _load_model_instance():
    """ 輔助函式：加載模型實例 """
    print(f"正在從 {MODEL_WEIGHTS_PATH} 加載模型權重...")
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"模型權重未找到: {MODEL_WEIGHTS_PATH}。")
    
    pipeline_model = model.PaperModel().to(DEVICE)
    pipeline_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    pipeline_model.eval()
    print("模型加載成功並設為 eval() 模式。")
    return pipeline_model

try:
    PIPELINE_MODEL = _load_model_instance()
except Exception as e:
    print(f"致命錯誤：無法加載模型或依賴檔案: {e}")
    PIPELINE_MODEL = None

# --- LangGraph 節點定義 ---

def run_inference_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    [LangGraph 節點 1: 推論] (V10 不變)
    """
    print("\n--- [Node: Inference] ---")
    try:
        if PIPELINE_MODEL is None:
            raise RuntimeError("模型未被成功加載, 無法執行推論。")
            
        nii_path = state.get("t1_native_path")
        if not nii_path:
            return {"error": "Missing 't1_native_path' in state."}

        slices_array = preprocessing.preprocess_nii_to_slices(nii_path)
        if slices_array is None:
            return {"error": f"Failed to preprocess NIfTI file: {nii_path}"}
            
        slices_tensor = torch.tensor(slices_array, dtype=torch.float32) / 255.0
        slices_tensor = slices_tensor.unsqueeze(0).to(DEVICE) 

        with torch.no_grad():
            logits, _ = PIPELINE_MODEL(slices_tensor)
            
        pred_idx = torch.argmax(logits, dim=1).item()
        result = "AD" if pred_idx == 1 else "NC"
        
        print(f"  推論結果: {result}")
        return {"classification_result": result}

    except Exception as e:
        print(f"  錯誤: {e}")
        return {"error": f"Inference node failed: {e}\n{traceback.format_exc()}"}


def run_post_processing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    [LangGraph 節點 2: XAI 後處理 - V10 (Integrated Gradients)]
    執行 Integrated Gradients -> 儲存 10 張 PNG 疊加圖
    """
    print("\n--- [Node: Post-Processing (XAI) - V10 (Integrated Gradients)] ---")
    try:
        if PIPELINE_MODEL is None:
            raise RuntimeError("模型未被成功加載, 無法執行 XAI。")

        nii_path = state.get("t1_native_path")
        if not nii_path:
            return {"error": "Missing 't1_native_path' in state."}
            
        base_output_dir = state.get("output_dir", "output")
        if not os.path.isabs(base_output_dir):
             base_output_dir = os.path.join(PROJECT_ROOT, base_output_dir)
        xai_png_output_dir = os.path.join(base_output_dir, "xai_visualizations")
        os.makedirs(xai_png_output_dir, exist_ok=True)
        print(f"  XAI PNGs 將儲存到: {xai_png_output_dir}")

        target_class_idx = 1 if state.get("classification_result") == "AD" else 0
        print(f"  XAI 目標: class {target_class_idx} ({state.get('classification_result', 'N/A')})")

        # 1. 預處理
        slices_array = preprocessing.preprocess_nii_to_slices(nii_path)
        if slices_array is None:
            return {"error": f"Failed to preprocess NIfTI file: {nii_path}"}
        slices_tensor = torch.tensor(slices_array, dtype=torch.float32) / 255.0
        slices_tensor = slices_tensor.unsqueeze(0).to(DEVICE)
        slices_tensor.requires_grad = True # IG 需要梯度

        # 2. [V10] 執行 Integrated Gradients
        # ig_attributions: (10, 128, 128)
        ig_attributions = xai.calculate_integrated_gradients(
            model=PIPELINE_MODEL,
            model_input_slices=slices_tensor,
            target_class_idx=target_class_idx
        )
        if ig_attributions is None:
            return {"error": "Integrated Gradients calculation failed."}

        # 3. [V10] 儲存 2D 視覺化 PNG 檔案
        png_paths = xai.save_2d_overlay_visualizations(
            attributions_128=ig_attributions, # 傳入 IG 歸因
            original_nii_path=nii_path,
            output_dir=xai_png_output_dir
        )
        
        if png_paths is None:
            return {"error": "Failed to save 2D overlay PNGs."}

        # 4. 返回 langgraph 需要的結果
        print("  XAI (V10) 後處理節點成功完成。")
        return {
            "visualization_paths": png_paths
        }

    except Exception as e:
        print(f"  錯誤: {e}")
        return {"error": f"Post-processing node failed: {e}\n{traceback.format_exc()}"}