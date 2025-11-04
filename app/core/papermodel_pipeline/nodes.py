# app/core/papermodel_pipeline/nodes.py
import torch
import numpy as np
import os
import traceback
from typing import Dict, Any

# 匯入此 pipeline 的所有組件
from . import model
from . import preprocessing
from . import xai

# --- 全局服務加載 (在真實應用中, 應使用依賴注入或單例模式) ---

# 這應該指向你訓練好的模型權重
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "model/fold_3_best_model.pth") 
# 這應該指向你的 MNI 模板
MNI_TEMPLATE_PATH = os.getenv("MNI_TEMPLATE_PATH", "data/affine/mni152_template.nii.gz")
# 這應該指向你的 Atlas (圖譜)
ATLAS_NII_PATH = os.getenv("ATLAS_NII_PATH", "data/AAL3v1_1mm.nii.gz")
# 這應該指向 Atlas 的 JSON 標籤
ATLAS_LABEL_PATH = os.getenv("ATLAS_LABEL_PATH", "data/AAL3v1.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def _load_model_instance():
    """ 輔助函式：加載模型實例 """
    print(f"正在從 {MODEL_WEIGHTS_PATH} 加載模型權重...")
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"模型權重未找到: {MODEL_WEIGHTS_PATH}。請檢查 .env 或 MODEL_WEIGHTS_PATH。")
    
    pipeline_model = model.PaperModel().to(DEVICE)
    pipeline_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    pipeline_model.eval()
    print("模型加載成功並設為 eval() 模式。")
    return pipeline_model

# 預先加載模型 (單例)
try:
    PIPELINE_MODEL = _load_model_instance()
    # 檢查所有依賴的資料檔案是否存在
    for p in [MNI_TEMPLATE_PATH, ATLAS_NII_PATH, ATLAS_LABEL_PATH]:
        if not os.path.exists(p):
            print(f"警告：依賴檔案未找到: {p}。Post-processing 節點可能會失敗。")
            
except Exception as e:
    print(f"致命錯誤：無法加載模型或依賴檔案: {e}")
    PIPELINE_MODEL = None

# --- LangGraph 節點定義 ---

def run_inference_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    [LangGraph 節點 1: 推論]
    接收 T1 NIfTI 路徑, 執行分類。
    
    Input state:
     - t1_native_path (str): 輸入 NIfTI 檔案的路徑
     
    Output state:
     - classification_result (str): "AD" 或 "NC"
     - error (str, optional): 錯誤訊息
    """
    print("\n--- [Node: Inference] ---")
    try:
        if PIPELINE_MODEL is None:
            raise RuntimeError("模型未被成功加載, 無法執行推論。")
            
        nii_path = state.get("t1_native_path")
        if not nii_path:
            return {"error": "Missing 't1_native_path' in state."}

        # 1. 預處理
        slices_array = preprocessing.preprocess_nii_to_slices(nii_path)
        if slices_array is None:
            return {"error": f"Failed to preprocess NIfTI file: {nii_path}"}
            
        slices_tensor = torch.tensor(slices_array, dtype=torch.float32) / 255.0
        slices_tensor = slices_tensor.unsqueeze(0).to(DEVICE) # 增加 Batch 維度 [1, 10, 1, 128, 128]

        # 2. 推論
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
    [LangGraph 節點 2: XAI 後處理 - V2 with QC]
    執行 Grad-CAM -> 3D 投影 -> ANTs -> Resample -> 腦區分析
    """
    print("\n--- [Node: Post-Processing (XAI)] ---")
    try:
        if PIPELINE_MODEL is None:
            raise RuntimeError("模型未被成功加載, 無法執行 XAI。")

        nii_path = state.get("t1_native_path")
        if not nii_path:
            return {"error": "Missing 't1_native_path' in state."}
            
        # [新增] 獲取唯一的 QC 輸出目錄
        # 假設 state 包含 'subject_id' 或我們使用 'output_dir'
        subject_id = state.get("subject_id", "default_subject")
        base_output_dir = state.get("output_dir", "output")
        xai_output_dir = os.path.join(base_output_dir, subject_id, "xai_artifacts")
        os.makedirs(xai_output_dir, exist_ok=True)
        print(f"  XAI artifacts 將儲存到: {xai_output_dir}")

        target_class_idx = 1 if state.get("classification_result") == "AD" else 0
        print(f"  XAI 目標: class {target_class_idx} ({state.get('classification_result', 'N/A')})")

        # 1. 預處理
        slices_array = preprocessing.preprocess_nii_to_slices(nii_path) # [修改]
        if slices_array is None:
            return {"error": f"Failed to preprocess NIfTI file: {nii_path}"}
        slices_tensor = torch.tensor(slices_array, dtype=torch.float32) / 255.0 # [修改]
        slices_tensor = slices_tensor.unsqueeze(0).to(DEVICE)
        slices_tensor.requires_grad = True 

        # 2. 執行 2D Grad-CAM
        stitched_heatmap_2d = xai.run_grad_cam_on_stitched_map(
            model=PIPELINE_MODEL,
            model_input_slices=slices_tensor,
            target_class_idx=target_class_idx
        )
        if stitched_heatmap_2d is None:
            return {"error": "Grad-CAM calculation failed."}

        # 3. 投影回 3D Native Space
        native_heatmap_nii = xai.reproject_heatmap_to_3d(stitched_heatmap_2d, nii_path)
        if native_heatmap_nii is None:
            return {"error": "2D-to-3D reprojection failed."}
            
        # [修改] 儲存 Native Heatmap
        native_heatmap_path = os.path.join(xai_output_dir, "heatmap_native.nii.gz")
        nib.save(native_heatmap_nii, native_heatmap_path)

        # 4. 標準化到 3D MNI Space (ANTs)
        mni_heatmap_ants = xai.normalize_native_to_mni(
            native_t1_path=nii_path,
            native_heatmap_nii=native_heatmap_nii,
            mni_template_path=MNI_TEMPLATE_PATH,
            output_dir=xai_output_dir # [修改] 傳入 QC 路徑
        )
        if mni_heatmap_ants is None:
            return {"error": "ANTs normalization failed."}

        # 5. 重採樣 (Resample) 到 Atlas
        resampled_heatmap_nii = xai.resample_to_atlas(
            mni_heatmap_ants=mni_heatmap_ants,
            atlas_nii_path=ATLAS_NII_PATH
        )
        if resampled_heatmap_nii is None:
            return {"error": "Resampling to atlas grid failed."}
            
        # [修改] 儲存最終的 MNI Heatmap
        mni_heatmap_path = os.path.join(xai_output_dir, "heatmap_mni_resampled.nii.gz")
        nib.save(resampled_heatmap_nii, mni_heatmap_path)

        # 6. 分析腦區
        analysis_df = xai.analyze_brain_regions(
            resampled_heatmap_nii=resampled_heatmap_nii,
            atlas_nii_path=ATLAS_NII_PATH,
            atlas_label_path=ATLAS_LABEL_PATH,
            threshold_percentile=95.0 
        )
        if analysis_df is None:
            return {"error": "Brain region analysis failed."}

        # 7. 返回 langgraph 需要的結果
        print("  XAI 後處理節點成功完成。")
        return {
            "activated_regions": analysis_df.to_dict('records'),
            "xai_artifact_paths": {
                "native_heatmap": native_heatmap_path,
                "mni_heatmap": mni_heatmap_path,
                "qc_warped_t1": os.path.join(xai_output_dir, "qc_t1_warped_to_mni.nii.gz")
            }
        }

    except Exception as e:
        print(f"  錯誤: {e}")
        return {"error": f"Post-processing node failed: {e}\n{traceback.format_exc()}"}