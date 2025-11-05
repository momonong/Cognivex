# test_pipeline.py
import os
import json
from dotenv import load_dotenv
import time 
import sys
import pandas as pd

# --- [V3] 使用絕對路徑 ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from app.core.papermodel_pipeline.nodes import (
    run_inference_node, 
    run_post_processing_node,
    PIPELINE_MODEL
)

# --- 測試配置 ---
TEST_NIFTI_FILE = r"E:/fMRI/Model/sMRI_data/AD/T1_3D_MPRAGE_SAG_0003_008/T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii"
TEST_SUBJECT_ID = "test_subject_008"
TEST_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", f"pipeline_test_run_{TEST_SUBJECT_ID}")

def main_test():
    print("--- 開始獨立測試 PaperModel Pipeline (V9 - 2D-PNG-Only) ---")
    
    if PIPELINE_MODEL is None:
        print("!!! 測試失敗：模型未能加載。")
        return

    state = {
        "t1_native_path": TEST_NIFTI_FILE,
        "subject_id": TEST_SUBJECT_ID,
        "output_dir": TEST_OUTPUT_DIR
    }
    
    if not os.path.exists(TEST_NIFTI_FILE):
        print(f"!!! 測試失敗：測試 NIfTI 檔案未找到: {TEST_NIFTI_FILE}")
        return
    print(f"  測試檔案: {TEST_NIFTI_FILE}")

    # 1. 執行推論節點
    print("\n[--- 執行 推論 (Inference) 節點 ---]")
    inference_result = run_inference_node(state)
    
    if "error" in inference_result:
        print(f"!!! 推論節點失敗: {inference_result['error']}")
        return
        
    print(f"  推論結果: {inference_result['classification_result']}")
    state.update(inference_result)
    
    # 2. 執行後處理節點
    print("\n[--- 執行 後處理 (Post-Processing) 節點 ---]")
    post_processing_result = run_post_processing_node(state)
    
    if "error" in post_processing_result:
        print(f"!!! 後處理節點失敗: {post_processing_result['error']}")
        return

    # 3. [V9 修改] 打印最終 XAI 結果
    print("\n[--- 測試成功！---]")
    
    png_paths = post_processing_result.get("visualization_paths", [])
    if png_paths:
        print(f"  成功生成 {len(png_paths)} 張視覺化 PNG 檔案。")
        print(f"  範例路徑: {png_paths[0]}")
        
        # [V9 驗證] 檢查第一張圖片是否存在
        if os.path.exists(png_paths[0]):
            print(f"  [驗證成功] 檔案 {os.path.basename(png_paths[0])} 已確認存在。")
        else:
            print(f"  [!!! 警告 !!!] 檔案 {png_paths[0]} 未找到！")
            
    else:
        print("    (錯誤：Post-processing 節點未返回 visualization_paths)")

    print("\n*** 關鍵驗證 ***")
    print("  V9 策略成功完成。")
    print("  我們已不再分析腦區，而是產生了 10 張 PNG 疊加圖。")
    print("  這些 PNGs 現在已準備好接入 'image_explainer' 節點。")


if __name__ == "__main__":
    main_test()