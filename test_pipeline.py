# test_papermodel_pipeline.py
import os
import json
from dotenv import load_dotenv

# 確保 .env 檔案被加載 (它應該在同一個根目錄)
load_dotenv()

# --- 導入你的新節點 ---
# (這假設你的 app 資料夾在 python 路徑上)
# 你可能需要 `export PYTHONPATH=$PYTHONPATH:.`
# 或者我們在腳本中手動添加
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.core.papermodel_pipeline.nodes import (
    run_inference_node, 
    run_post_processing_node,
    PIPELINE_MODEL # 檢查模型是否加載
)

# --- 測試配置 ---
# *** 請修改這個路徑，指向你用來測試的 T1 NIfTI 檔案 ***
TEST_NIFTI_FILE = "/Volumes/3T-disk/fMRI/Model/sMRI_data/AD/T1_3D_MPRAGE_SAG_0003_008/T1_3D_MPRAGE_SAG_0003_008_T1_3D_mprage_SAG_20231213144131_3b.nii"
TEST_SUBJECT_ID = "test_subject_008"
TEST_OUTPUT_DIR = f"output/pipeline_test_run_{TEST_SUBJECT_ID}"

def main_test():
    print("--- 開始獨立測試 PaperModel Pipeline ---")
    
    if PIPELINE_MODEL is None:
        print("!!! 測試失敗：模型未能加載，請檢查 nodes.py 中的錯誤訊息和 .env 路徑。")
        return

    # 1. 建立初始狀態 (Mock State)
    state = {
        "t1_native_path": TEST_NIFTI_FILE,
        "subject_id": TEST_SUBJECT_ID,
        "output_dir": TEST_OUTPUT_DIR
    }
    
    # 2. 檢查輸入檔案
    if not os.path.exists(TEST_NIFTI_FILE):
        print(f"!!! 測試失敗：測試 NIfTI 檔案未找到: {TEST_NIFTI_FILE}")
        return
    print(f"  測試檔案: {TEST_NIFTI_FILE}")

    # 3. 執行推論節點
    print("\n[--- 執行 推論 (Inference) 節點 ---]")
    inference_result = run_inference_node(state)
    
    if "error" in inference_result:
        print(f"!!! 推論節點失敗: {inference_result['error']}")
        return
        
    print(f"  推論結果: {inference_result['classification_result']}")
    
    # 將推論結果更新回 state, 供下一節點使用
    state.update(inference_result)
    
    # 4. 執行後處理節點
    print("\n[--- 執行 後處理 (Post-Processing) 節點 ---]")
    post_processing_result = run_post_processing_node(state)
    
    if "error" in post_processing_result:
        print(f"!!! 後處理節點失敗: {post_processing_result['error']}")
        return

    # 5. 打印最終 XAI 結果
    print("\n[--- 測試成功！---]")
    print("  XAI 腦區分析 (Top 5):")
    
    regions = post_processing_result.get("activated_regions", [])
    if regions:
        for i, region in enumerate(regions[:5]):
            print(f"    {i+1}. {region['region_name']} (Score: {region['activation_score']:.4f})")
    else:
        print("    (沒有找到高於閾值的激活腦區)")

    print("\n  生成的 Artifacts:")
    artifact_paths = post_processing_result.get("xai_artifact_paths", {})
    if artifact_paths:
        for key, path in artifact_paths.items():
            print(f"    - {key}: {path}")
            
        # --- 關鍵驗證步驟 ---
        qc_path = artifact_paths.get("qc_warped_t1")
        if qc_path and os.path.exists(qc_path):
            print(f"\n*** 關鍵驗證 ***")
            print(f"請在 FSLeyes (或其他 NIfTI 檢視器) 中打開以下兩個檔案：")
            print(f" 1. 你的 MNI 模板 (e.g., data/affine/mni152_template.nii.gz)")
            print(f" 2. QC 檔案: {qc_path}")
            print(f"請確認 T1 影像 (QC) 是否已成功對齊到 MNI 模板！")
        else:
            print(f"\n*** 警告：未找到 QC 檔案 {qc_path} ***")
            
    else:
        print("    (沒有生成 Artifacts)")

if __name__ == "__main__":
    main_test()