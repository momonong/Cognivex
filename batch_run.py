# batch_run.py
import json
import glob
import argparse
from pathlib import Path
from typing import List

# 從您的工作流檔案中匯入已編譯好的 app 物件
from app.graph.workflow import app

def run_batch_analysis(subjects_to_run: List[str], model_name: str, llm_provider: str):
    """
    對指定的一系列受試者執行批次分析。
    """
    model_paths = {
        "capsnet": "model/capsnet/best_capsnet_rnn.pth",
        "mcadnnet": "model/macadnnet/best_model.pth" # 假設這是 mcadnnet 的路徑
    }
    model_path = model_paths.get(model_name)
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"找不到模型 '{model_name}' 的權重檔案: {model_path}")

    output_dir = Path(f"output/hackathon/run_states")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*50)
    print("🚀 Starting Batch Analysis Pipeline")
    print(f"   Subjects to run: {len(subjects_to_run)}")
    print(f"   Model: {model_name}")
    print(f"   LLM Provider: {llm_provider}")
    print(f"   Output Directory: {output_dir}")
    print("="*50)

    success_count = 0
    failure_count = 0
    total_subjects = len(subjects_to_run)

    for i, subject_id in enumerate(subjects_to_run):
        print(f"\n--- ({i+1}/{total_subjects}) Processing Subject: {subject_id} ---")

        search_pattern = f"data/raw/*/{subject_id}/*.nii.gz"
        nii_file_list = glob.glob(search_pattern)

        if not nii_file_list:
            print(f"[Warning] ❌ No .nii.gz file found for {subject_id}. Skipping.")
            failure_count += 1
            continue
        
        fmri_scan_path = nii_file_list[0]
        print(f"   Found fMRI file: {fmri_scan_path}")

        initial_state = {
            "subject_id": subject_id,
            "fmri_scan_path": fmri_scan_path,
            "model_path": model_path,
            "model_name": model_name,
            "llm_provider": llm_provider, 
            "trace_log": [],
            "error_log": [],
        }

        try:
            final_state = app.invoke(initial_state)
            
            if final_state and not final_state.get("error_log"):
                output_path = output_dir / f"{subject_id}_final_state.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_state, f, indent=2, ensure_ascii=False)
                print(f"   [Success] ✅ Results for {subject_id} saved to {output_path}")
                success_count += 1
            else:
                error_message = final_state.get("error_log", ["Unknown error"])[-1]
                raise Exception(error_message)

        except Exception as e:
            print(f"   [ERROR] ❌ An error occurred while processing {subject_id}: {e}")
            error_log_path = output_dir / f"{subject_id}_error.log"
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Error processing {subject_id}:\n{str(e)}")
            failure_count += 1
            continue

    print("\n" + "="*50)
    print("✅ Batch processing completed!")
    print(f"   Total subjects: {total_subjects}")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failure: {failure_count}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch analysis for the fMRI pipeline.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="capsnet", 
        choices=["capsnet", "mcadnnet"],
        help="The model to use for inference."
    )
    parser.add_argument(
        "--llm", 
        type=str, 
        default="gemini", 
        choices=["gemini", "aws_bedrock", "gpt-oss-20b"],
        help="The LLM provider to use for generation tasks."
    )
    
    args = parser.parse_args()

    # --- 變更點：直接建立一個從 1 到 32 的受試者列表 ---
    subjects_to_run = [f"sub-{i:02d}" for i in range(1, 33)]
    
    run_batch_analysis(subjects_to_run, args.model, args.llm)