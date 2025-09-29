import os
import subprocess
import re
import json

# --- 設定 ---
project_base_dir = './data/cardinal_tien'
raw_data_dir = os.path.join(project_base_dir, 'raw')
output_dir = os.path.join(project_base_dir, 'nifti')
dcm2niix_path = 'dcm2niix'

# --- 主程式邏輯 ---
def convert_to_bids_style(raw_dir, out_dir, converter_path):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"已建立輸出目錄: {out_dir}")

    print("--- 開始進行轉換 ---")
    
    for subject_folder_name in os.listdir(raw_dir):
        match = re.search(r'TPMIC\d+', subject_folder_name)
        if not match:
            continue
            
        subject_id = match.group(0)
        subject_raw_path = os.path.join(raw_dir, subject_folder_name)

        if os.path.isdir(subject_raw_path):
            print(f"\n--- 正在處理受試者: {subject_id} ---")
            for scan_folder_name in os.listdir(subject_raw_path):
                dicom_source_path = os.path.join(subject_raw_path, scan_folder_name)
                if not os.path.isdir(dicom_source_path):
                    continue

                output_sub_dir = ''
                output_filename = ''

                if 'T1_3D_MPRAGE_SAG' in scan_folder_name:
                    output_sub_dir = os.path.join(out_dir, f'sub-{subject_id}', 'anat')
                    # ### 修正 1：移除檔名中多餘的 'a' ###
                    output_filename = f'sub-{subject_id}_T1w' 
                    print(f"  > 發現 T1 結構像: {scan_folder_name}")

                elif 'EP2D_FID_BOLD_REST' in scan_folder_name:
                    output_sub_dir = os.path.join(out_dir, f'sub-{subject_id}', 'func')
                    # ### 修正 1：移除檔名中多餘的 'a' ###
                    output_filename = f'sub-{subject_id}_task-rest_bold'
                    print(f"  > 發現 resting-state fMRI: {scan_folder_name}")
                else:
                    print(f"  > 跳過未知類型的資料夾: {scan_folder_name}")
                    continue

                if not os.path.exists(output_sub_dir):
                    os.makedirs(output_sub_dir)

                command = [converter_path, '-o', output_sub_dir, '-f', output_filename, '-z', 'y', dicom_source_path]

                try:
                    print(f"    執行轉換中...")
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"    成功轉換並儲存於: {output_sub_dir}")
                except subprocess.CalledProcessError as e:
                    print(f"    !!! 轉換失敗: {dicom_source_path}")
                    print(f"    錯誤訊息: {e.stderr.strip()}")

    print("\n--- 所有受試者轉換完畢，正在建立 BIDS 輔助檔案... ---")
    
    # ### 修正 2：將 Authors 欄位改為列表格式 ###
    bids_description = {
      "Name": "Cardinal Tien Hospital fMRI Study",
      "BIDSVersion": "1.8.0",
      "DatasetType": "raw",
      "License": "CC0",
      "Authors": ["Morris"] # 改為列表，即使只有一人
    }
    desc_file_path = os.path.join(out_dir, 'dataset_description.json')
    with open(desc_file_path, 'w', encoding='utf-8') as f:
        json.dump(bids_description, f, ensure_ascii=False, indent=4)
    print(f"--- 成功建立檔案: {desc_file_path} ---")

    # ### 修正 3：自動建立 README 檔案 ###
    readme_content = "This dataset contains structural (T1w) and resting-state functional (task-rest) MRI data.\n"
    readme_file_path = os.path.join(out_dir, 'README')
    with open(readme_file_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"--- 成功建立檔案: {readme_file_path} ---")

    print("\n--- 所有處理完成！ ---")

# --- 執行轉換 ---
if __name__ == "__main__":
    convert_to_bids_style(raw_data_dir, output_dir, dcm2niix_path)