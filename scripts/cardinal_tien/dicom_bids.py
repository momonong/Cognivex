import os
import shutil
import glob
import json

# --- 1. 設定路徑 ---
NIFTI_SOURCE_DIR = '/Volumes/3T-disk/fMRI/cardian_tien/nifti_output'
BIDS_ROOT_DIR = '/Volumes/3T-disk/fMRI/cardian_tien/bids_data' # 一個全新的資料夾

# --- 2. 建立 dataset_description.json ---
def create_dataset_description(bids_root):
    desc = {
        "Name": "Cognivex ADNI-like Dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "License": "n/a",
        "Authors": ["Morris"] # (你的名字)
    }
    desc_path = os.path.join(bids_root, 'dataset_description.json')
    if not os.path.exists(desc_path):
        with open(desc_path, 'w') as f:
            json.dump(desc, f, indent=4)
        print(f"Created {desc_path}")

# --- 3. 輔助函數：安全地複製/連結檔案 ---
# (改用 symlink，速度快，不佔額外空間)
def safe_link(src_path, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) or os.path.lexists(dest_path):
        os.remove(dest_path) # 移除舊的 link
    os.symlink(src_path, dest_path)
    # print(f"  Linking: {os.path.basename(src_path)} -> {dest_path}")

# --- 4. 主程式 ---
def main():
    print(f"--- BIDS Reorganization Started ---")
    os.makedirs(BIDS_ROOT_DIR, exist_ok=True)
    create_dataset_description(BIDS_ROOT_DIR)
    
    # 遍歷 nifti_output 裡的所有 'sub-...' 資料夾
    for subject_folder in sorted(glob.glob(os.path.join(NIFTI_SOURCE_DIR, 'sub-*'))):
        
        subject_name = os.path.basename(subject_folder)
        # BIDS 格式不喜歡 ID 中的底線, 我們把它移除
        bids_id = subject_name.replace('_', '') # e.g., 'sub-TPMIC03002F291YPAN'
        
        print(f"\nProcessing: {subject_name}  ->  {bids_id}")
        
        bids_anat_dir = os.path.join(BIDS_ROOT_DIR, bids_id, 'anat')
        bids_func_dir = os.path.join(BIDS_ROOT_DIR, bids_id, 'func')
        
        # 1. 處理 T1 影像 (anat)
        t1_files = glob.glob(os.path.join(subject_folder, 'T1_3D_mprage_SAG*.nii.gz'))
        t1_json = glob.glob(os.path.join(subject_folder, 'T1_3D_mprage_SAG*.json'))
        
        if t1_files:
            dest_nii = os.path.join(bids_anat_dir, f"{bids_id}_T1w.nii.gz")
            safe_link(os.path.abspath(t1_files[0]), dest_nii) # 使用絕對路徑
            if t1_json:
                dest_json = os.path.join(bids_anat_dir, f"{bids_id}_T1w.json")
                safe_link(os.path.abspath(t1_json[0]), dest_json)
        else:
            print(f"  Warning: No T1w file found for {subject_name}")

        # 2. 處理 fMRI 影像 (func)
        func_files = glob.glob(os.path.join(subject_folder, 'ep2d_fid_bold_REST*.nii.gz'))
        func_json = glob.glob(os.path.join(subject_folder, 'ep2d_fid_bold_REST*.json'))
        
        if func_files:
            dest_nii = os.path.join(bids_func_dir, f"{bids_id}_task-rest_bold.nii.gz")
            safe_link(os.path.abspath(func_files[0]), dest_nii)
            if func_json:
                dest_json = os.path.join(bids_func_dir, f"{bids_id}_task-rest_bold.json")
                safe_link(os.path.abspath(func_json[0]), dest_json)
        else:
            print(f"  Warning: No BOLD/REST file found for {subject_name}")

    print("\n--- BIDS Reorganization Complete ---")
    print(f"BIDS data is ready in: {BIDS_ROOT_DIR}")

if __name__ == "__main__":
    main()