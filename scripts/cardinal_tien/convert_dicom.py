import pandas as pd
import os
import subprocess
import sys

# --- 1. 設定你的路徑 ---

# 你的「黃金清單」(包含 124 個 scan_id)
# (請確認這個相對路徑是正確的)
SUBJECT_LIST_FILE = 'scripts/cardinal_tien/label.csv'

# 你存放所有 TPMIC... 資料夾的根目錄
DICOM_ROOT_DIR = '/Volumes/3T-disk/fMRI/cardian_tien/TPMIC03'

# 你「指定」的 NIfTI 輸出根目錄
NIFTI_OUTPUT_DIR = '/Volumes/3T-disk/fMRI/cardian_tien/nifti_output'

# --- 2. 輔助函數：自動搜尋 Session 資料夾 ---

def find_dicom_session_dir(subject_root_path):
    """
    在受試者資料夾內，自動搜尋包含 T1 和 fMRI 序列的
    那個「session」資料夾 (例如 'CTH_BRAIN_...')
    """
    print(f"   searching in: {subject_root_path}")
    for root, dirs, files in os.walk(subject_root_path):
        # 建立一個包含所有子資料夾名稱的 set
        dir_contents = set(dirs)
        
        # 檢查這個資料夾是否 "同時" 包含 T1 和 fMRI 序列資料夾
        # (我們用 startswith 來比對，比較保險)
        found_t1 = any(d.startswith("T1_3D_MPRAGE_SAG") for d in dir_contents)
        found_fmri = any(d.startswith("EP2D_FID_BOLD_REST") for d in dir_contents)
        
        if found_t1 and found_fmri:
            print(f"  Found session folder: {root}")
            return root # 找到了！這就是 dcm2niix 要處理的路徑
            
    return None # 遍歷完都沒找到

# --- 3. 主程式：執行轉換 ---

def main():
    # 檢查 dcm2niix 是否安裝
    try:
        subprocess.run(["dcm2niix", "-h"], capture_output=True, check=True)
    except FileNotFoundError:
        print("錯誤：找不到 'dcm2niix' 指令。")
        print("請確認你已經安裝 dcm2niix 並且它在你的系統 PATH 中。")
        sys.exit(1)
        
    # 建立輸出的根目錄
    os.makedirs(NIFTI_OUTPUT_DIR, exist_ok=True)
    
    # 讀取受試者清單
    try:
        df = pd.read_csv(SUBJECT_LIST_FILE)
    except FileNotFoundError:
        print(f"錯誤：找不到受試者清單檔案 '{SUBJECT_LIST_FILE}'")
        sys.exit(1)
        
    scan_ids = df['scan_id'].tolist()
    print(f"--- 總共 {len(scan_ids)} 筆資料準備處理 ---")
    
    success_count = 0
    fail_count = 0
    
    # 迴圈處理每一筆資料
    for scan_id in scan_ids:
        print(f"\n--- [ {success_count+fail_count+1} / {len(scan_ids)} ] Processing Subject: {scan_id} ---")
        
        # 1. 找到 DICOM 來源資料夾
        subject_dicom_root = os.path.join(DICOM_ROOT_DIR, scan_id)
        if not os.path.isdir(subject_dicom_root):
            print(f"  Warning: 來源資料夾不存在: {subject_dicom_root}. SKIPPING.")
            fail_count += 1
            continue
            
        # 2. 自動搜尋 session 資料夾
        session_dir = find_dicom_session_dir(subject_dicom_root)
        if not session_dir:
            print(f"  Warning: 在 {subject_dicom_root} 中找不到 T1/fMRI 序列. SKIPPING.")
            fail_count += 1
            continue
            
        # 3. 準備 NIfTI 輸出資料夾 (BIDS-like 命名)
        subject_nifti_output = os.path.join(NIFTI_OUTPUT_DIR, f"sub-{scan_id}")
        os.makedirs(subject_nifti_output, exist_ok=True)
        
        # 4. 準備 dcm2niix 指令
        cmd = [
            "dcm2niix",
            "-o", subject_nifti_output, # 輸出資料夾
            "-f", "%p_%s",            # 檔名格式: 協定名_序列號
            "-z", "y",                # 壓縮成 .nii.gz
            "-b", "y",                # 產生 BIDS .json sidecar
            session_dir               # 包含所有序列的輸入資料夾
        ]
        
        print(f"  Running dcm2niix...")
        # print(f"  CMD: {' '.join(cmd)}") # 如果你需要 debug，可以取消這行註解
        
        # 5. 執行指令
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
            print("  Conversion successful.")
            # print(result.stdout) # 印出 dcm2niix 的詳細輸出
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: dcm2niix 轉換失敗 for {scan_id}:")
            print(e.stderr)
            fail_count += 1
            
    # --- 迴圈結束，印出總結 ---
    print("\n--- 全部轉換完成！ ---")
    print(f"成功: {success_count} 筆")
    print(f"失敗/跳過: {fail_count} 筆")
    print(f"所有 NIfTI 檔案已存放在: {NIFTI_OUTPUT_DIR}")

# 執行主程式
if __name__ == "__main__":
    main()