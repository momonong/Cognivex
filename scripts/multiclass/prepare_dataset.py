import pandas as pd
import os
import glob
import re 

# ====================================================================
# 【1. 設定與配置】(保持不變)
# ====================================================================

DCM2NIIX_PATH = r"D:\tools\dcm2niix.exe" 
CSV_FILE_PATH = "E:/fMRI/cardian_tien/samsung/MRIsubjectList.xlsx" 
ID_COLUMN_NAME = '亞東案件編號'
DIAG_COLUMN_NAME = '目前診斷'
RAW_DATA_ROOT_DIRS = [
    "E:/fMRI/cardian_tien/samsung",
    "E:/fMRI/cardian_tien/wd"
]
NEW_DATASET_ROOT = "E:/fMRI/Model/sMRI_data_3Class/"

# ====================================================================
# 【2. 輔助函數】(保持不變)
# ====================================================================

def find_subject_folder(subject_id, root_dirs):
    pattern = re.compile(f"^{re.escape(subject_id)}(_.*|-.*)?$", re.IGNORECASE) 
    found_folders = []
    for root_dir in root_dirs:
        try:
            for folder_name in os.listdir(root_dir):
                if pattern.match(folder_name):
                    full_path = os.path.join(root_dir, folder_name)
                    if os.path.isdir(full_path):
                        found_folders.append(full_path)
        except FileNotFoundError:
            print(f"   ⚠️ 警告：根目錄 {root_dir} 不存在。")
            continue 
    if not found_folders: return None 
    for folder_path in found_folders:
        folder_name = os.path.basename(folder_path)
        if folder_name.upper() == subject_id.upper(): 
            return folder_path
    return found_folders[0]

def find_t1_mprage_folder(subject_folder):
    for root, dirs, files in os.walk(subject_folder):
        for d in dirs:
            if "T1_3D_MPRAGE" in d.upper(): 
                return os.path.join(root, d)
    return None

def normalize_diagnosis(diag_str):
    diag_str = str(diag_str).strip() # 保留大小寫以便精確匹配
    
    # 1. 檢查 AD (精確匹配)
    if diag_str == "Alzheimer’s disease" or diag_str == "Alzheimer's disease":
        return "AD"
        
    # 2. 檢查 NC (精確匹配)
    if diag_str == "Normal":
        return "NC"
        
    # 3. 檢查 MCI (包含匹配, 忽略大小寫)
    if "MCI" in diag_str.upper():
        return "MCI"
        
    return "Unknown"

# ====================================================================
# 【3. 主執行腳本 (🚨 核心修正點)】
# ====================================================================

def create_dataset_script():
    print(f"--- Cognivex 計畫 A (3-Class) V7 資料集準備 ---")
    
    if not os.path.exists(DCM2NIIX_PATH):
        print(f"🚨 致命錯誤：在 '{DCM2NIIX_PATH}' 找不到 dcm2niix.exe。")
        return

    os.makedirs(os.path.join(NEW_DATASET_ROOT, "NC"), exist_ok=True)
    os.makedirs(os.path.join(NEW_DATASET_ROOT, "MCI"), exist_ok=True)
    os.makedirs(os.path.join(NEW_DATASET_ROOT, "AD"), exist_ok=True)

    try:
        df = pd.read_excel(CSV_FILE_PATH)
        print(f"✅ 成功讀取 Excel 檔案: {CSV_FILE_PATH}")
    except Exception as e:
        print(f"🚨 錯誤：無法讀取 Excel 檔案。錯誤: {e}"); return

    if ID_COLUMN_NAME not in df.columns or DIAG_COLUMN_NAME not in df.columns:
        print(f"🚨 錯誤：在 Excel 檔案中找不到 '{ID_COLUMN_NAME}' 或 '{DIAG_COLUMN_NAME}' 欄位。")
        print(f"   目前找到的欄位有: {list(df.columns)}")
        return
    print("✅ 欄位名稱驗證成功。")

    commands = [] 
    mapping_data = []
    subject_counter = 0
    count_nc, count_mci, count_ad = 0, 0, 0
    
    for index, row in df.iterrows():
        subject_id = str(row[ID_COLUMN_NAME]).strip()
        diagnosis = normalize_diagnosis(row[DIAG_COLUMN_NAME])
        
        if diagnosis == "Unknown" or not subject_id or subject_id.lower() == 'nan':
            continue 

        print(f"\n-> 正在搜尋 Subject ID (模糊匹配): {subject_id} (診斷: {diagnosis})")
        
        subject_folder = find_subject_folder(subject_id, RAW_DATA_ROOT_DIRS)
        
        if not subject_folder:
            print(f"   ⚠️ 警告：找不到 {subject_id} 的任何匹配資料夾。")
            continue
        
        t1_dicom_folder = find_t1_mprage_folder(subject_folder)
        if not t1_dicom_folder:
            print(f"   ⚠️ 警告：在 {subject_folder} 中找不到 T1_3D_MPRAGE 資料夾。")
            continue
            
        print(f"   ✅ 找到 T1 DICOM 資料夾: {t1_dicom_folder}")

        subject_counter += 1
        if diagnosis == "NC": count_nc += 1
        elif diagnosis == "MCI": count_mci += 1
        elif diagnosis == "AD": count_ad += 1

        output_nii_dir = os.path.join(NEW_DATASET_ROOT, diagnosis)
        output_nii_filename = f"sub_{subject_counter:04d}_T1" 
        
        # 🚨 核心修正點：
        # 將 -b o (BIDS Only) 修正為 -b n (BIDS No)
        command = f'"{DCM2NIIX_PATH}" -o "{output_nii_dir}" -f "{output_nii_filename}" -z y -b n -m y "{t1_dicom_folder}"'
        commands.append(command)
        
        mapping_data.append({
            'new_id': output_nii_filename,
            'original_id': subject_id,
            'diagnosis': diagnosis
        })

    # 9. 將所有指令寫入一個 .bat 檔案
    bat_file_path = "run_dicom_conversion.bat"
    with open(bat_file_path, 'w', encoding='utf-8') as f:
        f.write("@echo off\n")
        f.write("echo Starting T1 DICOM to NIfTI conversion...\n")
        f.write("\n".join(commands))
        f.write("\necho All conversions finished.\n")
        f.write("pause\n")

    # 10. 儲存對照表
    mapping_csv_path = os.path.join(NEW_DATASET_ROOT, "_dataset_mapping.csv")
    try:
        df_mapping = pd.DataFrame(mapping_data)
        df_mapping.to_csv(mapping_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 成功儲存 ID 對照表至: {mapping_csv_path}")
    except Exception as e:
        print(f"\n⚠️ 警告：儲存 ID 對照表失敗。錯誤: {e}")

    print(f"\n--- 任務完成 ---")
    print(f"✅ 成功生成 {len(commands)} 筆轉換指令。")
    print(f"   - NC (健康對照組): {count_nc} 筆")
    print(f"   - MCI (輕度認知障礙): {count_mci} 筆")
    print(f"   - AD (阿茲海默症): {count_ad} 筆")
    print(f"請執行 '{bat_file_path}' 檔案來開始建立您的 3-Class 資料集。")
    print(f"輸出目錄: {NEW_DATASET_ROOT}")

if __name__ == "__main__":
    try: import pandas
    except ImportError: print("缺少 'pandas' 套件，請執行: pip install pandas")
    try: import openpyxl
    except ImportError: print("缺少 'openpyxl' 套件 (用於讀取 .xlsx)，請執行: pip install openpyxl")
        
    create_dataset_script()