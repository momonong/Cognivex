import pandas as pd
import os
import glob
import re 

# ====================================================================
# 【1. 設定與配置】(已更新)
# ====================================================================

DCM2NIIX_PATH = r"D:\tools\dcm2niix.exe" 
CSV_FILE_PATH = "E:/fMRI/cardian_tien/samsung/MRIsubjectList.xlsx" 
ID_COLUMN_NAME = '亞東案件編號'
DIAG_COLUMN_NAME = '目前診斷'

RAW_DATA_ROOT_DIRS = [
    "E:/fMRI/cardian_tien/samsung",
    "E:/fMRI/cardian_tien/wd"
]
# 🚨 新的輸出目錄
NEW_DATASET_ROOT = "E:/fMRI/Model/sMRI_data_MultiModal/"

# ====================================================================
# 【2. 輔助函數 (🚨 核心修正點)】
# ====================================================================

def find_subject_folder(subject_id, root_dirs):
    """
    在所有根目錄中「模糊」搜尋與 subject_id 匹配的資料夾 (忽略大小寫)。
    """
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
            
    if not found_folders:
        return None 

    # 優先返回精確匹配 (忽略大小寫)，處理基線
    for folder_path in found_folders:
        folder_name = os.path.basename(folder_path)
        if folder_name.upper() == subject_id.upper(): 
            return folder_path
            
    # 如果沒有精確匹配 (例如 Excel 是 'ID_f1' 且資料夾也是 'ID_F1')
    return found_folders[0]

def find_scan_folder(subject_folder, scan_type_keyword):
    """
    在病患資料夾內遞迴搜尋「包含特定關鍵字」的掃描資料夾。
    """
    for root, dirs, files in os.walk(subject_folder):
        for d in dirs:
            # 使用 .upper() 進行不區分大小寫的比對
            if scan_type_keyword.upper() in d.upper(): 
                return os.path.join(root, d)
    return None

def normalize_diagnosis(diag_str):
    diag_str = str(diag_str).strip() # 保留大小寫以便精確匹配
    if diag_str == "Alzheimer’s disease" or diag_str == "Alzheimer's disease": return "AD"
    if diag_str == "Normal": return "NC"
    if "MCI" in diag_str.upper(): return "MCI"
    return "Unknown"

# ====================================================================
# 【3. 主執行腳本 (🚨 核心修正點)】
# ====================================================================

def create_dataset_script():
    print(f"--- Cognivex 計畫 B (Multi-Modal) 資料集準備 ---")
    
    if not os.path.exists(DCM2NIIX_PATH):
        print(f"🚨 致命錯誤：在 '{DCM2NIIX_PATH}' 找不到 dcm2niix.exe。")
        return

    # 建立 NC, MCI, AD 輸出子目錄
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
    count_missing_modality = 0
    
    # 4. 遍歷 Excel 中的每一位病患
    for index, row in df.iterrows():
        subject_id = str(row[ID_COLUMN_NAME]).strip()
        diagnosis = normalize_diagnosis(row[DIAG_COLUMN_NAME])
        
        if diagnosis == "Unknown" or not subject_id or subject_id.lower() == 'nan':
            continue 

        print(f"\n-> 正在搜尋 Subject ID: {subject_id} (診斷: {diagnosis})")
        
        subject_folder = find_subject_folder(subject_id, RAW_DATA_ROOT_DIRS)
        
        if not subject_folder:
            print(f"   ⚠️ 警告：找不到 {subject_id} 的任何匹配資料夾。")
            continue
        
        # 🚨 核心邏輯：我們需要找到「所有」模態
        t1_dicom_folder = find_scan_folder(subject_folder, "T1_3D_MPRAGE")
        t2_dicom_folder = find_scan_folder(subject_folder, "T2_TIRM_TRA_DARK-FLUID")
        # 關鍵字 "DTI" 應該足以捕捉到 "DTI_2SHELLS_..."
        dti_dicom_folder = find_scan_folder(subject_folder, "DTI") 

        # 5. 過濾：確保所有模態都存在
        if not (t1_dicom_folder and t2_dicom_folder and dti_dicom_folder):
            print(f"   ❌ 錯誤：病患 {subject_id} 缺少 T1, T2 或 DTI 其中一種掃描。將跳過此病患。")
            if not t1_dicom_folder: print("      - 缺少 T1_3D_MPRAGE")
            if not t2_dicom_folder: print("      - 缺少 T2_TIRM_TRA_DARK-FLUID")
            if not dti_dicom_folder: print("      - 缺少 DTI")
            count_missing_modality += 1
            continue
            
        print(f"   ✅ 找到 T1: {os.path.basename(t1_dicom_folder)}")
        print(f"   ✅ 找到 T2: {os.path.basename(t2_dicom_folder)}")
        print(f"   ✅ 找到 DTI: {os.path.basename(dti_dicom_folder)}")

        # 6. 更新計數器與匿名化
        subject_counter += 1
        if diagnosis == "NC": count_nc += 1
        elif diagnosis == "MCI": count_mci += 1
        elif diagnosis == "AD": count_ad += 1

        output_nii_dir = os.path.join(NEW_DATASET_ROOT, diagnosis)
        
        # 7. 生成三種 dcm2niix 指令
        
        # T1 指令
        t1_filename = f"sub_{subject_counter:04d}_T1" 
        cmd_t1 = f'"{DCM2NIIX_PATH}" -o "{output_nii_dir}" -f "{t1_filename}" -z y -b n -m y "{t1_dicom_folder}"'
        
        # T2 指令
        t2_filename = f"sub_{subject_counter:04d}_T2_FLAIR"
        cmd_t2 = f'"{DCM2NIIX_PATH}" -o "{output_nii_dir}" -f "{t2_filename}" -z y -b n -m y "{t2_dicom_folder}"'
        
        # DTI 指令 (dcm2niix 會自動生成 _dwi.nii.gz, .bval, .bvec)
        dti_filename = f"sub_{subject_counter:04d}_DWI"
        cmd_dti = f'"{DCM2NIIX_PATH}" -o "{output_nii_dir}" -f "{dti_filename}" -z y -b n -m y "{dti_dicom_folder}"'

        commands.extend([cmd_t1, cmd_t2, cmd_dti])
        
        # 🚨 新增：儲存對照資訊
        mapping_data.append({
            'new_id_base': f"sub_{subject_counter:04d}",
            'original_id': subject_id,
            'diagnosis': diagnosis
        })

    # 9. 將所有指令寫入一個 .bat 檔案
    bat_file_path = "run_multimodal_conversion.bat"
    with open(bat_file_path, 'w', encoding='utf-8') as f:
        f.write("@echo off\n")
        f.write("echo Starting Multi-Modal (T1, T2, DTI) DICOM to NIfTI conversion...\n")
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
    print(f"✅ 成功為 {subject_counter} 位病患生成 {len(commands)} 筆轉換指令。")
    print(f"   - NC (健康對照組): {count_nc} 筆")
    print(f"   - MCI (輕度認知障礙): {count_mci} 筆")
    print(f"   - AD (阿茲海默症): {count_ad} 筆")
    print(f"   (因缺少 T1/T2/DTI 之一而被跳過的病患: {count_missing_modality} 筆)")
    print(f"請執行 '{bat_file_path}' 檔案來開始建立您的 3-Class 多模態資料集。")
    print(f"輸出目錄: {NEW_DATASET_ROOT}")

if __name__ == "__main__":
    try: import pandas
    except ImportError: print("缺少 'pandas' 套件，請執行: pip install pandas")
    try: import openpyxl
    except ImportError: print("缺少 'openpyxl' 套件 (用於讀取 .xlsx)，請執行: pip install openpyxl")
        
    create_dataset_script()