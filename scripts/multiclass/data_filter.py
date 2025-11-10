import os
import shutil

# --- 配置區塊 ---
# 根目錄：所有組別資料夾 (NC, AD, MCI) 所在的位置
ROOT_DIR = r"E:\fMRI\Model\sMRI_data_MultiModal"

# 損壞檔案的目標總資料夾
DAMAGED_ROOT = os.path.join(ROOT_DIR, "damaged")

# 所有可能包含損壞資料的組別資料夾名稱
GROUP_DIRS = ["NC", "AD", "MCI"]

# 根據 QC 報告，需要檢查並移動的 Subjects ID 列表
SUBJECTS_TO_CHECK = [
    "sub_0029",
    "sub_0120",
    "sub_0004",
    "sub_0057",
    "sub_0006",
    "sub_0071"
]
# --- 配置區塊結束 ---

def setup_directories():
    """確保頂層 damaged 資料夾及其組別子資料夾存在。"""
    if not os.path.isdir(DAMAGED_ROOT):
        os.makedirs(DAMAGED_ROOT)
        print(f"已創建總 damaged 資料夾: {DAMAGED_ROOT}")
    else:
        print(f"總 damaged 資料夾已存在: {DAMAGED_ROOT}")
        
    for group in GROUP_DIRS:
        group_damaged_dir = os.path.join(DAMAGED_ROOT, group)
        if not os.path.isdir(group_damaged_dir):
            os.makedirs(group_damaged_dir)
            print(f"已創建 damaged 子資料夾: {group_damaged_dir}")


def move_subject_files(subject_id):
    """
    在所有組別資料夾中尋找特定 subject 的檔案並移動。
    """
    print(f"\n--- 檢查並處理病患: {subject_id} ---")
    found_and_moved = False
    
    for group_name in GROUP_DIRS:
        source_dir = os.path.join(ROOT_DIR, group_name)
        destination_dir = os.path.join(DAMAGED_ROOT, group_name)
        
        # 檢查原始組別資料夾是否存在
        if not os.path.isdir(source_dir):
            # print(f"警告: 組別資料夾 {group_name} 不存在，跳過檢查。")
            continue

        # 尋找所有以該 subject ID 開頭的檔案
        # e.g., 'sub_0029_T1.nii.gz', 'sub_0029_DWI.bval'
        
        all_files_in_source = os.listdir(source_dir)
        
        files_to_move = [
            f for f in all_files_in_source 
            if f.startswith(subject_id + "_") and os.path.isfile(os.path.join(source_dir, f))
        ]
        
        if files_to_move:
            print(f"  -> 在 {group_name} 組找到 {len(files_to_move)} 個檔案。")
            for filename in files_to_move:
                source_path = os.path.join(source_dir, filename)
                dest_path = os.path.join(destination_dir, filename)
                
                try:
                    shutil.move(source_path, dest_path)
                    print(f"    > 已移動到 damaged/{group_name}: {filename}")
                    found_and_moved = True
                except Exception as e:
                    print(f"    🚨 移動檔案 {filename} 失敗: {e}")
            
            # 如果在某組找到檔案並移動，則停止在其他組別中尋找，因為一個 subject 應該只屬於一組。
            break 
    
    if not found_and_moved:
        print(f"  - 在所有組別資料夾 ({', '.join(GROUP_DIRS)}) 中均未找到 {subject_id} 的檔案。")


# --- 主要執行區塊 ---
if __name__ == "__main__":
    
    # 步驟 1: 設置資料夾結構
    setup_directories()
    
    print("\n--- 開始執行跨組別檔案移動作業 ---")
    
    # 步驟 2: 處理每個需要檢查的 subject
    for subject in SUBJECTS_TO_CHECK:
        move_subject_files(subject)

    print("\n=== 所有檔案移動作業完成 ===")