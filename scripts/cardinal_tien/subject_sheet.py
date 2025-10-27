import pandas as pd
import numpy as np

# --- 請修改這裡 ---
file_name = '/Volumes/3T-disk/fMRI/cardian_tien/TPMIC03/MRIsubjectList.xlsx'
# --- 修改結束 ---

try:
    df = pd.read_excel(file_name, engine='openpyxl')
    print(f"成功讀取檔案: {file_name}")
except Exception as e:
    print(f"讀取檔案時發生錯誤：{e}")
    exit()

# --- 1. 資料清理 (Clean Data) ---
print(f"\n--- 1. 開始清理資料 ---")
print(f"原始資料筆數: {len(df)}")

# 步驟 1.1: 移除 79 筆完全空白的紀錄 (以「目前診斷」為基準)
df_cleaned = df.dropna(subset=['目前診斷', '亞東案件編號']).copy()
print(f"移除空白紀錄後筆數: {len(df_cleaned)}")

# 步驟 1.2: 移除「追蹤時間(年)」中的極端錯誤值 (e.g., > 20 年)
# 我們將其設為 NaN (空值)，而不是刪除整筆紀錄
original_max_followup = df_cleaned['追蹤時間(年)'].max()
df_cleaned.loc[df_cleaned['追蹤時間(年)'] > 20, '追蹤時間(年)'] = np.nan
print(f"已修正 '追蹤時間(年)' 的錯誤值 (原最大值: {original_max_followup})")

# 步驟 1.3: 移除重複的「掃描 ID」
# keep='first' 保留第一筆，刪除後面的重複筆
df_cleaned = df_cleaned.drop_duplicates(subset=['亞東案件編號'], keep='first')
print(f"移除重複掃描 ID 後筆數: {len(df_cleaned)}")


# --- 2. 建立「黃金三組」 (Create Groups) ---
print("\n--- 2. 根據診斷建立「黃金三組」 ---")

def define_group(diagnosis):
    diagnosis = str(diagnosis)
    
    # Group 1: Normal Control
    if diagnosis == 'Normal':
        return 'NC'
    
    # Group 2: Alzheimer's Disease
    elif diagnosis == 'Alzheimer’s disease':
        return 'AD'
    
    # Group 3: Mild Cognitive Impairment
    elif diagnosis == 'aMCI' or diagnosis == 'naMCI':
        return 'MCI'
    
    # 其他所有情況，都標記為 "Exclude"
    else:
        return 'Exclude'

# 應用這個函數，建立新的 'Group' 欄位
df_cleaned['Group'] = df_cleaned['目前診斷'].apply(define_group)

print("分組完成！各組筆數統計：")
print(df_cleaned['Group'].value_counts())

# --- 3. 儲存結果 ---
# 只保留我們需要的黃金三組資料
final_df = df_cleaned[df_cleaned['Group'].isin(['NC', 'MCI', 'AD'])].copy()

# 儲存一份乾淨、已分組的 Excel 檔案
output_file = 'scripts/cardinal_tien/subject_list.xlsx'
final_df.to_excel(output_file, index=False, engine='openpyxl')

print(f"\n--- 3. 清理與分組完成！ ---")
print(f"總共 {len(final_df)} 筆資料 (NC, MCI, AD) 已儲存至:")
print(f"{output_file}")

print("\n最終三組的檔案列表 (亞東案件編號):")
print("\nNC 組：")
print(final_df[final_df['Group'] == 'NC']['亞東案件編號'].tolist())
print("\nMCI 組：")
print(final_df[final_df['Group'] == 'MCI']['亞東案件編號'].tolist())
print("\nAD 組：")
print(final_df[final_df['Group'] == 'AD']['亞東案件編號'].tolist())