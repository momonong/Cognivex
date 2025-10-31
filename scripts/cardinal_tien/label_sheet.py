import pandas as pd
import os

# --- 設定 ---
# 1. 我們上一步產生的「黃金清單」檔案
cleaned_list_file = 'scripts/cardinal_tien/subject_list.xlsx'

# 2. 你的 MRI 原始資料夾 (你 ls 的地方)
#    '.' 代表「目前這個資料夾」
raw_data_directory = '/Volumes/3T-disk/fMRI/cardian_tien/TPMIC03' 

# 3. 最終要輸出的「模型輸入清單」檔案名稱
output_file = 'scripts/cardinal_tien/label.csv'

# --- 腳本開始 ---
print("--- 正在生成「模型輸入清單」 ---")

# 1. 讀取「黃金清單」
try:
    df_clean = pd.read_excel(cleaned_list_file)
    print(f"成功讀取: {cleaned_list_file}")
    # 建立一個 {亞東案件編號: Group} 的 "字典" (dictionary)，方便快速查找
    # .set_index().to_dict() 是一種高效的查找方法
    label_map = df_clean.set_index('亞東案件編號')['Group'].to_dict()
    print(f"黃金清單中包含 {len(label_map)} 筆 (NC, MCI, AD) 紀錄。")
except FileNotFoundError:
    print(f"錯誤：找不到 '{cleaned_list_file}'！")
    print("請先執行上一個腳本 (04-clean_and_group.py) 來產生這個檔案。")
    exit()
except Exception as e:
    print(f"讀取 {cleaned_list_file} 時出錯: {e}")
    exit()

# 2. 掃描「實際資料夾」
print(f"正在掃描 '{raw_data_directory}' 中的實際資料夾...")
# os.listdir() 會列出所有檔案和資料夾
# os.path.isdir() 會判斷是否為資料夾
# 我們只抓 'TPMIC' 或 'TMPIC' 開頭的
try:
    all_files_and_dirs = os.listdir(raw_data_directory)
    existing_folders = [d for d in all_files_and_dirs 
                        if os.path.isdir(os.path.join(raw_data_directory, d)) 
                        and (d.startswith('TPMIC') or d.startswith('TMPIC'))]
    print(f"掃描到 {len(existing_folders)} 個相關的資料夾。")
except Exception as e:
    print(f"掃描資料夾時出錯: {e}")
    exit()


# 3. 比對兩份清單，建立最終 Model Input List
print("--- 正在比對清單並建立最終檔案 ---")
model_input_data = [] # 用來存放最終結果

for folder_name in existing_folders:
    # 檢查這個資料夾是否在我們的「黃金清單」中
    if folder_name in label_map:
        # 如果在，就取得它的標籤 (NC/MCI/AD)
        label = label_map[folder_name]
        
        # 把 (資料夾名稱, 標籤) 加入我們的清單
        model_input_data.append({
            'scan_id': folder_name,
            'label': label
        })
    else:
        # 這個資料夾存在，但不在黃金清單中 (例如它是 'VaD' 或 'SCD')
        # 我們就 "忽略" 它
        pass 

# 4. 轉換成 DataFrame 並儲存
if not model_input_data:
    print("錯誤：沒有任何資料夾同時存在於「黃金清單」和「實際資料夾」中。")
    print("請檢查你的檔案路徑和 Excel 內容是否正確。")
else:
    df_final_list = pd.DataFrame(model_input_data)
    
    # 儲存成 CSV 檔案
    df_final_list.to_csv(output_file, index=False)
    
    print("\n--- 完成！ ---")
    print(f"已成功建立 '{output_file}'。")
    print("\n最終模型輸入資料分佈：")
    print(df_final_list['label'].value_counts())
    
    print(f"\n總共有 {len(df_final_list)} 筆資料可供模型訓練/測試。")