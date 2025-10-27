import pandas as pd

# --- 請修改這裡 ---
# 請將 'your_file.xlsx' 換成你實際的 Excel 檔案名稱
file_name = '/Volumes/3T-disk/fMRI/cardian_tien/TPMIC03/MRIsubjectList.xlsx'
# --- 修改結束 ---

try:
    # 讀取 Excel 檔案
    # openpyxl 是 pandas 讀取 .xlsx 檔案時需要用到的套件
    df = pd.read_excel(file_name, engine='openpyxl')
except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{file_name}'。")
    print("請確認檔案名稱是否正確，且和這個 Python 腳本在同一個資料夾中。")
    exit()
except Exception as e:
    print(f"讀取檔案時發生錯誤：{e}")
    print("請確認你已安裝 'openpyxl' 函式庫 (pip install openpyxl)")
    exit()

print(f"--- 檔案 '{file_name}' 分析報告 ---")

# 1. 基本資訊
print("\n--- 1. 檔案基本資訊 ---")
print(f"總共有 {df.shape[0]} 筆紀錄 (rows)")
print(f"總共有 {df.shape[1]} 個欄位 (columns)")
print("\n所有欄位名稱：")
print(df.columns.to_list())

# 2. 資料預覽
print("\n--- 2. 資料內容預覽 (前 5 筆) ---")
print(df.head())

# 3. 資料型態與缺失值
print("\n--- 3. 欄位資料型態與缺失值 (Missing Values) ---")
# df.info() 會印出每個欄位的 "非空值" 數量與資料型態
df.info()

# 4. 關鍵「類別」欄位分析
print("\n--- 4. 關鍵「類別」欄位分析 (統計各種標籤的數量) ---")

# 分析「目前診斷」
if '目前診斷' in df.columns:
    print("\n[目前診斷] 欄位分佈：")
    print(df['目前診斷'].value_counts(dropna=False)) # dropna=False 會把 "缺失值" 也算進去
else:
    print("\n找不到 '目前診斷' 欄位。")

# 分析「Conversion to」
if 'Conversion to' in df.columns:
    print("\n[Conversion to] 欄位分佈：")
    # 這個欄位可能混雜日期和文字，value_counts 很適合拿來看
    print(df['Conversion to'].value_counts(dropna=False))
else:
    print("\n找不到 'Conversion to' 欄位。")

# 5. 關鍵「ID」欄位分析
print("\n--- 5. 關鍵「ID」欄位分析 ---")
if 'CTH_ID' in df.columns:
    unique_subjects = df['CTH_ID'].nunique()
    print(f"\n總共有 {unique_subjects} 位獨立的受試者 (Unique CTH_ID)。")
    print("每位受試者的掃描次數 (追蹤次數) 分佈：")
    print(df['CTH_ID'].value_counts())
else:
    print("\n找不到 'CTH_ID' (受試者ID) 欄位。")

if '亞東案件編號' in df.columns:
    unique_scans = df['亞東案件編號'].nunique()
    print(f"\n總共有 {unique_scans} 筆獨立的掃描 (Unique 亞東案件編號)。")
    if unique_scans != df.shape[0]:
        print(f"注意：總紀錄數 ({df.shape[0]}) 和獨立掃描ID數 ({unique_scans}) 不一致，可能有重複的掃描紀錄。")
else:
    print("\n找不到 '亞東案件編號' (掃描ID) 欄位。")


# 6. 關鍵「數值」欄位分析
print("\n--- 6. 關鍵「數值」欄位分析 (描述性統計) ---")

# 分析「追蹤時間(年)」
if '追蹤時間(年)' in df.columns:
    # 確保該欄位是數值型態，無法轉換的會變成 NaT (Not a Number)
    df['追蹤時間(年)'] = pd.to_numeric(df['追蹤時間(年)'], errors='coerce')
    print("\n[追蹤時間(年)] 欄位統計：")
    # .describe() 會顯示平均值、標準差、最小值、最大值等
    print(df['追蹤時間(年)'].describe())
else:
    print("\n找不到 '追蹤時間(年)' 欄位。")

# 分析「年齡」
# 從「生日」和「MRI」日期計算
if '生日' in df.columns and 'MRI' in df.columns:
    try:
        # 轉換為日期格式，無法轉換的會變成 NaT (Not a Time)
        birth_date = pd.to_datetime(df['生日'], errors='coerce')
        mri_date = pd.to_datetime(df['MRI'], errors='coerce')
        
        # 計算年齡
        age = (mri_date - birth_date) / pd.Timedelta(days=365.25)
        age.name = "Age_at_MRI" # 給這個新的 Series 一個名字
        
        print("\n[掃描時年齡 (Age_at_MRI)] 欄位統計 (自動計算)：")
        print(age.describe())
    except Exception as e:
        print(f"\n計算年齡時出錯：{e}")
        print("可能是 '生日' 或 'MRI' 欄位的日期格式無法被辨識。")


print("\n--- 分析報告結束 ---")