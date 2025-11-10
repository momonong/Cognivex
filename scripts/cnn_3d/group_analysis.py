import pandas as pd
import glob
import os
import numpy as np

# ====================================================================
# 【1. 設定與配置】
# ====================================================================

# 🚨 必須與 05 腳本的輸出目錄完全一致
RESULTS_DIR = "output/cnn_3d/final_analysis_results/"
OUTPUT_SUMMARY_FILE = os.path.join(RESULTS_DIR, "_SUMMARY_GLOBAL_ACTIVATION_RANKING.csv")

# ====================================================================
# 【2. 主執行腳本 (已修正)】
# ====================================================================

def summarize_all_results():
    print(f"--- Cognivex V3 匯總分析啟動 ---")
    print(f"正在讀取 {RESULTS_DIR} 中的所有 .csv 檔案...")

    # 1. 尋找所有 .csv 報告
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "*_brain_region_activations.csv"))
    
    if not csv_files:
        print(f"🚨 錯誤：在 {RESULTS_DIR} 中找不到任何 '_brain_region_activations.csv' 檔案。")
        print("請先確認 05_run_batch_analysis.py 腳本是否已成功執行。")
        return

    print(f"找到 {len(csv_files)} 份 .csv 報告。")

    # 2. 讀取所有 .csv 並合併
    all_dataframes = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            subject_id = os.path.basename(f).replace('_brain_region_activations.csv', '')
            df['subject_id'] = subject_id
            all_dataframes.append(df)
        except Exception as e:
            print(f"⚠️ 警告：讀取 {f} 失敗。錯誤: {e}")

    if not all_dataframes:
        print("🚨 錯誤：未能成功讀取任何 .csv 檔案。")
        return

    full_data = pd.concat(all_dataframes, ignore_index=True)
    print(f"已成功合併 {len(all_dataframes)} 筆資料。")

    # 3. 核心分析：按腦區分組，並計算「平均激活值」
    
    # 🚨 核心修正點：
    # 原始錯誤代碼：valid_data = full_data[full_data['Num_Voxels'] > 0]
    # 修正後：我們只篩選「平均激活值」大於 0 的腦區，
    # 這會自動排除 'Background' 和其他 0 體素的區域。
    valid_data = full_data[full_data['Average_Activation'] > 0]
    
    # 按 'Region' 和 'Label_ID' 分組，然後計算 'Average_Activation' 的平均值
    global_ranking = valid_data.groupby(['Region', 'Label_ID'])['Average_Activation'].mean()
    
    global_ranking_df = global_ranking.reset_index()
    global_ranking_df = global_ranking_df.sort_values(by='Average_Activation', ascending=False)
    
    # 儲存完整的全局排名
    global_ranking_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)

    print(f"✅ 成功計算全局腦區激活排名！")
    print(f"--- 最終匯總報告已儲存至: {OUTPUT_SUMMARY_FILE} ---")

    # 4. 打印最終結果
    print("\n--- 【全局】最活躍腦區 (N=38 病患平均前 15 名) ---")
    print(global_ranking_df.head(15).to_string(index=False))
    
    # 5. (可選) 檢查小腦的排名
    cerebellum_regions = global_ranking_df[global_ranking_df['Region'].str.contains('Cerebelum|Vermis', case=False)]
    print("\n--- 小腦區域 (Cerebellum/Vermis) 的全局平均排名 ---")
    print(cerebellum_regions.to_string(index=False))

if __name__ == '__main__':
    summarize_all_results()