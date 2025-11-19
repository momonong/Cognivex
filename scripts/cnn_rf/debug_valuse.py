import os
import sys
import pandas as pd
import numpy as np
import joblib

# --- 路徑設定 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'roi_features.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'cnn', 'rf_model_NC_vs_AD.joblib')
# 假設你的 Scaler 存在這裡 (如果沒有這個檔，那就是問題所在！)
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'cnn', 'scaler.joblib') 

def debug_inference_values():
    print("========================================================")
    print("   DEBUG: RAW vs SCALED VALUES INSPECTION")
    print("========================================================")

    # 1. 檢查 Scaler 是否存在
    if not os.path.exists(SCALER_PATH):
        print(f"❌ CRITICAL ERROR: Scaler not found at {SCALER_PATH}")
        print("   The model was trained with scaled data, but inference is missing the scaler.")
        print("   This explains why features with large raw values (Motor Area) dominate.")
        return

    # 2. 載入資產
    print(f"[Loading] Model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"[Loading] Scaler: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    
    # 3. 載入數據
    df = pd.read_csv(DATA_PATH)
    if 'subject_id' in df.columns: df.set_index('subject_id', inplace=True)
    if 'diagnosis' in df.columns: df = df.drop(columns=['diagnosis'])
    
    # 確保欄位順序一致
    if hasattr(model, 'feature_names_in_'):
        df = df[model.feature_names_in_]
    
    # 4. 挑選兩個受試者
    sub_ad = 'sub-0005' # 假設是 AD
    sub_nc = 'sub-0010' # 假設是 NC
    
    if sub_ad not in df.index or sub_nc not in df.index:
        print("⚠️ Sample subjects not found, using first two rows.")
        sub_ad = df.index[0]
        sub_nc = df.index[1]

    print(f"\nComparing {sub_ad} vs {sub_nc}...")

    # 5. 提取關鍵特徵數值
    # 我們看 "Supp_Motor_Area_L" (目前的 Top 1) 和 "Hippocampus_L" (應該要是 Top 1)
    targets = [
        col for col in df.columns 
        if 'Supp_Motor_Area_L_GM' in col or 'Hippocampus_L_GM' in col
    ]

    print("\n{:<30} | {:<20} | {:<20} | {:<20}".format("Feature", "Subject", "Raw Value", "Scaled Value"))
    print("-" * 100)

    for feat in targets:
        # 取得原始值
        raw_ad = df.loc[sub_ad, feat]
        raw_nc = df.loc[sub_nc, feat]
        
        # 模擬推理時的轉換 (transform)
        # 注意：這裡我們要確認是否正確使用了 scaler
        vec_ad = df.loc[[sub_ad]].values
        vec_nc = df.loc[[sub_nc]].values
        
        scaled_vec_ad = scaler.transform(vec_ad)
        scaled_vec_nc = scaler.transform(vec_nc)
        
        # 找出該特徵在向量中的 index
        feat_idx = list(df.columns).index(feat)
        
        val_scaled_ad = scaled_vec_ad[0][feat_idx]
        val_scaled_nc = scaled_vec_nc[0][feat_idx]
        
        print(f"{feat:<30} | {sub_ad:<20} | {raw_ad:10.4f}           | {val_scaled_ad:10.4f}")
        print(f"{'':<30} | {sub_nc:<20} | {raw_nc:10.4f}           | {val_scaled_nc:10.4f}")
        print("-" * 100)

    # 6. 檢查特徵統計
    print("\n[Global Stats Check]")
    feat_motor = 'Supp_Motor_Area_L_GM'
    feat_hippo = 'Hippocampus_L_GM'
    
    if feat_motor in df.columns:
        print(f"{feat_motor}: Mean={df[feat_motor].mean():.2f}, Max={df[feat_motor].max():.2f}")
    if feat_hippo in df.columns:
        print(f"{feat_hippo}:     Mean={df[feat_hippo].mean():.2f}, Max={df[feat_hippo].max():.2f}")

if __name__ == "__main__":
    debug_inference_values()