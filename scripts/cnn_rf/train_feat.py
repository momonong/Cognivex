import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import joblib # [!!] 匯入 joblib
import os     # [!!] 匯入 os

# --- 1. 設定 ---
ROI_FEATURES_CSV = r"data/roi_features.csv"
MODEL_SAVE_DIR = r"model/cnn"  # [!!] 新增模型儲存路徑
N_FEATURES_TO_SELECT = 30  # 我們要模型找出最重要的 30 個特徵
N_SPLITS = 5               # 5 折交叉驗證
RANDOM_STATE = 42

def train_model(data, target_classes):
    """
    使用指定的類別來訓練和評估隨機森林模型。
    """
    print("\n" + "="*50)
    print(f"--- 正在訓練模型: {target_classes} ---")
    
    # --- 2. 準備資料 ---
    df_filtered = data[data['Group'].isin(target_classes)].copy()
    df_filtered['label'] = df_filtered['Group'].astype('category').cat.codes
    
    X = df_filtered.drop(columns=['Subject_ID', 'Group', 'label'])
    y = df_filtered['label']
    
    feature_names = X.columns
    class_names = df_filtered['Group'].astype('category').cat.categories
    
    print(f"[*] 資料集大小: {len(df_filtered)} 人 ( {len(feature_names)} 個特徵)")
    print(f"[*] 類別分佈:\n{df_filtered['Group'].value_counts().to_string()}")

    # --- 3. 建立一個完整的「管線」(Pipeline) ---
    # 步驟 1: 特徵縮放 (StandardScaler)
    scaler = StandardScaler()
    
    # 步驟 2: 訓練一個初步的隨機森林，用它來「選特徵」
    selector_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=RANDOM_STATE, 
        class_weight='balanced'
    )
    # SelectFromModel 會自動選出最重要的特徵
    selector = SelectFromModel(
        selector_model, 
        threshold='median', # 先用中位數過濾
        max_features=N_FEATURES_TO_SELECT # 最終只保留 30 個
    )
    
    # 步驟 3: 訓練一個「最終的」隨機森林
    final_model = RandomForestClassifier(
        n_estimators=200, 
        random_state=RANDOM_STATE, 
        class_weight='balanced'
    )
    
    # 將所有步驟串聯起來
    pipeline = Pipeline([
        ('scale', scaler),
        ('select', selector),
        ('model', final_model)
    ])

    # --- 4. 執行 5 折交叉驗證 (Stratified K-Fold) ---
    print(f"\n[*] 正在執行 {N_SPLITS} 折交叉驗證 (Stratified K-Fold)...")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # cross_val_score 會自動幫我們訓練和評估 5 次
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    
    print("\n--- 交叉驗證結果 ---")
    print(f"  每次的準確率: {np.round(scores, 3)}")
    print(f"  平均準確率 (Mean Accuracy): {np.mean(scores):.3f}")
    print(f"  標準差 (Std Dev):           {np.std(scores):.3f}")

    # --- 5. [!! 修改 !!] 訓練最終模型並儲存 ---
    print(f"\n[*] 正在於 *所有* {len(df_filtered)} 筆資料上訓練最終模型...")
    pipeline.fit(X, y)
    print("    ...最終模型訓練完成。")

    # 確保儲存目錄存在
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 決定儲存名稱
    if len(target_classes) == 2:
        model_filename = "rf_model_NC_vs_AD.joblib"
    else:
        model_filename = "rf_model_NC_MCI_AD.joblib"
    
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    
    # 儲存管線
    joblib.dump(pipeline, model_save_path)
    print(f"    [v] 成功儲存最終模型至: {model_save_path}")


    # --- 6. 找出最重要的特徵 ---
    print(f"\n[*] 正在找出最重要的 {N_FEATURES_TO_SELECT} 個特徵...")
    
    # [!! 修改 !!] 從已 fit 的 pipeline 中提取 selector
    try:
        final_selector = pipeline.named_steps['select']
        selected_mask = final_selector.get_support()
        selected_features = feature_names[selected_mask]
        
        print("\n--- 最重要的特徵 (Top Features) ---")
        if len(selected_features) > 0:
            for i, f in enumerate(selected_features[:10]): # 只印出前 10 個
                print(f"  {i+1}. {f}")
        else:
            print("[!] 特徵選擇失敗 (可能所有特徵都被保留了)。")
            # 如果 max_features > P，get_support() 會是 all True
            # 我們可以改為顯示 feature_importances_
            final_model_in_pipeline = pipeline.named_steps['model']
            importances = final_model_in_pipeline.feature_importances_
            
            # 我們需要 pre-selection 的 feature names
            pre_selector = pipeline.named_steps['select']
            pre_mask = pre_selector.get_support()
            pre_features = feature_names[pre_mask]

            if len(importances) == len(pre_features):
                indices = np.argsort(importances)[::-1]
                print("--- (備案) 最重要的特徵 (來自模型權重) ---")
                for i in range(10):
                    print(f"  {i+1}. {pre_features[indices[i]]} (權重: {importances[indices[i]]:.4f})")
            else:
                 print("[!] 無法提取特徵權重。")


    except Exception as e:
        print(f"[!] 提取特徵時發生錯誤: {e}")


    return np.mean(scores)

def main():
    try:
        data = pd.read_csv(ROI_FEATURES_CSV)
    except FileNotFoundError:
        print(f"[!] 錯誤: 找不到特徵檔案 {ROI_FEATURES_CSV}")
        print("[!] 請先執行 'python -m src.data.extract_roi_features' 腳本。")
        return
        
    # --- 實驗 1：NC vs AD ---
    train_model(data, target_classes=['NC', 'AD'])
    
    # --- 實驗 2：三分類 (NC vs MCI vs AD) ---
    train_model(data, target_classes=['NC', 'MCI', 'AD'])

if __name__ == "__main__":
    main()