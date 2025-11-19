import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
import matplotlib.pyplot as plt

# --- 1. 設定 ---
ROI_FEATURES_CSV = r"data/roi_features.csv"
MODEL_SAVE_DIR = r"model/cnn"

# 定義我們要評估的模型和對應的類別
MODELS_TO_EVALUATE = [
    {
        "name": "NC vs AD (二分類)",
        "model_path": os.path.join(MODEL_SAVE_DIR, "rf_model_NC_vs_AD.joblib"),
        "classes": ['NC', 'AD']
    },
    {
        "name": "NC vs MCI vs AD (三分類)",
        "model_path": os.path.join(MODEL_SAVE_DIR, "rf_model_NC_MCI_AD.joblib"),
        "classes": ['NC', 'MCI', 'AD']
    }
]

def evaluate_models():
    # --- 2. 載入完整的特徵資料 ---
    try:
        data = pd.read_csv(ROI_FEATURES_CSV)
    except FileNotFoundError:
        print(f"[!] 錯誤: 找不到特徵檔案 {ROI_FEATURES_CSV}")
        print("[!] 請先執行 'python -m src.data.extract_roi_features' 腳本。")
        return

    print(f"[*] 成功載入 {len(data)} 筆總特徵資料。")
    
    # --- 3. 依序評估每個模型 ---
    for config in MODELS_TO_EVALUATE:
        model_name = config["name"]
        model_path = config["model_path"]
        target_classes_config = config["classes"] # 這是我們*希望*評估的類別
        
        print("\n" + "="*50)
        print(f"--- 正在評估模型: {model_name} ---")
        
        # 3.1 載入模型管線 (Pipeline)
        try:
            pipeline = joblib.load(model_path)
        except FileNotFoundError:
            print(f"[!] 錯誤: 找不到模型檔案 {model_path}")
            print("[!] 請先執行 'python -m src.train_features' (或 src.cnn.train_feat) 來訓練模型。")
            continue
            
        # 3.2 準備特定類別的資料
        df_filtered = data[data['Group'].isin(target_classes_config)].copy()
        
        X = df_filtered.drop(columns=['Subject_ID', 'Group'])
        
        # [!!] --- Bug 修正 --- [!!]
        
        # 1. 獲取真實標籤 (文字)
        y_true_text = df_filtered['Group'] 
        
        # 2. 獲取真實標籤 (數字)，必須使用與訓練時 *完全相同* 的方式
        #    Pandas 會自動按字母排序 (e.g., 'AD'=0, 'NC'=1)
        #    我們使用 .astype('category') 並指定 categories 來確保順序
        cat_type = pd.CategoricalDtype(categories=sorted(target_classes_config), ordered=True)
        y_true_numeric = y_true_text.astype(cat_type).cat.codes
        
        # 3. 獲取按字母排序的標籤名稱 (e.g., ['AD', 'NC'])
        #    這才是模型 *真正* 的標籤順序
        model_class_names = sorted(target_classes_config)
        
        # [!!] --- Bug 修正結束 --- [!!]

        if X.empty:
            print(f"[!] 警告: 找不到類別為 {target_classes_config} 的資料。")
            continue

        # 3.3 執行預測
        # pipeline.predict() 輸出的 *就是* 數字標籤 (e.g., 0, 1)
        y_pred_numeric = pipeline.predict(X)
        
        # --- 4. 顯示評估報告 ---
        print("\n--- 總體評估報告 (針對所有資料) ---")
        # 我們現在比較 數字 vs 數字，並使用模型真正的標籤名稱
        print(classification_report(y_true_numeric, y_pred_numeric, target_names=model_class_names))
        
        print("\n--- 混淆矩陣 (Confusion Matrix) ---")
        # 同樣，比較 數字 vs 數字
        cm = confusion_matrix(y_true_numeric, y_pred_numeric)
        
        # 並使用模型真正的標籤名稱
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_class_names)
        
        # 為了在終端機中顯示，我們手動格式化
        header = " (Pred) " + " ".join([f"{c:<10}" for c in model_class_names])
        print(header)
        for i, true_label in enumerate(model_class_names):
            row_str = f"(True) {true_label:<7}"
            for val in cm[i]:
                row_str += f"{val:<10} "
            print(row_str)

    print("\n" + "="*50)
    print("[*] 評估完成。")

if __name__ == "__main__":
    # 確保你安裝了 scikit-learn 和 joblib
    # pip install scikit-learn joblib
    evaluate_models()