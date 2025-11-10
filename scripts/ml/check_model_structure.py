"""
檢查模型檔案結構
"""
import joblib
from pathlib import Path

model_path = Path('model/ml/rf_model.pkl')

if model_path.exists():
    print(f"載入模型: {model_path}")
    model_data = joblib.load(model_path)
    
    print(f"\n模型類型: {type(model_data)}")
    
    if isinstance(model_data, dict):
        print("\n模型是字典，包含以下鍵:")
        for key in model_data.keys():
            print(f"  - {key}: {type(model_data[key])}")
    else:
        print("\n模型不是字典，直接是模型物件")
        print(f"模型類別: {model_data.__class__.__name__}")
        
        # 檢查是否有 feature_names_in_ 屬性
        if hasattr(model_data, 'feature_names_in_'):
            print(f"\n特徵名稱數量: {len(model_data.feature_names_in_)}")
            print(f"前 5 個特徵: {list(model_data.feature_names_in_[:5])}")
        
        # 檢查是否有 n_features_in_ 屬性
        if hasattr(model_data, 'n_features_in_'):
            print(f"特徵數量: {model_data.n_features_in_}")
else:
    print(f"找不到模型檔案: {model_path}")

# 也檢查訓練數據
train_path = Path('data/processed/train_features.csv')
if train_path.exists():
    import pandas as pd
    train_df = pd.read_csv(train_path)
    print(f"\n訓練數據形狀: {train_df.shape}")
    print(f"欄位: {list(train_df.columns[:10])}...")
else:
    print(f"\n找不到訓練數據: {train_path}")
