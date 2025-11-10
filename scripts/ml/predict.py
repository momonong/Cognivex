"""
使用訓練好的模型進行預測

用法:
    python scripts/ml/predict.py --input path/to/image_T1.nii.gz
"""

import numpy as np
import argparse
import joblib
import os
from nilearn import datasets, image as nimg
from nilearn.maskers import NiftiLabelsMasker

# 配置
MODEL_DIR = "model/ml/"
IMPORTANT_ROIS = {
    'Hippocampus_L': 37, 'Hippocampus_R': 38,
    'ParaHippocampal_L': 39, 'ParaHippocampal_R': 40,
    'Amygdala_L': 41, 'Amygdala_R': 42,
    'Temporal_Sup_L': 79, 'Temporal_Sup_R': 80,
    'Temporal_Mid_L': 85, 'Temporal_Mid_R': 86,
    'Temporal_Inf_L': 89, 'Temporal_Inf_R': 90,
    'Parietal_Sup_L': 59, 'Parietal_Sup_R': 60,
    'Parietal_Inf_L': 61, 'Parietal_Inf_R': 62,
    'Cingulum_Ant_L': 31, 'Cingulum_Ant_R': 32,
    'Cingulum_Post_L': 35, 'Cingulum_Post_R': 36,
    'Frontal_Sup_L': 1, 'Frontal_Sup_R': 2,
    'Frontal_Mid_L': 7, 'Frontal_Mid_R': 8,
}

def load_model():
    """載入訓練好的模型"""
    model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"找不到 scaler 檔案: {scaler_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def extract_features(image_path):
    """從影像提取 ROI 特徵"""
    # 載入 AAL atlas
    aal_atlas = datasets.fetch_atlas_aal(version='SPM12')
    aal_img = nimg.load_img(aal_atlas.maps)
    masker = NiftiLabelsMasker(labels_img=aal_img, standardize=False, strategy='mean')
    
    # 載入影像
    img = nimg.load_img(image_path)
    masker.fit(img)
    
    # 提取特徵
    roi_features = masker.transform(img).flatten()
    important_roi_indices = [i-1 for i in IMPORTANT_ROIS.values()]
    roi_features = roi_features[important_roi_indices]
    
    return roi_features.reshape(1, -1)

def predict(image_path):
    """預測影像的類別"""
    print("="*60)
    print("AD 分類預測")
    print("="*60)
    
    # 載入模型
    print("\n載入模型...")
    model, scaler = load_model()
    print("✅ 模型載入成功")
    
    # 提取特徵
    print(f"\n提取特徵: {os.path.basename(image_path)}")
    features = extract_features(image_path)
    print(f"✅ 特徵提取完成 (維度: {features.shape})")
    
    # 標準化
    features_scaled = scaler.transform(features)
    
    # 預測
    print("\n進行預測...")
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # 顯示結果
    print("\n" + "="*60)
    print("預測結果")
    print("="*60)
    
    label = "NC (正常)" if prediction == 0 else "AD (阿茲海默症)"
    print(f"\n預測類別: {label}")
    print(f"\n信心度:")
    print(f"  NC (正常):      {probability[0]:.2%}")
    print(f"  AD (阿茲海默症): {probability[1]:.2%}")
    
    return prediction, probability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用訓練好的模型預測 AD")
    parser.add_argument("--input", type=str, required=True, help="輸入影像路徑 (T1.nii.gz)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 錯誤：找不到檔案 {args.input}")
        exit(1)
    
    predict(args.input)
