#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Script - 診斷預測問題

檢查三個關鍵問題：
1. 我們有沒有用正確的模型？
2. 我們有沒有對應到正確的標籤？
3. 我們有沒有用錯誤的標籤個數（是兩個還是三個）？
"""

import sys
import joblib
import pandas as pd
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.cnn_rf.extract_roi_features import ROIFeatureExtractor


def check_loocv_model():
    """檢查 1: LOOCV 模型"""
    print("=" * 70)
    print("CHECK 1: LOOCV 模型檢查")
    print("=" * 70)
    
    # 檢查一個 LOOCV 模型
    model_path = Path("model/loocv_models_binary_opt/rf_model_sub-0005.joblib")
    
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return
    
    print(f"Loading: {model_path}")
    model = joblib.load(model_path)
    
    print(f"\nModel type: {type(model)}")
    print(f"Model classes: {model.classes_}")
    print(f"Number of classes: {len(model.classes_)}")
    
    # 檢查是否有 class_mapping
    if hasattr(model, 'class_mapping_'):
        print(f"Class mapping: {model.class_mapping_}")
    else:
        print("No class_mapping_ attribute")
    
    # 檢查 pipeline 結構
    if hasattr(model, 'named_steps'):
        print(f"\nPipeline steps: {list(model.named_steps.keys())}")
        if 'model' in model.named_steps:
            rf_model = model.named_steps['model']
            print(f"RF model classes: {rf_model.classes_}")
    
    print("\n結論:")
    if len(model.classes_) == 2:
        print("✓ LOOCV 模型是二分類 (正確)")
        print("  Classes: [0, 1] 應該對應 [NC, AD]")
    else:
        print("✗ LOOCV 模型不是二分類!")
    
    print()


def check_general_model():
    """檢查 2: 通用模型"""
    print("=" * 70)
    print("CHECK 2: 通用模型檢查")
    print("=" * 70)
    
    model_path = Path("model/cnn_rf/rf_model_NC_MCI_AD.joblib")
    
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return
    
    print(f"Loading: {model_path}")
    model = joblib.load(model_path)
    
    print(f"\nModel type: {type(model)}")
    print(f"Model classes: {model.classes_}")
    print(f"Number of classes: {len(model.classes_)}")
    
    if hasattr(model, 'named_steps'):
        print(f"\nPipeline steps: {list(model.named_steps.keys())}")
        if 'model' in model.named_steps:
            rf_model = model.named_steps['model']
            print(f"RF model classes: {rf_model.classes_}")
    
    print("\n結論:")
    if len(model.classes_) == 3:
        print("✓ 通用模型是三分類 (正確)")
        print("  Classes: [0, 1, 2] 應該對應 [AD, MCI, NC] (字母順序)")
    else:
        print("✗ 通用模型不是三分類!")
    
    print()


def check_label_encoding():
    """檢查 3: Label 編碼"""
    print("=" * 70)
    print("CHECK 3: Label 編碼檢查")
    print("=" * 70)
    
    # 檢查訓練數據
    csv_path = Path("data/roi_features.csv")
    
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return
    
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"\nGroup distribution:")
    print(df['Group'].value_counts())
    
    # 測試 category encoding
    print("\n測試 1: astype('category').cat.codes (三分類訓練用)")
    df_test = df.copy()
    df_test['label'] = df_test['Group'].astype('category').cat.codes
    print(df_test[['Group', 'label']].drop_duplicates().sort_values('label'))
    
    # 測試 custom mapping
    print("\n測試 2: custom mapping (LOOCV 二分類用)")
    custom_mapping = {'NC': 0, 'AD': 1}
    df_binary = df[df['Group'].isin(['NC', 'AD'])].copy()
    df_binary['label'] = df_binary['Group'].map(custom_mapping)
    print(df_binary[['Group', 'label']].drop_duplicates().sort_values('label'))
    
    print("\n結論:")
    print("三分類模型 (通用): AD=0, MCI=1, NC=2")
    print("二分類模型 (LOOCV): NC=0, AD=1")
    print()


def test_prediction():
    """檢查 4: 實際預測測試"""
    print("=" * 70)
    print("CHECK 4: 實際預測測試")
    print("=" * 70)
    
    # 測試一個 AD 受試者
    subject_id = "sub-0005"
    ground_truth = "AD"
    
    print(f"Testing: {subject_id} (Ground Truth: {ground_truth})")
    print()
    
    # 初始化特徵提取器
    print("Initializing feature extractor...")
    extractor = ROIFeatureExtractor(
        atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
        atlas_labels_path="data/aal3/AAL3v1.json"
    )
    
    # 提取特徵
    subject_dir = Path(f"data/MRI_processed/{ground_truth}/{subject_id}")
    print(f"Extracting features from: {subject_dir}")
    features = extractor.extract_subject_features(subject_dir)
    feature_df = pd.DataFrame([features])
    print(f"Features extracted: {len(features)} features")
    print()
    
    # 測試 LOOCV 模型
    print("--- Test 1: LOOCV Model ---")
    loocv_model_path = Path(f"model/loocv_models_binary_opt/rf_model_{subject_id}.joblib")
    
    if loocv_model_path.exists():
        loocv_model = joblib.load(loocv_model_path)
        pred_idx = loocv_model.predict(feature_df)[0]
        proba = loocv_model.predict_proba(feature_df)[0]
        
        print(f"Model classes: {loocv_model.classes_}")
        print(f"Predicted index: {pred_idx}")
        print(f"Probabilities: {proba}")
        
        # 正確的對應
        classes_binary = ['NC', 'AD']  # 0=NC, 1=AD
        prediction = classes_binary[pred_idx]
        confidence = proba[pred_idx]
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Correct: {prediction == ground_truth}")
    else:
        print(f"LOOCV model not found: {loocv_model_path}")
    
    print()
    
    # 測試通用模型
    print("--- Test 2: General Model ---")
    general_model_path = Path("model/cnn_rf/rf_model_NC_MCI_AD.joblib")
    
    if general_model_path.exists():
        general_model = joblib.load(general_model_path)
        pred_idx = general_model.predict(feature_df)[0]
        proba = general_model.predict_proba(feature_df)[0]
        
        print(f"Model classes: {general_model.classes_}")
        print(f"Predicted index: {pred_idx}")
        print(f"Probabilities: {proba}")
        
        # 正確的對應
        classes_three = ['AD', 'MCI', 'NC']  # 0=AD, 1=MCI, 2=NC
        prediction = classes_three[pred_idx]
        confidence = proba[pred_idx]
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Correct: {prediction == ground_truth}")
    else:
        print(f"General model not found: {general_model_path}")
    
    print()


def check_cdda_toolkit():
    """檢查 5: CDDA Toolkit 的 label 對應"""
    print("=" * 70)
    print("CHECK 5: CDDA Toolkit Label 對應")
    print("=" * 70)
    
    # 檢查 cdda_tools.py 中的 label 對應
    toolkit_file = Path("app/core/ml_processing/cdda_tools.py")
    
    if toolkit_file.exists():
        print(f"Checking: {toolkit_file}")
        
        with open(toolkit_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 搜尋 classes 定義
        import re
        
        # 搜尋 self.classes 的定義
        classes_matches = re.findall(r"self\.classes\s*=\s*\[([^\]]+)\]", content)
        if classes_matches:
            print("\nFound self.classes definitions:")
            for match in classes_matches:
                print(f"  self.classes = [{match}]")
        
        # 搜尋 class_names 的定義
        class_names_matches = re.findall(r"class_names\s*=\s*\[([^\]]+)\]", content)
        if class_names_matches:
            print("\nFound class_names definitions:")
            for match in class_names_matches:
                print(f"  class_names = [{match}]")
        
        # 搜尋 prediction_result 的賦值
        pred_matches = re.findall(r"prediction_result['\"]?\s*=\s*([^\n]+)", content)
        if pred_matches:
            print("\nFound prediction_result assignments:")
            for match in pred_matches[:3]:
                print(f"  {match}")
    
    print()


def main():
    print("\n")
    print("=" * 70)
    print("PREDICTION DIAGNOSTIC TOOL".center(70))
    print("=" * 70)
    print()
    
    check_loocv_model()
    check_general_model()
    check_label_encoding()
    test_prediction()
    check_cdda_toolkit()
    
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print("1. LOOCV 模型應該是二分類: NC=0, AD=1")
    print("2. 通用模型應該是三分類: AD=0, MCI=1, NC=2")
    print("3. 檢查 CDDA Toolkit 是否使用正確的對應")
    print()


if __name__ == "__main__":
    main()
