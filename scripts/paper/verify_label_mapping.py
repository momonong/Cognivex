#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Label Mapping - 完整驗證標籤對應

追蹤整個預測流程，確認每一步的標籤對應都正確
"""

import sys
import joblib
import pandas as pd
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.cnn_rf.extract_roi_features import ROIFeatureExtractor
from app.core.cnn_rf.end_to_end_inference import EndToEndPredictor
from app.core.ml_processing.cdda_tools import CDDAToolKit


def test_direct_model():
    """Test 1: 直接使用模型預測"""
    print("=" * 70)
    print("TEST 1: 直接模型預測")
    print("=" * 70)
    
    # 測試一個 AD 受試者
    subject_id = "sub-0005"
    ground_truth = "AD"
    
    print(f"Subject: {subject_id}")
    print(f"Ground Truth: {ground_truth}")
    print()
    
    # 載入 LOOCV 模型
    model_path = Path(f"model/loocv_models_binary_opt/rf_model_{subject_id}.joblib")
    model = joblib.load(model_path)
    
    print(f"Model classes: {model.classes_}")
    
    # 檢查 class_mapping
    if hasattr(model, 'class_mapping_'):
        print(f"Class mapping: {model.class_mapping_}")
        print("  -> 這是訓練時存的對應")
    
    # 提取特徵
    extractor = ROIFeatureExtractor(
        atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
        atlas_labels_path="data/aal3/AAL3v1.json"
    )
    
    subject_dir = Path(f"data/MRI_processed/{ground_truth}/{subject_id}")
    features = extractor.extract_subject_features(subject_dir)
    feature_df = pd.DataFrame([features])
    
    # 預測
    pred_idx = model.predict(feature_df)[0]
    proba = model.predict_proba(feature_df)[0]
    
    print(f"\nPrediction index: {pred_idx}")
    print(f"Probabilities: {proba}")
    
    # 使用 class_mapping 轉換
    if hasattr(model, 'class_mapping_'):
        predicted_label = model.class_mapping_[pred_idx]
        print(f"\nUsing class_mapping_:")
        print(f"  {pred_idx} -> {predicted_label}")
    else:
        print(f"\nNo class_mapping_, using manual mapping:")
        print(f"  Assuming: 0=NC, 1=AD")
        class_names = ['NC', 'AD']
        predicted_label = class_names[pred_idx]
        print(f"  {pred_idx} -> {predicted_label}")
    
    print(f"\nFinal Prediction: {predicted_label}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Correct: {predicted_label == ground_truth}")
    print()


def test_end_to_end_predictor():
    """Test 2: EndToEndPredictor"""
    print("=" * 70)
    print("TEST 2: EndToEndPredictor")
    print("=" * 70)
    
    subject_id = "sub-0005"
    ground_truth = "AD"
    
    print(f"Subject: {subject_id}")
    print(f"Ground Truth: {ground_truth}")
    print()
    
    # 使用 LOOCV 模型
    model_path = f"model/loocv_models_binary_opt/rf_model_{subject_id}.joblib"
    
    predictor = EndToEndPredictor(
        model_path=model_path,
        data_root="data/MRI_processed"
    )
    
    print(f"Predictor classes: {predictor.classes}")
    print()
    
    # 預測
    result = predictor.predict_subject(subject_id, verbose=False)
    
    print(f"Predicted label: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Correct: {result['predicted_label'] == ground_truth}")
    print()


def test_cdda_toolkit():
    """Test 3: CDDAToolKit"""
    print("=" * 70)
    print("TEST 3: CDDAToolKit")
    print("=" * 70)
    
    subject_id = "sub-0005"
    ground_truth = "AD"
    
    print(f"Subject: {subject_id}")
    print(f"Ground Truth: {ground_truth}")
    print()
    
    # 使用 LOOCV 模型
    model_path = f"model/loocv_models_binary_opt/rf_model_{subject_id}.joblib"
    
    toolkit = CDDAToolKit(
        model_path=model_path,
        data_root="data/MRI_processed"
    )
    
    print(f"Toolkit classes: {toolkit.classes}")
    print()
    
    # 獲取診斷報告
    report = toolkit.get_diagnostic_report(subject_id, verbose=False)
    
    print(f"Prediction result: {report['prediction_result']}")
    print(f"Confidence: {report['confidence']:.4f}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Correct: {report['prediction_result'] == ground_truth}")
    print()


def test_all_three():
    """Test 4: 測試三個 AD 受試者"""
    print("=" * 70)
    print("TEST 4: 測試三個 AD 受試者")
    print("=" * 70)
    
    test_subjects = ["sub-0005", "sub-0011", "sub-0020"]
    
    for subject_id in test_subjects:
        model_path = f"model/loocv_models_binary_opt/rf_model_{subject_id}.joblib"
        
        toolkit = CDDAToolKit(
            model_path=model_path,
            data_root="data/MRI_processed"
        )
        
        report = toolkit.get_diagnostic_report(subject_id, verbose=False)
        
        pred = report['prediction_result']
        conf = report['confidence']
        correct = (pred == "AD")
        
        status = "OK" if correct else "X"
        print(f"{subject_id}: AD -> {pred} [{status}] (Conf: {conf:.2f})")
    
    print()


def main():
    print("\n")
    print("=" * 70)
    print("LABEL MAPPING VERIFICATION".center(70))
    print("=" * 70)
    print()
    
    print("Testing label mapping through entire pipeline...")
    print()
    
    try:
        test_direct_model()
        test_end_to_end_predictor()
        test_cdda_toolkit()
        test_all_three()
        
        print("=" * 70)
        print("VERIFICATION COMPLETE")
        print("=" * 70)
        print()
        print("If all tests show 'Correct: True', label mapping is fixed!")
        print()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
