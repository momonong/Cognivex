#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test LOOCV Model Accuracy - 測試 LOOCV 專屬模型的準確率
"""

import sys
import glob
import joblib
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.cnn_rf.extract_roi_features import ROIFeatureExtractor

# LOOCV 模型目錄
LOOCV_MODEL_DIR = Path("model/loocv_models_binary_opt")


def scan_subjects():
    """掃描 AD 和 NC 受試者"""
    subjects = {}
    
    for group in ['AD', 'NC']:
        data_folders = glob.glob(f"data/MRI_processed/{group}/sub-*")
        for folder_path in data_folders:
            subject_id = Path(folder_path).name
            nii_files = list(Path(folder_path).glob("*.nii.gz"))
            if len(nii_files) >= 3:
                subjects[subject_id] = group
    
    return subjects


def main():
    print("=" * 70)
    print("LOOCV MODEL ACCURACY TEST")
    print("=" * 70)
    print()
    
    # 掃描受試者
    subjects = scan_subjects()
    ad_subjects = [s for s, g in subjects.items() if g == 'AD']
    nc_subjects = [s for s, g in subjects.items() if g == 'NC']
    
    print(f"Found: AD={len(ad_subjects)}, NC={len(nc_subjects)}, Total={len(subjects)}")
    print()
    
    # 初始化特徵提取器
    print("Initializing feature extractor...")
    extractor = ROIFeatureExtractor(
        atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
        atlas_labels_path="data/aal3/AAL3v1.json"
    )
    print("Ready")
    print()
    
    # 測試每個受試者
    print("=" * 70)
    print("TESTING WITH LOOCV MODELS")
    print("=" * 70)
    print()
    
    results = []
    confusion = defaultdict(int)
    
    # LOOCV 二分類模型的 label 對應：NC=0, AD=1
    classes = ['NC', 'AD']
    
    for i, (subject_id, ground_truth) in enumerate(subjects.items(), 1):
        # 尋找專屬模型
        model_path = LOOCV_MODEL_DIR / f"rf_model_{subject_id}.joblib"
        
        if not model_path.exists():
            print(f"[{i}/{len(subjects)}] {subject_id}: NO MODEL FOUND")
            continue
        
        try:
            # 載入專屬模型
            model = joblib.load(model_path)
            
            # 找到受試者目錄
            subject_dir = Path(f"data/MRI_processed/{ground_truth}/{subject_id}")
            
            # 提取特徵
            features = extractor.extract_subject_features(subject_dir)
            feature_df = pd.DataFrame([features])
            
            # 預測
            pred_idx = model.predict(feature_df)[0]
            prediction = classes[pred_idx]
            proba = model.predict_proba(feature_df)[0]
            confidence = proba[pred_idx]
            
            correct = (prediction == ground_truth)
            results.append({
                'subject_id': subject_id,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': correct,
                'confidence': confidence,
                'model': model_path.name
            })
            
            confusion[f"{ground_truth}_as_{prediction}"] += 1
            
            status = "OK" if correct else "X "
            print(f"[{i}/{len(subjects)}] {subject_id}: {ground_truth} -> {prediction} [{status}] ({confidence:.2f})")
            
        except Exception as e:
            print(f"[{i}/{len(subjects)}] {subject_id}: ERROR - {str(e)[:50]}")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    
    # 計算準確率
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"Total Tested: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {total_count - correct_count}")
    print(f"ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # 混淆矩陣
    print("Confusion Matrix:")
    print(f"  AD -> AD: {confusion.get('AD_as_AD', 0)}")
    print(f"  AD -> NC: {confusion.get('AD_as_NC', 0)}")
    print(f"  NC -> AD: {confusion.get('NC_as_AD', 0)}")
    print(f"  NC -> NC: {confusion.get('NC_as_NC', 0)}")
    print()
    
    # 按類別準確率
    ad_correct = confusion.get('AD_as_AD', 0)
    ad_total = sum(v for k, v in confusion.items() if k.startswith('AD_'))
    nc_correct = confusion.get('NC_as_NC', 0)
    nc_total = sum(v for k, v in confusion.items() if k.startswith('NC_'))
    
    if ad_total > 0:
        ad_acc = ad_correct / ad_total
        print(f"AD Accuracy: {ad_acc:.4f} ({ad_acc*100:.2f}%) - {ad_correct}/{ad_total}")
    
    if nc_total > 0:
        nc_acc = nc_correct / nc_total
        print(f"NC Accuracy: {nc_acc:.4f} ({nc_acc*100:.2f}%) - {nc_correct}/{nc_total}")
    
    print()
    
    # 計算 Precision, Recall, F1
    if ad_total > 0 and nc_total > 0:
        # Binary metrics (treating AD as positive class)
        tp = confusion.get('AD_as_AD', 0)
        tn = confusion.get('NC_as_NC', 0)
        fp = confusion.get('NC_as_AD', 0)
        fn = confusion.get('AD_as_NC', 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("Binary Classification Metrics (AD as positive):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall (Sensitivity): {recall:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print()
    
    print("=" * 70)
    
    # 保存結果
    import json
    output_file = Path("output/loocv_accuracy_test.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'total': total_count,
            'correct': correct_count,
            'accuracy': accuracy,
            'confusion_matrix': dict(confusion),
            'results': results
        }, f, indent=2)
    
    print(f"Results saved: {output_file}")


if __name__ == "__main__":
    main()
