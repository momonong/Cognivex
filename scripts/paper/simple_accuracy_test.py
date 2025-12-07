#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Accuracy Test - 直接用 ML 模型測試準確率
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


def scan_subjects():
    """掃描所有受試者"""
    subjects = {}
    
    for group in ['AD', 'NC', 'MCI']:
        data_folders = glob.glob(f"data/MRI_processed/{group}/sub-*")
        for folder_path in data_folders:
            subject_id = Path(folder_path).name
            nii_files = list(Path(folder_path).glob("*.nii.gz"))
            if len(nii_files) >= 3:
                subjects[subject_id] = group
    
    return subjects


def main():
    print("Simple Accuracy Test")
    print("=" * 60)
    
    # 掃描受試者
    subjects = scan_subjects()
    ad_subjects = [s for s, g in subjects.items() if g == 'AD']
    nc_subjects = [s for s, g in subjects.items() if g == 'NC']
    mci_subjects = [s for s, g in subjects.items() if g == 'MCI']
    
    print(f"AD: {len(ad_subjects)}, NC: {len(nc_subjects)}, MCI: {len(mci_subjects)}")
    print()
    
    # 載入模型
    print("Loading model...")
    model = joblib.load("model/cnn_rf/rf_model_NC_MCI_AD.joblib")
    # 正確的對應：按字母順序編碼 AD=0, MCI=1, NC=2
    classes = ['AD', 'MCI', 'NC']
    print(f"Model loaded: {classes}")
    print()
    
    # 初始化特徵提取器
    print("Initializing feature extractor...")
    extractor = ROIFeatureExtractor(
        atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
        atlas_labels_path="data/aal3/AAL3v1.json"
    )
    print("Ready")
    print()
    
    # 測試 AD/NC
    print("Testing AD/NC subjects...")
    print("-" * 60)
    
    results = []
    confusion = defaultdict(int)
    
    test_subjects = [(s, 'AD') for s in ad_subjects] + [(s, 'NC') for s in nc_subjects]
    
    for i, (subject_id, ground_truth) in enumerate(test_subjects, 1):
        try:
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
                'confidence': confidence
            })
            
            confusion[f"{ground_truth}_as_{prediction}"] += 1
            
            status = "OK" if correct else "X"
            print(f"[{i}/{len(test_subjects)}] {subject_id}: {ground_truth} -> {prediction} [{status}] ({confidence:.2f})")
            
        except Exception as e:
            print(f"[{i}/{len(test_subjects)}] {subject_id}: ERROR - {str(e)[:40]}")
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # 計算準確率
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"Total: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # 混淆矩陣
    print("Confusion Matrix:")
    print(f"  AD -> AD: {confusion.get('AD_as_AD', 0)}")
    print(f"  AD -> NC: {confusion.get('AD_as_NC', 0)}")
    print(f"  AD -> MCI: {confusion.get('AD_as_MCI', 0)}")
    print(f"  NC -> AD: {confusion.get('NC_as_AD', 0)}")
    print(f"  NC -> NC: {confusion.get('NC_as_NC', 0)}")
    print(f"  NC -> MCI: {confusion.get('NC_as_MCI', 0)}")
    print()
    
    # 按類別準確率
    ad_correct = confusion.get('AD_as_AD', 0)
    ad_total = sum(v for k, v in confusion.items() if k.startswith('AD_'))
    nc_correct = confusion.get('NC_as_NC', 0)
    nc_total = sum(v for k, v in confusion.items() if k.startswith('NC_'))
    
    if ad_total > 0:
        print(f"AD Accuracy: {ad_correct/ad_total:.4f} ({ad_correct/ad_total*100:.2f}%) - {ad_correct}/{ad_total}")
    if nc_total > 0:
        print(f"NC Accuracy: {nc_correct/nc_total:.4f} ({nc_correct/nc_total*100:.2f}%) - {nc_correct}/{nc_total}")
    
    print()
    print("=" * 60)
    
    # 保存結果
    import json
    output_file = Path("output/simple_accuracy_test.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'total': total_count,
            'correct': correct_count,
            'accuracy': accuracy,
            'confusion_matrix': dict(confusion),
            'results': results
        }, f, indent=2)
    
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
