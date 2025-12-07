#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Accuracy Test - 快速驗證 AD/NC 預測準確率
"""

import sys
import glob
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


def scan_ad_nc_subjects():
    """掃描 AD 和 NC 受試者"""
    subjects = {'AD': [], 'NC': []}
    
    for group in ['AD', 'NC']:
        data_folders = glob.glob(f"data/MRI_processed/{group}/sub-*")
        for folder_path in data_folders:
            subject_id = Path(folder_path).name
            nii_files = list(Path(folder_path).glob("*.nii.gz"))
            if len(nii_files) >= 3:
                subjects[group].append(subject_id)
    
    return subjects


def main():
    print("=" * 80)
    print("QUICK ACCURACY TEST - AD/NC Subjects")
    print("=" * 80)
    print()
    
    # 掃描受試者
    subjects = scan_ad_nc_subjects()
    print(f"Found subjects:")
    print(f"  AD: {len(subjects['AD'])}")
    print(f"  NC: {len(subjects['NC'])}")
    print(f"  Total: {len(subjects['AD']) + len(subjects['NC'])}")
    print()
    
    # 初始化 Agent
    print("Initializing CDDA Agent...")
    try:
        agent = CDDAAgent(use_llm=False, verbose=False)
        print("[OK] Agent initialized")
        print()
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        sys.exit(1)
    
    # 測試預測
    results = []
    confusion = defaultdict(int)
    
    print("=" * 80)
    print("TESTING PREDICTIONS")
    print("=" * 80)
    print()
    
    all_subjects = [(s, 'AD') for s in subjects['AD']] + [(s, 'NC') for s in subjects['NC']]
    
    for i, (subject_id, ground_truth) in enumerate(all_subjects, 1):
        print(f"[{i}/{len(all_subjects)}] {subject_id} (GT: {ground_truth})...", end=' ', flush=True)
        
        try:
            result = agent.run_analysis(subject_id)
            prediction = result.prediction
            correct = (prediction == ground_truth)
            
            results.append({
                'subject_id': subject_id,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': correct,
                'confidence': result.confidence
            })
            
            confusion[f"{ground_truth}_as_{prediction}"] += 1
            
            status = "[OK]" if correct else "[X]"
            print(f"{status} Pred: {prediction} (Conf: {result.confidence:.2f})")
            
        except Exception as e:
            print(f"[ERROR] Error: {str(e)[:50]}")
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    # 計算準確率
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"Total Subjects: {total_count}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Incorrect Predictions: {total_count - correct_count}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # 混淆矩陣
    print("Confusion Matrix:")
    print(f"  AD → AD: {confusion['AD_as_AD']}")
    print(f"  AD → NC: {confusion['AD_as_NC']}")
    print(f"  NC → AD: {confusion['NC_as_AD']}")
    print(f"  NC → NC: {confusion['NC_as_NC']}")
    print()
    
    # 按類別準確率
    ad_correct = confusion['AD_as_AD']
    ad_total = confusion['AD_as_AD'] + confusion['AD_as_NC']
    nc_correct = confusion['NC_as_NC']
    nc_total = confusion['NC_as_NC'] + confusion['NC_as_AD']
    
    if ad_total > 0:
        ad_accuracy = ad_correct / ad_total
        print(f"AD Accuracy: {ad_accuracy:.4f} ({ad_accuracy*100:.2f}%) - {ad_correct}/{ad_total}")
    
    if nc_total > 0:
        nc_accuracy = nc_correct / nc_total
        print(f"NC Accuracy: {nc_accuracy:.4f} ({nc_accuracy*100:.2f}%) - {nc_correct}/{nc_total}")
    
    print()
    print("=" * 80)
    
    # 保存結果
    import json
    output_file = Path("output/quick_accuracy_test.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'total': total_count,
            'correct': correct_count,
            'accuracy': accuracy,
            'confusion_matrix': dict(confusion),
            'results': results
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
