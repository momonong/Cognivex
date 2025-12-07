#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper Script: Convert Comprehensive Stats JSON to CSV for Visualization
用法: python scripts/paper/convert_json_to_csv.py
"""

import json
import csv
import pandas as pd
from pathlib import Path

# [設定] 請確認這裡的路徑指向你剛剛跑完產生的 JSON
INPUT_JSON = Path("output/comprehensive_stats_v2/comprehensive_stats.json")
OUTPUT_CSV = Path("output/comprehensive_stats_v2/final_results.csv")

def main():
    if not INPUT_JSON.exists():
        print(f"[!] 錯誤: 找不到輸入檔案 {INPUT_JSON}")
        return

    print(f"[*] Reading JSON from {INPUT_JSON}...")
    
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取 results 部分 (包含 nc, ad, mci 三個 list)
    results = data.get('results', {})
    
    # 合併所有受試者數據
    all_rows = []
    
    for group, items in results.items():
        print(f"    - Processing group '{group}': {len(items)} subjects")
        for item in items:
            # 確保欄位名稱符合 visualization.py 的需求
            # 必須包含: group, confidence, uq_score, agent_decision, is_fp_corrected
            
            # 處理可能缺失的欄位 (以防萬一)
            row = item.copy()
            row['group'] = item.get('ground_truth', group.upper()) # 確保有 group 欄位
            
            # 確保 is_fp_corrected 存在 (如果舊版沒存到，這裡補算)
            if 'is_fp_corrected' not in row:
                gt = row.get('ground_truth', 'UNKNOWN')
                pred = row.get('prediction', 'UNKNOWN')
                agent_final = row.get('agent_final', 'Unchanged')
                
                # 簡單判定邏輯
                row['is_fp_corrected'] = (gt == 'NC' and pred == 'AD' and agent_final == 'NC')
                
            all_rows.append(row)
            
    # 轉成 DataFrame 並存檔
    df = pd.DataFrame(all_rows)
    
    # 確保輸出目錄存在
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"[*] Success! CSV saved to: {OUTPUT_CSV}")
    print(f"    Total records: {len(df)}")
    print(f"    Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()