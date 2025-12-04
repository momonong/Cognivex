#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Paper Test Script

快速測試腳本，用於驗證系統是否正常運作。
分析 1-2 個受試者，快速生成結果。

使用方法:
    python scripts/quick_paper_test.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess


def main():
    print("=" * 80)
    print("CDDA Quick Paper Test")
    print("=" * 80)
    print()
    print("This script will analyze 2 subjects to verify the system is working.")
    print("Results will be saved to: output/quick_test/")
    print()
    
    # 選擇測試受試者
    test_subjects = ["sub-0005", "sub-0015"]
    
    print(f"Test subjects: {', '.join(test_subjects)}")
    print()
    
    # 構建命令
    cmd = [
        sys.executable,
        "scripts/paper_analysis.py",
        "--subjects"
    ] + test_subjects + [
        "--output", "output/quick_test",
        "--use-4bit"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("=" * 80)
    print()
    
    # 執行分析
    try:
        subprocess.run(cmd, check=True)
        
        print()
        print("=" * 80)
        print("Quick test completed successfully!")
        print("=" * 80)
        print()
        print("Check the results in: output/quick_test/")
        print()
        print("If everything looks good, you can run the full analysis:")
        print("  python scripts/paper_analysis.py --subjects sub-0001 sub-0002 ...")
        print()
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print("Quick test failed!")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        print("Please check:")
        print("  1. Models are downloaded and paths are correct")
        print("  2. Data files exist in data/MRI_processed/")
        print("  3. GPU has sufficient memory")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
