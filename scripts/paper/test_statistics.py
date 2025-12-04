#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Comprehensive Statistics Script

快速測試統計腳本，只分析少量受試者。

使用方法:
    python scripts/test_statistics.py
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 100)
    print("CDDA Comprehensive Statistics - Quick Test")
    print("=" * 100)
    print()
    print("This script will analyze 3 subjects to test the statistics system.")
    print("Results will be saved to: output/test_statistics/")
    print()
    
    # 構建命令
    cmd = [
        sys.executable,
        "scripts/comprehensive_statistics.py",
        "--output", "output/test_statistics",
        "--limit", "3",
        "--use-4bit"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("=" * 100)
    print()
    
    # 執行統計分析
    try:
        subprocess.run(cmd, check=True)
        
        print()
        print("=" * 100)
        print("Quick test completed successfully!")
        print("=" * 100)
        print()
        print("Check the results in: output/test_statistics/")
        print()
        print("Files generated:")
        print("  - comprehensive_statistics_report.txt")
        print("  - comprehensive_statistics.json")
        print("  - comprehensive_statistics.csv")
        print()
        print("If everything looks good, you can run the full analysis:")
        print("  python scripts/comprehensive_statistics.py")
        print()
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 100)
        print("Quick test failed!")
        print("=" * 100)
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
