"""
尋找 MCI 資料的輔助腳本

此腳本會掃描常見的資料位置，幫助你找到 MCI 資料
"""

import os
from pathlib import Path
import pandas as pd


def check_csv_files():
    """檢查 CSV 檔案中是否有 MCI 標籤"""
    print("=" * 60)
    print("檢查 CSV 檔案中的標籤")
    print("=" * 60)
    
    csv_files = [
        "data/processed/all_aal_roi_features.csv",
        "data/processed/all_116_roi_features.csv",
    ]
    
    for csv_file in csv_files:
        if Path(csv_file).exists():
            print(f"\n檢查: {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                if 'label' in df.columns:
                    labels = df['label'].unique()
                    counts = df['label'].value_counts()
                    
                    print(f"  標籤: {labels}")
                    print(f"  數量:")
                    for label, count in counts.items():
                        print(f"    {label}: {count}")
                    
                    if 'MCI' in labels:
                        print(f"  ✓ 找到 MCI 資料！")
                        mci_subjects = df[df['label'] == 'MCI']['subject_id'].tolist()
                        print(f"  MCI 受試者: {mci_subjects[:5]}..." if len(mci_subjects) > 5 else f"  MCI 受試者: {mci_subjects}")
                    else:
                        print(f"  ✗ 沒有 MCI 標籤")
                else:
                    print(f"  ⚠️  沒有 'label' 欄位")
            except Exception as e:
                print(f"  ❌ 讀取失敗: {e}")
        else:
            print(f"\n✗ 檔案不存在: {csv_file}")


def search_directories():
    """搜尋可能包含 MCI 資料的目錄"""
    print("\n" + "=" * 60)
    print("搜尋可能的 MCI 資料位置")
    print("=" * 60)
    
    # 常見的資料目錄
    search_paths = [
        "data/raw",
        "data/fMRI",
        "data/sMRI",
        "data/processed",
        "../data",  # 上層目錄
        "D:/data",  # Windows 常見位置
        "E:/data",
    ]
    
    found_locations = []
    
    for search_path in search_paths:
        path = Path(search_path)
        if path.exists():
            print(f"\n檢查: {search_path}")
            
            # 搜尋包含 MCI 的目錄
            try:
                for item in path.rglob("*"):
                    if item.is_dir() and "MCI" in item.name.upper():
                        print(f"  ✓ 找到: {item}")
                        found_locations.append(str(item))
            except PermissionError:
                print(f"  ⚠️  權限不足，無法存取")
            except Exception as e:
                print(f"  ⚠️  錯誤: {e}")
    
    if found_locations:
        print(f"\n找到 {len(found_locations)} 個可能的 MCI 資料位置:")
        for loc in found_locations:
            print(f"  - {loc}")
    else:
        print("\n✗ 沒有找到包含 'MCI' 的目錄")


def check_current_structure():
    """檢查當前的 sMRI 資料結構"""
    print("\n" + "=" * 60)
    print("當前 sMRI 資料結構")
    print("=" * 60)
    
    smri_path = Path("data/sMRI")
    
    if not smri_path.exists():
        print("✗ data/sMRI 目錄不存在")
        return
    
    # 列出所有類別
    categories = [d for d in smri_path.iterdir() if d.is_dir()]
    
    print(f"\n找到 {len(categories)} 個類別:")
    for cat in categories:
        subjects = [d for d in cat.iterdir() if d.is_dir()]
        print(f"  {cat.name}: {len(subjects)} 個受試者")
        
        # 顯示樣本
        if subjects:
            sample = subjects[0]
            files = list(sample.glob("*.nii.gz"))
            print(f"    樣本: {sample.name}")
            print(f"    檔案: {[f.name for f in files]}")


def suggest_next_steps():
    """建議下一步操作"""
    print("\n" + "=" * 60)
    print("建議的下一步操作")
    print("=" * 60)
    
    print("""
1. 如果你知道 MCI 資料的位置:
   python scripts/migrate_mci_data.py --source <MCI資料路徑> --dry-run
   
2. 如果 MCI 資料在 CSV 檔案中:
   需要從原始 NIfTI 檔案中提取對應的受試者
   
3. 如果需要從 ADNI 下載 MCI 資料:
   a. 登入 ADNI 網站: https://adni.loni.usc.edu/
   b. 下載 MCI 受試者的 T1 MRI 資料
   c. 使用 migrate_mci_data.py 腳本組織資料
   
4. 如果 MCI 資料在其他位置:
   請提供資料路徑，我可以幫你創建自訂的遷移腳本
    """)


def main():
    print("=" * 60)
    print("MCI 資料尋找輔助工具")
    print("=" * 60)
    
    # 檢查 CSV 檔案
    check_csv_files()
    
    # 搜尋目錄
    search_directories()
    
    # 檢查當前結構
    check_current_structure()
    
    # 建議下一步
    suggest_next_steps()


if __name__ == "__main__":
    main()
