"""
MCI 資料遷移腳本

此腳本用於將 MCI (Mild Cognitive Impairment) 資料遷移到 data/sMRI/MCI/ 目錄，
並按照現有的資料結構組織。

使用方法:
    python scripts/migrate_mci_data.py --source <來源目錄> [--dry-run]

參數:
    --source: MCI 資料的來源目錄
    --dry-run: 只顯示將要執行的操作，不實際複製檔案
    --pattern: 檔案名稱模式 (預設: *T1*.nii.gz)
"""

import os
import shutil
import argparse
from pathlib import Path
import re


def find_t1_files(source_dir, pattern="*T1*.nii.gz"):
    """
    在來源目錄中尋找 T1 MRI 檔案
    
    Args:
        source_dir: 來源目錄路徑
        pattern: 檔案名稱模式
    
    Returns:
        找到的檔案路徑列表
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"來源目錄不存在: {source_dir}")
    
    # 遞迴搜尋所有符合模式的檔案
    t1_files = list(source_path.rglob(pattern))
    
    print(f"\n在 {source_dir} 中找到 {len(t1_files)} 個 T1 檔案")
    
    return t1_files


def extract_subject_id(file_path):
    """
    從檔案路徑中提取受試者 ID
    
    支援的格式:
    - sub-0001
    - sub_0001
    - 0001
    - ADNI_001_S_0001
    
    Args:
        file_path: 檔案路徑
    
    Returns:
        標準化的受試者 ID (格式: sub-XXXX)
    """
    file_name = Path(file_path).name
    
    # 嘗試不同的模式
    patterns = [
        r'sub[-_](\d{4})',  # sub-0001 或 sub_0001
        r'(\d{4})',          # 0001
        r'_S_(\d{4})',       # ADNI_001_S_0001
    ]
    
    for pattern in patterns:
        match = re.search(pattern, file_name)
        if match:
            subject_num = match.group(1)
            return f"sub-{subject_num}"
    
    # 如果都不匹配，使用檔案名稱的前綴
    base_name = Path(file_path).stem.split('.')[0]
    print(f"  ⚠️  無法從 {file_name} 提取標準 ID，使用: {base_name}")
    return base_name


def organize_mci_data(source_dir, target_dir="data/sMRI/MCI", pattern="*T1*.nii.gz", dry_run=False):
    """
    組織 MCI 資料到目標目錄
    
    目標結構:
    data/sMRI/MCI/
    ├── sub-0001/
    │   └── sub_0001_T1.nii.gz
    ├── sub-0002/
    │   └── sub_0002_T1.nii.gz
    └── ...
    
    Args:
        source_dir: 來源目錄
        target_dir: 目標目錄
        pattern: 檔案名稱模式
        dry_run: 是否只模擬執行
    """
    # 尋找所有 T1 檔案
    t1_files = find_t1_files(source_dir, pattern)
    
    if not t1_files:
        print("❌ 沒有找到任何 T1 檔案")
        return
    
    # 創建目標目錄
    target_path = Path(target_dir)
    
    if not dry_run:
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ 創建目標目錄: {target_dir}")
    else:
        print(f"\n[DRY RUN] 將創建目標目錄: {target_dir}")
    
    # 處理每個檔案
    copied_count = 0
    skipped_count = 0
    
    print(f"\n開始處理 {len(t1_files)} 個檔案...")
    print("=" * 60)
    
    for file_path in t1_files:
        # 提取受試者 ID
        subject_id = extract_subject_id(file_path)
        
        # 創建受試者目錄
        subject_dir = target_path / subject_id
        
        # 標準化檔案名稱
        subject_id_underscore = subject_id.replace('-', '_')
        target_file_name = f"{subject_id_underscore}_T1.nii.gz"
        target_file_path = subject_dir / target_file_name
        
        # 檢查是否已存在
        if target_file_path.exists() and not dry_run:
            print(f"⚠️  跳過 {subject_id}: 檔案已存在")
            skipped_count += 1
            continue
        
        # 顯示操作
        print(f"\n處理: {subject_id}")
        print(f"  來源: {file_path}")
        print(f"  目標: {target_file_path}")
        
        if not dry_run:
            # 創建受試者目錄
            subject_dir.mkdir(parents=True, exist_ok=True)
            
            # 複製檔案
            try:
                shutil.copy2(file_path, target_file_path)
                print(f"  ✓ 複製成功")
                copied_count += 1
            except Exception as e:
                print(f"  ❌ 複製失敗: {e}")
        else:
            print(f"  [DRY RUN] 將複製檔案")
            copied_count += 1
    
    # 顯示摘要
    print("\n" + "=" * 60)
    print("處理完成！")
    print(f"  成功: {copied_count} 個檔案")
    print(f"  跳過: {skipped_count} 個檔案")
    print(f"  總計: {len(t1_files)} 個檔案")
    
    if dry_run:
        print("\n[DRY RUN] 這是模擬執行，沒有實際複製檔案")
        print("移除 --dry-run 參數以實際執行")


def verify_data_structure(target_dir="data/sMRI/MCI"):
    """
    驗證資料結構是否正確
    
    Args:
        target_dir: 目標目錄
    """
    target_path = Path(target_dir)
    
    if not target_path.exists():
        print(f"❌ 目標目錄不存在: {target_dir}")
        return False
    
    # 統計受試者數量
    subject_dirs = [d for d in target_path.iterdir() if d.is_dir()]
    
    print(f"\n驗證資料結構...")
    print(f"  目標目錄: {target_dir}")
    print(f"  受試者數量: {len(subject_dirs)}")
    
    # 檢查每個受試者目錄
    valid_count = 0
    invalid_count = 0
    
    for subject_dir in subject_dirs:
        t1_files = list(subject_dir.glob("*_T1.nii.gz"))
        
        if len(t1_files) == 1:
            valid_count += 1
        else:
            print(f"  ⚠️  {subject_dir.name}: 找到 {len(t1_files)} 個 T1 檔案（預期 1 個）")
            invalid_count += 1
    
    print(f"\n驗證結果:")
    print(f"  有效: {valid_count} 個受試者")
    print(f"  無效: {invalid_count} 個受試者")
    
    return invalid_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="遷移 MCI 資料到 data/sMRI/MCI/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 模擬執行（不實際複製）
  python scripts/migrate_mci_data.py --source /path/to/mci/data --dry-run
  
  # 實際執行
  python scripts/migrate_mci_data.py --source /path/to/mci/data
  
  # 使用自訂檔案模式
  python scripts/migrate_mci_data.py --source /path/to/mci/data --pattern "*T1w*.nii.gz"
  
  # 驗證已遷移的資料
  python scripts/migrate_mci_data.py --verify
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        help='MCI 資料的來源目錄'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='data/sMRI/MCI',
        help='目標目錄 (預設: data/sMRI/MCI)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*T1*.nii.gz',
        help='檔案名稱模式 (預設: *T1*.nii.gz)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只顯示將要執行的操作，不實際複製檔案'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='驗證已遷移的資料結構'
    )
    
    args = parser.parse_args()
    
    # 驗證模式
    if args.verify:
        verify_data_structure(args.target)
        return
    
    # 檢查必要參數
    if not args.source:
        parser.error("需要指定 --source 參數（或使用 --verify 驗證已有資料）")
    
    # 執行遷移
    print("=" * 60)
    print("MCI 資料遷移腳本")
    print("=" * 60)
    print(f"來源目錄: {args.source}")
    print(f"目標目錄: {args.target}")
    print(f"檔案模式: {args.pattern}")
    print(f"模式: {'模擬執行 (DRY RUN)' if args.dry_run else '實際執行'}")
    print("=" * 60)
    
    try:
        organize_mci_data(
            source_dir=args.source,
            target_dir=args.target,
            pattern=args.pattern,
            dry_run=args.dry_run
        )
        
        if not args.dry_run:
            print("\n" + "=" * 60)
            verify_data_structure(args.target)
            
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
