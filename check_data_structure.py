"""
全面檢查資料結構
"""

import os
from pathlib import Path
import glob

print("="*70)
print("🔍 全面資料結構檢查")
print("="*70)

# 檢查所有可能的資料目錄
data_locations = [
    "data/raw",
    "data/cardinal_tien",
    "data/processed",
    "data",
    "E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI"
]

print("\n[1] 檢查資料目錄是否存在...")
for loc in data_locations:
    path = Path(loc)
    exists = "✅" if path.exists() else "❌"
    print(f"   {exists} {loc}")
    if path.exists():
        # 列出子目錄
        subdirs = [d.name for d in path.iterdir() if d.is_dir()]
        if subdirs:
            print(f"      子目錄: {', '.join(subdirs[:5])}")
            if len(subdirs) > 5:
                print(f"      ... 還有 {len(subdirs) - 5} 個")

# 檢查 data/raw 的詳細結構
print("\n[2] 檢查 data/raw 詳細結構...")
raw_path = Path("data/raw")
if raw_path.exists():
    print(f"   📂 {raw_path} 存在")
    
    # 檢查 AD 和 NC
    for label in ["AD", "NC"]:
        label_path = raw_path / label
        if label_path.exists():
            print(f"\n   📂 {label}/")
            
            # 檢查是否有子目錄
            subdirs = [d for d in label_path.iterdir() if d.is_dir()]
            files = [f for f in label_path.iterdir() if f.is_file()]
            
            print(f"      子目錄數: {len(subdirs)}")
            print(f"      檔案數: {len(files)}")
            
            if subdirs:
                print(f"      子目錄範例:")
                for d in subdirs[:3]:
                    sub_files = list(d.glob("*.nii.gz"))
                    print(f"         - {d.name}/ ({len(sub_files)} 個檔案)")
            
            if files:
                print(f"      檔案範例:")
                for f in files[:5]:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"         - {f.name} ({size_mb:.2f} MB)")
                if len(files) > 5:
                    print(f"         ... 還有 {len(files) - 5} 個檔案")
        else:
            print(f"   ❌ {label}/ 不存在")
else:
    print(f"   ❌ {raw_path} 不存在")

# 檢查 data/cardinal_tien 的詳細結構
print("\n[3] 檢查 data/cardinal_tien 詳細結構...")
ct_path = Path("data/cardinal_tien")
if ct_path.exists():
    print(f"   📂 {ct_path} 存在")
    
    for label in ["AD", "NC"]:
        label_path = ct_path / label
        if label_path.exists():
            print(f"\n   📂 {label}/")
            
            subdirs = [d for d in label_path.iterdir() if d.is_dir()]
            files = [f for f in label_path.iterdir() if f.is_file()]
            
            print(f"      子目錄數: {len(subdirs)}")
            print(f"      檔案數: {len(files)}")
            
            if subdirs:
                print(f"      子目錄範例:")
                for d in subdirs[:3]:
                    sub_files = list(d.glob("*.nii.gz"))
                    print(f"         - {d.name}/ ({len(sub_files)} 個檔案)")
                    for sf in sub_files[:2]:
                        print(f"            * {sf.name}")
        else:
            print(f"   ❌ {label}/ 不存在")
else:
    print(f"   ❌ {ct_path} 不存在")

# 搜尋所有 T1 檔案
print("\n[4] 搜尋所有 T1 檔案...")
patterns = [
    "data/raw/*/*.nii.gz",
    "data/raw/*/*/*.nii.gz",
    "data/cardinal_tien/*/*.nii.gz",
    "data/cardinal_tien/*/*/*.nii.gz",
]

for pattern in patterns:
    files = glob.glob(pattern)
    t1_files = [f for f in files if "_T1" in f or "T1" in f]
    if t1_files:
        print(f"\n   模式: {pattern}")
        print(f"   找到 {len(t1_files)} 個 T1 檔案")
        for f in t1_files[:3]:
            print(f"      - {f}")
        if len(t1_files) > 3:
            print(f"      ... 還有 {len(t1_files) - 3} 個")

# 檢查外接硬碟
print("\n[5] 檢查外接硬碟...")
external_path = Path("E:/fMRI/Model/sMRI_data_MultiModal_Aligned_MNI")
if external_path.exists():
    print(f"   ✅ 外接硬碟路徑存在")
    for label in ["AD", "NC"]:
        label_path = external_path / label
        if label_path.exists():
            files = list(label_path.glob("*.nii.gz"))
            t1_files = [f for f in files if "_T1" in f.name]
            print(f"      {label}: {len(t1_files)} 個 T1 檔案")
else:
    print(f"   ❌ 外接硬碟路徑不存在")

# 總結
print("\n" + "="*70)
print("📊 總結")
print("="*70)

# 統計所有 T1 檔案
all_t1_patterns = [
    "data/**/*_T1.nii.gz",
    "data/**/*T1*.nii.gz",
]

total_t1 = 0
for pattern in all_t1_patterns:
    files = glob.glob(pattern, recursive=True)
    total_t1 += len(files)

print(f"\n專案中總共找到 {total_t1} 個 T1 檔案")

# 建議
print("\n💡 建議:")
if total_t1 == 0:
    print("   ❌ 專案中沒有找到任何 T1 檔案")
    print("   建議:")
    print("   1. 執行 python scripts/copy_smri_data.py 複製資料")
    print("   2. 或手動複製資料到 data/raw/ 或 data/cardinal_tien/")
else:
    print(f"   ✅ 找到 {total_t1} 個 T1 檔案")
    print("   請確認 app.py 使用正確的資料路徑")

print("\n" + "="*70)
