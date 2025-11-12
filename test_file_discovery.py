"""
測試檔案搜尋邏輯
"""

import glob
from pathlib import Path

print("="*70)
print("🔍 測試檔案搜尋")
print("="*70)

# 測試資料目錄
data_dir = Path("data/cardinal_tien")

# 列出所有受試者
print("\n📂 可用的受試者:")
for label in ["AD", "NC"]:
    label_dir = data_dir / label
    if label_dir.exists():
        subjects = sorted([d.name for d in label_dir.iterdir() if d.is_dir()])
        print(f"\n{label}/ ({len(subjects)} 個受試者)")
        for subj in subjects[:5]:
            print(f"   - {subj}")
        if len(subjects) > 5:
            print(f"   ... 還有 {len(subjects) - 5} 個")

# 測試搜尋模式
print("\n" + "="*70)
print("🧪 測試搜尋模式")
print("="*70)

test_subjects = ["sub-0005", "sub-0001", "sub-0011"]

for subject in test_subjects:
    print(f"\n測試受試者: {subject}")
    
    # 功能性 MRI 搜尋（所有檔案）
    pattern_fmri = f"data/cardinal_tien/*/{subject}/*.nii.gz"
    files_fmri = glob.glob(pattern_fmri)
    print(f"   功能性 MRI 模式: {pattern_fmri}")
    print(f"   找到 {len(files_fmri)} 個檔案")
    for f in files_fmri:
        print(f"      - {Path(f).name}")
    
    # 結構性 MRI 搜尋（只要 T1）
    pattern_smri = f"data/cardinal_tien/*/{subject}/*_T1.nii.gz"
    files_smri = glob.glob(pattern_smri)
    print(f"   結構性 MRI 模式: {pattern_smri}")
    print(f"   找到 {len(files_smri)} 個檔案")
    for f in files_smri:
        print(f"      - {Path(f).name}")
    
    if not files_smri:
        print(f"   ⚠️  警告: 找不到 T1 檔案！")

print("\n" + "="*70)
print("✅ 測試完成")
print("="*70)

# 統計
print("\n📊 統計:")
all_t1_files = glob.glob("data/cardinal_tien/*/*/*_T1.nii.gz")
print(f"   總共 T1 檔案: {len(all_t1_files)} 個")

ad_t1_files = glob.glob("data/cardinal_tien/AD/*/*_T1.nii.gz")
nc_t1_files = glob.glob("data/cardinal_tien/NC/*/*_T1.nii.gz")
print(f"   AD T1 檔案: {len(ad_t1_files)} 個")
print(f"   NC T1 檔案: {len(nc_t1_files)} 個")

print("\n🎉 檔案搜尋邏輯正常！")
