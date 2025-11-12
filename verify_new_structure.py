"""
驗證新的資料結構
"""

import glob
from pathlib import Path

print("="*70)
print("✅ 驗證新的資料結構")
print("="*70)

# 檢查新的資料夾
print("\n[1] 檢查資料夾存在...")
folders = {
    "data/fMRI": "功能性 MRI",
    "data/sMRI": "結構性 MRI"
}

for folder, desc in folders.items():
    path = Path(folder)
    if path.exists():
        print(f"   ✅ {folder} ({desc})")
    else:
        print(f"   ❌ {folder} 不存在")

# 檢查功能性 MRI 資料
print("\n[2] 檢查功能性 MRI 資料 (data/fMRI)...")
fmri_folders = glob.glob("data/fMRI/*/sub-*")
print(f"   找到 {len(fmri_folders)} 個受試者資料夾")

if fmri_folders:
    # 按標籤分組
    ad_count = len([f for f in fmri_folders if "/AD/" in f or "\\AD\\" in f])
    nc_count = len([f for f in fmri_folders if "/NC/" in f or "\\NC\\" in f or "/CN/" in f or "\\CN\\" in f])
    print(f"   - AD: {ad_count} 個")
    print(f"   - NC/CN: {nc_count} 個")
    
    # 顯示範例
    print(f"\n   範例:")
    for folder in fmri_folders[:3]:
        files = list(Path(folder).glob("*.nii.gz"))
        print(f"      {folder} ({len(files)} 個檔案)")

# 檢查結構性 MRI 資料
print("\n[3] 檢查結構性 MRI 資料 (data/sMRI)...")
smri_folders = glob.glob("data/sMRI/*/sub-*")
print(f"   找到 {len(smri_folders)} 個受試者資料夾")

if smri_folders:
    # 按標籤分組
    ad_count = len([f for f in smri_folders if "/AD/" in f or "\\AD\\" in f])
    nc_count = len([f for f in smri_folders if "/NC/" in f or "\\NC\\" in f])
    print(f"   - AD: {ad_count} 個")
    print(f"   - NC: {nc_count} 個")
    
    # 顯示範例
    print(f"\n   範例:")
    for folder in smri_folders[:3]:
        files = list(Path(folder).glob("*.nii.gz"))
        t1_files = [f for f in files if "_T1" in f.name]
        print(f"      {folder}")
        print(f"         總檔案: {len(files)} 個")
        print(f"         T1 檔案: {len(t1_files)} 個")

# 測試搜尋模式
print("\n[4] 測試搜尋模式...")

# 功能性 MRI
print("\n   功能性 MRI:")
test_subjects_fmri = ["sub-002", "sub-003", "sub-004"]
for subj in test_subjects_fmri:
    pattern = f"data/fMRI/*/{subj}/*.nii.gz"
    files = glob.glob(pattern)
    if files:
        print(f"      ✅ {subj}: 找到 {len(files)} 個檔案")
        break
else:
    # 如果沒找到，列出實際的受試者
    actual_subjects = [Path(f).name for f in fmri_folders[:3]]
    print(f"      實際受試者: {', '.join(actual_subjects)}")

# 結構性 MRI
print("\n   結構性 MRI:")
test_subjects_smri = ["sub-0005", "sub-0001", "sub-0011"]
for subj in test_subjects_smri:
    subject_folder = subj.replace("_", "-")
    pattern = f"data/sMRI/*/{subject_folder}/*_T1.nii.gz"
    files = glob.glob(pattern)
    if files:
        print(f"      ✅ {subj} ({subject_folder}): 找到 {len(files)} 個 T1 檔案")

# 總結
print("\n" + "="*70)
print("📊 總結")
print("="*70)

print("\n新的資料結構:")
print("data/")
print("├── fMRI/          (功能性 MRI)")
fmri_ad = len([f for f in fmri_folders if '/AD/' in f or '\\AD\\' in f]) if fmri_folders else 0
fmri_nc = len([f for f in fmri_folders if '/NC/' in f or '\\NC\\' in f or '/CN/' in f or '\\CN\\' in f]) if fmri_folders else 0
print(f"│   ├── AD/        ({fmri_ad} 個受試者)")
print(f"│   └── NC/CN/     ({fmri_nc} 個受試者)")
print("└── sMRI/          (結構性 MRI)")
smri_ad = len([f for f in smri_folders if '/AD/' in f or '\\AD\\' in f]) if smri_folders else 0
smri_nc = len([f for f in smri_folders if '/NC/' in f or '\\NC\\' in f]) if smri_folders else 0
print(f"    ├── AD/        ({smri_ad} 個受試者)")
print(f"    └── NC/        ({smri_nc} 個受試者)")

print("\n✅ 資料結構驗證完成！")
print("\n下一步:")
print("   1. 重新啟動 Streamlit: streamlit run app.py")
print("   2. 選擇分析模式")
print("   3. 選擇受試者並開始分析")

print("\n" + "="*70)
