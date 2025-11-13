"""
重新命名資料夾以更直觀
data/raw -> data/fMRI (功能性 MRI)
data/cardinal_tien -> data/sMRI (結構性 MRI)
"""

import shutil
from pathlib import Path

print("="*70)
print("📁 重新命名資料夾")
print("="*70)

# 定義重新命名對應
renames = [
    ("data/raw", "data/fMRI"),
    ("data/cardinal_tien", "data/sMRI")
]

for old_path, new_path in renames:
    old = Path(old_path)
    new = Path(new_path)
    
    print(f"\n處理: {old_path} -> {new_path}")
    
    if old.exists():
        if new.exists():
            print(f"   ⚠️  目標已存在: {new_path}")
            response = input(f"   是否覆蓋？ (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print(f"   ⏭️  跳過")
                continue
            # 刪除舊的目標
            shutil.rmtree(new)
            print(f"   🗑️  已刪除舊的 {new_path}")
        
        # 重新命名
        shutil.move(str(old), str(new))
        print(f"   ✅ 已重新命名")
    else:
        print(f"   ❌ 來源不存在: {old_path}")

# 驗證結果
print("\n" + "="*70)
print("📊 驗證結果")
print("="*70)

for old_path, new_path in renames:
    new = Path(new_path)
    if new.exists():
        # 統計內容
        subdirs = [d for d in new.iterdir() if d.is_dir()]
        files = [f for f in new.iterdir() if f.is_file()]
        
        print(f"\n✅ {new_path}")
        print(f"   子目錄: {len(subdirs)} 個")
        print(f"   檔案: {len(files)} 個")
        
        if subdirs:
            for d in subdirs[:3]:
                print(f"      - {d.name}/")
    else:
        print(f"\n❌ {new_path} 不存在")

print("\n" + "="*70)
print("✅ 完成！")
print("="*70)

print("\n新的資料結構:")
print("data/")
print("├── fMRI/     (功能性 MRI 資料)")
print("│   ├── AD/")
print("│   └── NC/")
print("└── sMRI/     (結構性 MRI 資料)")
print("    ├── AD/")
print("    └── NC/")
