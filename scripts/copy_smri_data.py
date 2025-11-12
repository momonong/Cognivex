"""
複製結構性 MRI 資料從外接硬碟到專案資料夾
從: E:\fMRI\Model\sMRI_data_MultiModal_Aligned_MNI
到: data/raw/
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# 來源和目標路徑
SOURCE_DIR = Path(r"E:\fMRI\Model\sMRI_data_MultiModal_Aligned_MNI")
TARGET_DIR = Path("data/cardinal_tien")

def get_file_size_mb(file_path):
    """取得檔案大小（MB）"""
    return file_path.stat().st_size / (1024 * 1024)

def count_files(directory):
    """計算目錄中的檔案數量"""
    return sum(1 for _ in directory.rglob("*.nii.gz"))

def copy_data():
    """複製資料"""
    
    print("="*70)
    print("🧠 結構性 MRI 資料複製工具")
    print("="*70)
    
    # 檢查來源目錄
    if not SOURCE_DIR.exists():
        print(f"\n❌ 錯誤: 找不到來源目錄")
        print(f"   路徑: {SOURCE_DIR}")
        print(f"\n請確認:")
        print(f"   1. 外接硬碟已連接")
        print(f"   2. 路徑正確")
        return
    
    print(f"\n📂 來源目錄: {SOURCE_DIR}")
    print(f"📂 目標目錄: {TARGET_DIR}")
    
    # 檢查 AD 和 NC 資料夾
    ad_source = SOURCE_DIR / "AD"
    nc_source = SOURCE_DIR / "NC"
    
    if not ad_source.exists():
        print(f"\n⚠️  警告: 找不到 AD 資料夾: {ad_source}")
    if not nc_source.exists():
        print(f"\n⚠️  警告: 找不到 NC 資料夾: {nc_source}")
    
    # 統計檔案
    print("\n📊 統計來源檔案...")
    ad_files = list(ad_source.rglob("*.nii.gz")) if ad_source.exists() else []
    nc_files = list(nc_source.rglob("*.nii.gz")) if nc_source.exists() else []
    
    total_files = len(ad_files) + len(nc_files)
    
    if total_files == 0:
        print("\n❌ 錯誤: 找不到任何 .nii.gz 檔案")
        return
    
    print(f"\n找到檔案:")
    print(f"   AD: {len(ad_files)} 個檔案")
    print(f"   NC: {len(nc_files)} 個檔案")
    print(f"   總計: {total_files} 個檔案")
    
    # 計算總大小
    total_size = sum(get_file_size_mb(f) for f in ad_files + nc_files)
    print(f"\n總大小: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    # 確認
    print("\n" + "="*70)
    response = input("確定要開始複製嗎？ (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\n❌ 取消複製")
        return
    
    # 建立目標目錄
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 複製檔案
    print("\n" + "="*70)
    print("📦 開始複製檔案...")
    print("="*70)
    
    copied_files = 0
    copied_size = 0
    skipped_files = 0
    
    # 複製 AD 資料
    if ad_files:
        print(f"\n[1/2] 複製 AD 資料 ({len(ad_files)} 個檔案)...")
        ad_target = TARGET_DIR / "AD"
        ad_target.mkdir(parents=True, exist_ok=True)
        
        for file_path in tqdm(ad_files, desc="AD", unit="file"):
            # 取得相對路徑（從 AD 資料夾開始）
            rel_path = file_path.relative_to(ad_source)
            target_path = ad_target / rel_path
            
            # 建立目標子目錄
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 檢查檔案是否已存在
            if target_path.exists():
                # 比較檔案大小，如果相同就跳過
                if target_path.stat().st_size == file_path.stat().st_size:
                    skipped_files += 1
                    continue
            
            # 複製檔案
            try:
                shutil.copy2(file_path, target_path)
                copied_files += 1
                copied_size += get_file_size_mb(file_path)
            except Exception as e:
                print(f"\n⚠️  複製失敗: {file_path.name}")
                print(f"   錯誤: {e}")
    
    # 複製 NC 資料
    if nc_files:
        print(f"\n[2/2] 複製 NC 資料 ({len(nc_files)} 個檔案)...")
        nc_target = TARGET_DIR / "NC"
        nc_target.mkdir(parents=True, exist_ok=True)
        
        for file_path in tqdm(nc_files, desc="NC", unit="file"):
            # 取得相對路徑（從 NC 資料夾開始）
            rel_path = file_path.relative_to(nc_source)
            target_path = nc_target / rel_path
            
            # 建立目標子目錄
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 檢查檔案是否已存在
            if target_path.exists():
                # 比較檔案大小，如果相同就跳過
                if target_path.stat().st_size == file_path.stat().st_size:
                    skipped_files += 1
                    continue
            
            # 複製檔案
            try:
                shutil.copy2(file_path, target_path)
                copied_files += 1
                copied_size += get_file_size_mb(file_path)
            except Exception as e:
                print(f"\n⚠️  複製失敗: {file_path.name}")
                print(f"   錯誤: {e}")
    
    # 完成
    print("\n" + "="*70)
    print("✅ 複製完成！")
    print("="*70)
    
    print(f"\n📊 統計:")
    print(f"   複製檔案: {copied_files} 個")
    print(f"   跳過檔案: {skipped_files} 個（已存在）")
    print(f"   複製大小: {copied_size:.2f} MB ({copied_size/1024:.2f} GB)")
    
    # 驗證目標目錄
    print(f"\n📂 目標目錄結構:")
    ad_target = TARGET_DIR / "AD"
    nc_target = TARGET_DIR / "NC"
    
    if ad_target.exists():
        ad_subjects = sorted([d.name for d in ad_target.iterdir() if d.is_dir()])
        print(f"\n   AD/ ({len(ad_subjects)} 個受試者)")
        for subj in ad_subjects[:5]:  # 只顯示前 5 個
            files = list((ad_target / subj).glob("*.nii.gz"))
            print(f"      - {subj}/ ({len(files)} 個檔案)")
        if len(ad_subjects) > 5:
            print(f"      ... 還有 {len(ad_subjects) - 5} 個受試者")
    
    if nc_target.exists():
        nc_subjects = sorted([d.name for d in nc_target.iterdir() if d.is_dir()])
        print(f"\n   NC/ ({len(nc_subjects)} 個受試者)")
        for subj in nc_subjects[:5]:  # 只顯示前 5 個
            files = list((nc_target / subj).glob("*.nii.gz"))
            print(f"      - {subj}/ ({len(files)} 個檔案)")
        if len(nc_subjects) > 5:
            print(f"      ... 還有 {len(nc_subjects) - 5} 個受試者")
    
    print("\n🎉 資料已準備好！")
    print(f"\n下一步:")
    print(f"   1. 啟動應用: streamlit run app.py")
    print(f"   2. 選擇 'Structural MRI (T1)' 模式")
    print(f"   3. 選擇受試者並開始分析")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        copy_data()
    except KeyboardInterrupt:
        print("\n\n❌ 使用者中斷")
    except Exception as e:
        print(f"\n\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
