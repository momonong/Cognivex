"""
重新組織資料結構
從: data/raw/AD/sub_XXXX_T1.nii.gz
到: data/raw/AD/sub-XXXX/sub_XXXX_T1.nii.gz
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

# 目標路徑
DATA_DIR = Path("data/cardinal_tien")

def reorganize_data():
    """重新組織資料結構"""
    
    print("="*70)
    print("📁 重新組織資料結構")
    print("="*70)
    
    for label in ["AD", "NC"]:
        label_dir = DATA_DIR / label
        
        if not label_dir.exists():
            print(f"\n⚠️  跳過 {label}（目錄不存在）")
            continue
        
        print(f"\n處理 {label} 資料...")
        
        # 找到所有 .nii.gz 檔案
        files = list(label_dir.glob("*.nii.gz"))
        
        if not files:
            print(f"   沒有找到檔案")
            continue
        
        # 按受試者分組
        subject_files = defaultdict(list)
        for file_path in files:
            # 從檔名提取受試者 ID
            # 例如: sub_0005_T1.nii.gz -> sub_0005
            parts = file_path.stem.replace(".nii", "").split("_")
            if len(parts) >= 2:
                subject_id = f"{parts[0]}_{parts[1]}"  # sub_0005
                subject_files[subject_id].append(file_path)
        
        print(f"   找到 {len(subject_files)} 個受試者")
        
        # 為每個受試者建立子目錄並移動檔案
        for subject_id, file_list in subject_files.items():
            # 建立受試者目錄（使用 sub-XXXX 格式）
            subject_dir_name = subject_id.replace("_", "-")  # sub_0005 -> sub-0005
            subject_dir = label_dir / subject_dir_name
            subject_dir.mkdir(exist_ok=True)
            
            # 移動檔案
            for file_path in file_list:
                target_path = subject_dir / file_path.name
                if not target_path.exists():
                    shutil.move(str(file_path), str(target_path))
        
        print(f"   ✅ 完成 {label}")
    
    # 驗證結果
    print("\n" + "="*70)
    print("📊 驗證結果")
    print("="*70)
    
    for label in ["AD", "NC"]:
        label_dir = DATA_DIR / label
        
        if not label_dir.exists():
            continue
        
        # 統計受試者和檔案
        subjects = [d for d in label_dir.iterdir() if d.is_dir()]
        total_files = sum(len(list(s.glob("*.nii.gz"))) for s in subjects)
        
        print(f"\n{label}/")
        print(f"   受試者: {len(subjects)} 個")
        print(f"   檔案: {total_files} 個")
        
        # 顯示前幾個受試者
        for subj in sorted(subjects)[:3]:
            files = list(subj.glob("*.nii.gz"))
            print(f"      {subj.name}/ ({len(files)} 個檔案)")
            for f in files:
                print(f"         - {f.name}")
        
        if len(subjects) > 3:
            print(f"      ... 還有 {len(subjects) - 3} 個受試者")
    
    print("\n" + "="*70)
    print("✅ 重新組織完成！")
    print("="*70)
    
    print("\n🎉 資料結構已更新！")
    print("\n現在可以:")
    print("   1. 啟動應用: streamlit run app.py")
    print("   2. 選擇 'Structural MRI (T1)' 模式")
    print("   3. 選擇受試者（例如：sub-0005）")
    print("   4. 開始分析")


if __name__ == "__main__":
    try:
        reorganize_data()
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
