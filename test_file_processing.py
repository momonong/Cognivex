#!/usr/bin/env python3
"""
檔案處理功能測試
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import nibabel as nib

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.api.services.file_processor import file_processor

def create_test_nifti():
    """創建測試用的 NIfTI 檔案"""
    # 創建 4D fMRI 測試數據
    data = np.random.randn(64, 64, 30, 100).astype(np.float32)
    
    # 創建仿射矩陣
    affine = np.eye(4)
    affine[:3, :3] = np.diag([3.0, 3.0, 4.0])  # 體素大小
    
    # 創建 NIfTI 影像
    img = nib.Nifti1Image(data, affine)
    
    # 設定 TR
    img.header.set_zooms([3.0, 3.0, 4.0, 2.0])  # 最後一個是 TR
    
    return img

def test_file_processor():
    """測試檔案處理器功能"""
    
    print("🧪 開始測試檔案處理功能")
    print("-" * 50)
    
    # 測試 1: 檔案格式驗證
    print("1️⃣ 測試檔案格式驗證...")
    
    test_files = [
        "test.nii.gz",
        "test.nii", 
        "test.dcm",
        "test.json",
        "test.txt"
    ]
    
    for filename in test_files:
        is_valid = file_processor.validate_file_format(filename)
        file_type = file_processor.get_file_type(filename)
        print(f"   {filename}: 有效={is_valid}, 類型={file_type}")
    
    # 測試 2: NIfTI 元數據提取
    print("\n2️⃣ 測試 NIfTI 元數據提取...")
    
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
        # 創建測試 NIfTI 檔案
        test_img = create_test_nifti()
        nib.save(test_img, tmp_file.name)
        
        # 提取元數據
        metadata = file_processor.extract_nifti_metadata(tmp_file.name)
        
        print(f"   檔案路徑: {tmp_file.name}")
        print(f"   維度: {metadata.get('dimensions')}")
        print(f"   體素大小: {metadata.get('voxel_size')}")
        print(f"   數據類型: {metadata.get('data_type')}")
        print(f"   是否為 4D: {metadata.get('is_4d')}")
        print(f"   時間點數: {metadata.get('time_points')}")
        print(f"   TR: {metadata.get('tr')}")
        
        # 測試 3: NIfTI 檔案驗證
        print("\n3️⃣ 測試 NIfTI 檔案驗證...")
        
        validation = file_processor.validate_nifti_file(tmp_file.name)
        
        print(f"   驗證結果: {'通過' if validation['is_valid'] else '失敗'}")
        print(f"   檢查項目: {validation['checks']}")
        if validation['warnings']:
            print(f"   警告: {validation['warnings']}")
        if validation['errors']:
            print(f"   錯誤: {validation['errors']}")
        
        # 測試 4: 預處理建議
        print("\n4️⃣ 測試預處理建議...")
        
        recommendations = file_processor.get_preprocessing_recommendations(metadata)
        print(f"   建議:")
        for i, rec in enumerate(recommendations, 1):
            print(f"     {i}. {rec}")
        
        # 清理臨時檔案
        Path(tmp_file.name).unlink()
    
    # 測試 5: JSON 元數據處理
    print("\n5️⃣ 測試 JSON 元數據處理...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_json:
        test_json_data = {
            "scanner": "Siemens Magnetom Prisma",
            "field_strength": 3.0,
            "tr": 2000,
            "te": 30,
            "patient_id": "test_001"
        }
        
        import json
        json.dump(test_json_data, tmp_json)
        tmp_json.flush()
        
        json_metadata = file_processor.extract_json_metadata(tmp_json.name)
        
        print(f"   JSON 內容: {json_metadata.get('content')}")
        print(f"   鍵值: {json_metadata.get('keys')}")
        
        # 清理臨時檔案
        Path(tmp_json.name).unlink()
    
    print("\n" + "-" * 50)
    print("🎉 檔案處理功能測試完成！")

if __name__ == "__main__":
    test_file_processor()