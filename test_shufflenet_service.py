#!/usr/bin/env python3
"""
測試 ShuffleNet 服務
"""

import sys
import asyncio
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.api.services.shufflenet_service_simple import simple_shufflenet_service


async def test_shufflenet_service():
    """測試 ShuffleNet 服務功能"""
    
    print("🧪 開始測試 ShuffleNet 服務")
    print("-" * 60)
    
    # 測試 1: 獲取模型資訊
    print("1️⃣ 測試獲取模型資訊...")
    model_info = simple_shufflenet_service.get_model_info()
    print(f"   模型名稱: {model_info['model_name']}")
    print(f"   模型類型: {model_info['model_type']}")
    print(f"   模型存在: {model_info['model_exists']}")
    print(f"   模型載入: {model_info['model_loaded']}")
    print(f"   設備: {model_info['device']}")
    
    # 測試 2: 健康檢查
    print("\n2️⃣ 測試健康檢查...")
    health_status = simple_shufflenet_service.health_check()
    print(f"   健康狀態: {health_status['status']}")
    print(f"   檢查項目:")
    for check in health_status['checks']:
        print(f"     {check}")
    
    if health_status['issues']:
        print(f"   問題:")
        for issue in health_status['issues']:
            print(f"     {issue}")
    
    # 測試 3: 檔案驗證
    print("\n3️⃣ 測試檔案驗證...")
    
    # 測試不存在的檔案
    validation = simple_shufflenet_service.validate_input_file("nonexistent.nii.gz")
    print(f"   不存在檔案驗證: {'通過' if not validation['is_valid'] else '失敗'}")
    
    # 測試錯誤格式
    validation = simple_shufflenet_service.validate_input_file("test.txt")
    print(f"   錯誤格式驗證: {'通過' if not validation['is_valid'] else '失敗'}")
    
    # 測試真實檔案 (如果存在)
    test_files = [
        "data/raw/AD/sub-01/dswausub-009_S_0751_task-rest_bold.nii.gz",
        "data/raw/NC/sub-01/dswausub-009_S_0751_task-rest_bold.nii.gz"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            validation = simple_shufflenet_service.validate_input_file(test_file)
            print(f"   檔案 {test_file}: {'有效' if validation['is_valid'] else '無效'}")
            if validation['file_info']:
                print(f"     維度: {validation['file_info'].get('dimensions')}")
                print(f"     大小: {validation['file_info'].get('size_mb')} MB")
            break
    else:
        print("   ⚠️ 找不到測試用的 NIfTI 檔案")
    
    # 測試 4: 模擬分析 (如果模型可用)
    if model_info['model_loaded']:
        print("\n4️⃣ 測試模擬分析...")
        
        # 找一個真實的檔案進行測試
        for test_file in test_files:
            if Path(test_file).exists():
                print(f"   使用檔案: {test_file}")
                
                try:
                    result = await simple_shufflenet_service.run_analysis(
                        patient_id="test_patient",
                        fmri_file_path=test_file
                    )
                    
                    print(f"   分析結果:")
                    print(f"     成功: {result['success']}")
                    print(f"     預測: {result['prediction']}")
                    print(f"     信心度: {result['confidence']:.3f}")
                    print(f"     處理時間: {result['processing_time']:.2f} 秒")
                    
                    if result['trace_log']:
                        print(f"     執行日誌:")
                        for log in result['trace_log']:
                            print(f"       - {log}")
                    
                    if result['error']:
                        print(f"     錯誤: {result['error']}")
                    
                except Exception as e:
                    print(f"   ❌ 分析測試失敗: {e}")
                
                break
        else:
            print("   ⚠️ 跳過分析測試 - 找不到測試檔案")
    else:
        print("\n4️⃣ 跳過分析測試 - 模型未載入")
    
    print("\n" + "-" * 60)
    print("🎉 ShuffleNet 服務測試完成！")


if __name__ == "__main__":
    asyncio.run(test_shufflenet_service())