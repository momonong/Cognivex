#!/usr/bin/env python3
"""
簡單的 ShuffleNet 測試 - 使用模擬數據
"""

import sys
import asyncio
import tempfile
import numpy as np
import nibabel as nib
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.api.services.shufflenet_service_simple import simple_shufflenet_service


def create_test_fmri_data():
    """創建測試用的 4D fMRI 數據"""
    # 創建模擬的 4D fMRI 數據 (64x64x30x100)
    data = np.random.randn(64, 64, 30, 100).astype(np.float32)
    
    # 添加一些結構性信號
    for z in range(30):
        for t in range(100):
            # 創建一個簡單的腦部結構
            center_x, center_y = 32, 32
            radius = 20
            
            y, x = np.ogrid[:64, :64]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # 在腦部區域添加信號
            data[mask, z, t] += np.random.normal(0.5, 0.1)
    
    # 創建仿射矩陣
    affine = np.eye(4)
    affine[:3, :3] = np.diag([3.0, 3.0, 4.0])  # 體素大小
    
    # 創建 NIfTI 影像
    img = nib.Nifti1Image(data, affine)
    
    return img


async def test_with_simulated_data():
    """使用模擬數據測試 ShuffleNet 分析"""
    
    print("🧪 開始 ShuffleNet 模擬數據測試")
    print("-" * 60)
    
    # 創建臨時測試檔案
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
        # 創建測試數據
        print("1️⃣ 創建模擬 fMRI 數據...")
        test_img = create_test_fmri_data()
        nib.save(test_img, tmp_file.name)
        
        print(f"   測試檔案: {tmp_file.name}")
        print(f"   數據形狀: {test_img.shape}")
        
        # 驗證檔案
        print("\n2️⃣ 驗證測試檔案...")
        validation = simple_shufflenet_service.validate_input_file(tmp_file.name)
        
        print(f"   驗證結果: {'通過' if validation['is_valid'] else '失敗'}")
        if validation['file_info']:
            print(f"   檔案資訊:")
            print(f"     維度: {validation['file_info'].get('dimensions')}")
            print(f"     大小: {validation['file_info'].get('size_mb')} MB")
            print(f"     4D 數據: {validation['file_info'].get('is_4d')}")
            if validation['file_info'].get('is_4d'):
                print(f"     時間點: {validation['file_info'].get('time_points')}")
        
        if validation['warnings']:
            print(f"   警告: {validation['warnings']}")
        
        # 執行分析
        if validation['is_valid']:
            print("\n3️⃣ 執行 ShuffleNet 分析...")
            
            try:
                result = await simple_shufflenet_service.run_analysis(
                    patient_id="test_patient_001",
                    fmri_file_path=tmp_file.name
                )
                
                print(f"\n📊 分析結果:")
                print(f"   成功: {result['success']}")
                print(f"   患者 ID: {result['patient_id']}")
                print(f"   模型: {result['model_name']}")
                print(f"   預測: {result['prediction']}")
                print(f"   信心度: {result['confidence']:.4f}")
                print(f"   處理時間: {result['processing_time']:.2f} 秒")
                
                if result['trace_log']:
                    print(f"\n📝 執行日誌:")
                    for i, log in enumerate(result['trace_log'], 1):
                        print(f"   {i}. {log}")
                
                if result['generated_reports']:
                    print(f"\n📋 生成報告:")
                    if 'zh' in result['generated_reports']:
                        print(f"   中文: {result['generated_reports']['zh']}")
                    if 'en' in result['generated_reports']:
                        print(f"   英文: {result['generated_reports']['en']}")
                
                if result['error']:
                    print(f"\n❌ 錯誤: {result['error']}")
                
                # 測試預處理功能
                print("\n4️⃣ 測試數據預處理...")
                try:
                    input_tensor = simple_shufflenet_service.preprocess_nifti_data(tmp_file.name)
                    print(f"   預處理成功")
                    print(f"   輸入張量形狀: {input_tensor.shape}")
                    print(f"   數據範圍: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
                    
                except Exception as e:
                    print(f"   預處理失敗: {e}")
                
            except Exception as e:
                print(f"❌ 分析失敗: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print("❌ 檔案驗證失敗，跳過分析")
        
        # 清理臨時檔案
        try:
            Path(tmp_file.name).unlink()
            print(f"\n🧹 清理臨時檔案: {tmp_file.name}")
        except:
            pass
    
    print("\n" + "-" * 60)
    print("🎉 ShuffleNet 模擬數據測試完成！")


async def test_model_performance():
    """測試模型效能"""
    
    print("\n🚀 開始 ShuffleNet 效能測試")
    print("-" * 60)
    
    # 獲取模型資訊
    model_info = simple_shufflenet_service.get_model_info()
    print(f"模型資訊:")
    print(f"  名稱: {model_info['model_name']}")
    print(f"  架構: {model_info['architecture']['backbone']}")
    print(f"  注意力機制: {model_info['architecture']['attention']}")
    print(f"  設備: {model_info['device']}")
    
    # 健康檢查
    health = simple_shufflenet_service.health_check()
    print(f"\n健康狀態: {health['status']}")
    
    if health['model_info']:
        print(f"模型詳情:")
        print(f"  檔案大小: {health['model_info'].get('file_size_mb', 'N/A')} MB")
        print(f"  PyTorch 版本: {health['model_info'].get('pytorch_version', 'N/A')}")
        print(f"  CUDA 可用: {health['model_info'].get('cuda_available', False)}")
        
        if health['model_info'].get('cuda_available'):
            print(f"  CUDA 設備數: {health['model_info'].get('cuda_device_count', 0)}")
            print(f"  CUDA 設備名: {health['model_info'].get('cuda_device_name', 'N/A')}")


if __name__ == "__main__":
    async def main():
        await test_with_simulated_data()
        await test_model_performance()
    
    asyncio.run(main())