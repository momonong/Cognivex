#!/usr/bin/env python3
"""
測試 ShuffleNet API 整合
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

from fastapi.testclient import TestClient
from app.api.main import app


def create_test_fmri_file():
    """創建測試用的 fMRI 檔案"""
    # 創建模擬的 4D fMRI 數據
    data = np.random.randn(64, 64, 30, 100).astype(np.float32)
    
    # 添加一些結構性信號
    for z in range(30):
        for t in range(100):
            center_x, center_y = 32, 32
            radius = 20
            
            y, x = np.ogrid[:64, :64]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            data[mask, z, t] += np.random.normal(0.5, 0.1)
    
    # 創建 NIfTI 影像
    affine = np.eye(4)
    affine[:3, :3] = np.diag([3.0, 3.0, 4.0])
    img = nib.Nifti1Image(data, affine)
    
    # 保存到臨時檔案
    tmp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    nib.save(img, tmp_file.name)
    
    return tmp_file.name


def test_shufflenet_api_integration():
    """測試 ShuffleNet API 整合"""
    
    print("🧪 開始測試 ShuffleNet API 整合")
    print("=" * 70)
    
    # 創建測試客戶端
    client = TestClient(app)
    
    # 測試 1: 檢查 API 基本功能
    print("1️⃣ 測試 API 基本功能...")
    
    response = client.get("/")
    assert response.status_code == 200
    print("   ✅ 根端點正常")
    
    response = client.get("/api/health")
    assert response.status_code == 200
    print("   ✅ 健康檢查正常")
    
    # 測試 2: 測試模型管理 API
    print("\n2️⃣ 測試模型管理 API...")
    
    response = client.get("/api/models/")
    assert response.status_code == 200
    models = response.json()
    print(f"   ✅ 獲取模型列表成功: {len(models)} 個模型")
    
    # 檢查 ShuffleNet 模型
    shufflenet_model = None
    for model in models:
        if model.get("model_id") == "shufflenet":
            shufflenet_model = model
            break
    
    if shufflenet_model:
        print(f"   ✅ 找到 ShuffleNet 模型")
        print(f"     狀態: {shufflenet_model.get('status')}")
        print(f"     檔案存在: {shufflenet_model.get('file_exists')}")
    else:
        print("   ⚠️ 未找到 ShuffleNet 模型")
    
    # 測試 3: 測試 ShuffleNet 模型詳情
    print("\n3️⃣ 測試 ShuffleNet 模型詳情...")
    
    response = client.get("/api/models/shufflenet")
    if response.status_code == 200:
        model_info = response.json()
        print("   ✅ 獲取 ShuffleNet 詳情成功")
        print(f"     名稱: {model_info.get('name')}")
        print(f"     準確度: {model_info.get('accuracy')}")
        print(f"     檔案大小: {model_info.get('file_size')} bytes")
    else:
        print(f"   ❌ 獲取模型詳情失敗: {response.status_code}")
    
    # 測試 4: 測試 ShuffleNet 健康檢查
    print("\n4️⃣ 測試 ShuffleNet 健康檢查...")
    
    response = client.post("/api/models/shufflenet/health-check")
    if response.status_code == 200:
        health_result = response.json()
        print("   ✅ ShuffleNet 健康檢查成功")
        print(f"     狀態: {health_result.get('status')}")
        print(f"     檢查項目: {len(health_result.get('checks', []))}")
        
        if health_result.get('issues'):
            print(f"     問題: {health_result['issues']}")
    else:
        print(f"   ❌ 健康檢查失敗: {response.status_code}")
    
    # 測試 5: 創建測試患者
    print("\n5️⃣ 創建測試患者...")
    
    test_patient = {
        "name": "ShuffleNet 測試患者",
        "age": 70,
        "gender": "M",
        "diagnosis": "AD",
        "scan_date": "2025-01-31T10:00:00Z",
        "hospital_info": {
            "institution_name": "測試醫院",
            "department": "神經內科",
            "scanner_model": "Siemens Magnetom Prisma",
            "magnetic_field_strength": 3.0
        },
        "clinical_notes": "ShuffleNet API 測試患者"
    }
    
    response = client.post("/api/patients/", json=test_patient)
    assert response.status_code == 200
    patient_response = response.json()
    patient_id = patient_response["patient"]["id"]
    print(f"   ✅ 患者創建成功: {patient_id}")
    
    # 測試 6: 上傳測試 fMRI 檔案
    print("\n6️⃣ 上傳測試 fMRI 檔案...")
    
    # 創建測試檔案
    test_file_path = create_test_fmri_file()
    
    try:
        with open(test_file_path, 'rb') as f:
            files = {"files": ("test_fmri.nii.gz", f, "application/gzip")}
            data = {"description": "ShuffleNet 測試用 fMRI 數據"}
            
            response = client.post(f"/api/files/upload/{patient_id}", files=files, data=data)
        
        if response.status_code == 200:
            upload_response = response.json()
            print("   ✅ 檔案上傳成功")
            
            if upload_response["uploaded_files"]:
                uploaded_file = upload_response["uploaded_files"][0]
                file_id = uploaded_file["id"]
                file_path = uploaded_file["file_path"]
                print(f"     檔案 ID: {file_id}")
                print(f"     檔案路徑: {file_path}")
                
                # 測試 7: 開始 ShuffleNet 分析
                print("\n7️⃣ 開始 ShuffleNet 分析...")
                
                analysis_request = {
                    "patient_id": patient_id,
                    "fmri_file_path": file_path,
                    "analysis_options": {
                        "include_grad_cam": True,
                        "include_network_analysis": True,
                        "atlas_type": "aal3",
                        "network_type": "yeo7",
                        "quality_threshold": 0.8
                    },
                    "priority": 1
                }
                
                response = client.post("/api/analysis/start", json=analysis_request)
                
                if response.status_code == 200:
                    analysis_response = response.json()
                    analysis_id = analysis_response["analysis_id"]
                    print(f"   ✅ 分析任務創建成功: {analysis_id}")
                    print(f"     狀態: {analysis_response['status']}")
                    
                    # 等待分析完成
                    import time
                    max_wait = 30  # 最多等待 30 秒
                    wait_time = 0
                    
                    while wait_time < max_wait:
                        time.sleep(2)
                        wait_time += 2
                        
                        # 檢查分析狀態
                        status_response = client.get(f"/api/analysis/{analysis_id}/status")
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            print(f"     進度: {status_data.get('progress', 0):.1%} - {status_data.get('current_step', 'Unknown')}")
                            
                            if status_data.get('status') == 'completed':
                                print("   ✅ 分析完成！")
                                
                                # 獲取分析結果
                                results_response = client.get(f"/api/analysis/{analysis_id}/results")
                                if results_response.status_code == 200:
                                    results = results_response.json()
                                    print(f"\n📊 分析結果:")
                                    print(f"     預測: {results.get('prediction')}")
                                    print(f"     信心度: {results.get('confidence', 0):.3f}")
                                    print(f"     處理時間: {results.get('processing_time', 0):.2f} 秒")
                                else:
                                    print(f"   ❌ 獲取分析結果失敗: {results_response.status_code}")
                                break
                            elif status_data.get('status') == 'failed':
                                print(f"   ❌ 分析失敗: {status_data.get('error', 'Unknown error')}")
                                break
                    else:
                        print("   ⚠️ 分析超時")
                
                else:
                    print(f"   ❌ 分析任務創建失敗: {response.status_code}")
                    print(f"     錯誤: {response.text}")
            
            else:
                print("   ❌ 沒有成功上傳的檔案")
        
        else:
            print(f"   ❌ 檔案上傳失敗: {response.status_code}")
            print(f"     錯誤: {response.text}")
    
    finally:
        # 清理測試檔案
        try:
            Path(test_file_path).unlink()
        except:
            pass
    
    print("\n" + "=" * 70)
    print("🎉 ShuffleNet API 整合測試完成！")


if __name__ == "__main__":
    test_shufflenet_api_integration()