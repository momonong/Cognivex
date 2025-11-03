#!/usr/bin/env python3
"""
FastAPI 服務測試腳本
驗證 API 基本功能
"""

import sys
from pathlib import Path
import asyncio
import json

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from app.api.main import app

def test_api_basic_functionality():
    """測試 API 基本功能"""
    
    print("🧪 開始測試 FastAPI 基本功能")
    print("-" * 50)
    
    # 創建測試客戶端
    client = TestClient(app)
    
    # 測試 1: 根端點
    print("1️⃣ 測試根端點...")
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    print(f"   ✅ 根端點正常: {data['message']}")
    
    # 測試 2: 健康檢查
    print("2️⃣ 測試健康檢查...")
    response = client.get("/api/health")
    assert response.status_code == 200
    health_data = response.json()
    assert health_data["status"] == "healthy"
    print(f"   ✅ 健康檢查正常: {health_data['status']}")
    
    # 測試 3: 獲取可用模型
    print("3️⃣ 測試模型管理...")
    response = client.get("/api/models/")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    print(f"   ✅ 模型列表獲取成功: 找到 {len(models)} 個模型")
    
    # 測試 4: 患者管理 - 獲取患者列表
    print("4️⃣ 測試患者管理...")
    response = client.get("/api/patients/")
    assert response.status_code == 200
    patients = response.json()
    assert isinstance(patients, list)
    print(f"   ✅ 患者列表獲取成功: 目前有 {len(patients)} 個患者")
    
    # 測試 5: 創建測試患者
    print("5️⃣ 測試創建患者...")
    test_patient = {
        "name": "測試患者",
        "age": 65,
        "gender": "M",
        "diagnosis": "AD",
        "scan_date": "2025-01-30T10:00:00Z",
        "hospital_info": {
            "institution_name": "測試醫院",
            "department": "神經內科",
            "scanner_model": "Siemens Magnetom Prisma",
            "magnetic_field_strength": 3.0
        },
        "clinical_notes": "測試患者記錄"
    }
    
    response = client.post("/api/patients/", json=test_patient)
    assert response.status_code == 200
    patient_response = response.json()
    assert patient_response["success"] is True
    patient_id = patient_response["patient"]["id"]
    print(f"   ✅ 患者創建成功: ID = {patient_id}")
    
    # 測試 6: 獲取患者詳情
    print("6️⃣ 測試獲取患者詳情...")
    response = client.get(f"/api/patients/{patient_id}")
    assert response.status_code == 200
    patient_detail = response.json()
    assert patient_detail["name"] == "測試患者"
    print(f"   ✅ 患者詳情獲取成功: {patient_detail['name']}")
    
    print("-" * 50)
    print("🎉 所有基本功能測試通過！")
    
    return True

def test_api_error_handling():
    """測試 API 錯誤處理"""
    
    print("\n🧪 開始測試 API 錯誤處理")
    print("-" * 50)
    
    client = TestClient(app)
    
    # 測試 1: 獲取不存在的患者
    print("1️⃣ 測試獲取不存在的患者...")
    response = client.get("/api/patients/nonexistent-id")
    assert response.status_code == 404
    print("   ✅ 正確返回 404 錯誤")
    
    # 測試 2: 無效的患者數據
    print("2️⃣ 測試無效的患者數據...")
    invalid_patient = {
        "name": "",  # 空名稱
        "age": -5,   # 無效年齡
        "gender": "INVALID"  # 無效性別
    }
    
    response = client.post("/api/patients/", json=invalid_patient)
    assert response.status_code == 422  # Validation error
    print("   ✅ 正確返回驗證錯誤")
    
    # 測試 3: 獲取不存在的模型
    print("3️⃣ 測試獲取不存在的模型...")
    response = client.get("/api/models/nonexistent-model")
    assert response.status_code == 404
    print("   ✅ 正確返回 404 錯誤")
    
    print("-" * 50)
    print("🎉 所有錯誤處理測試通過！")
    
    return True

def main():
    """主測試函數"""
    
    print("🚀 FastAPI 服務測試開始")
    print("=" * 60)
    
    try:
        # 基本功能測試
        test_api_basic_functionality()
        
        # 錯誤處理測試
        test_api_error_handling()
        
        print("\n" + "=" * 60)
        print("🎊 所有測試完成！API 服務運行正常")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)