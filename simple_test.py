#!/usr/bin/env python3
"""
簡單的 FastAPI 測試
"""

import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from fastapi.testclient import TestClient
    from app.api.main import app
    
    print("✅ 成功導入 FastAPI 應用程式")
    
    # 創建測試客戶端
    client = TestClient(app)
    print("✅ 成功創建測試客戶端")
    
    # 測試根端點
    print("🧪 測試根端點...")
    response = client.get("/")
    print(f"狀態碼: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"響應數據: {data}")
        print("✅ 根端點測試成功")
    else:
        print(f"❌ 根端點測試失敗: {response.text}")
        
except Exception as e:
    print(f"❌ 錯誤: {str(e)}")
    import traceback
    traceback.print_exc()