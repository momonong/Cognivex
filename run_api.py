#!/usr/bin/env python3
"""
FastAPI 服務啟動腳本
多模態臨床儀表板 API 服務
"""

import uvicorn
import os
import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """啟動 FastAPI 服務"""
    
    # 環境配置
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    log_level = os.getenv("API_LOG_LEVEL", "info")
    
    print("🚀 啟動多模態臨床儀表板 API 服務")
    print(f"📍 服務地址: http://{host}:{port}")
    print(f"📚 API 文檔: http://{host}:{port}/api/docs")
    print(f"🔄 熱重載: {'啟用' if reload else '停用'}")
    print(f"📝 日誌級別: {log_level}")
    print("-" * 50)
    
    # 啟動服務
    uvicorn.run(
        "app.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()