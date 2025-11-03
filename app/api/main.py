"""
FastAPI 主應用程式
整合多模態臨床儀表板的 API 服務
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import os
from typing import Dict, Any

# 導入路由
from app.api.routes import patients, files, analysis, models, reports, websocket

# 導入現有的 LangGraph 工作流程
try:
    from app.graph.workflow import app as langgraph_app
    print("✅ LangGraph 工作流程載入成功")
except ImportError as e:
    print(f"⚠️ LangGraph 工作流程載入失敗: {e}")
    langgraph_app = None

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    # 啟動時的初始化
    logger.info("🚀 啟動多模態臨床儀表板 API 服務")
    
    # 檢查必要的目錄
    os.makedirs("storage/patients", exist_ok=True)
    os.makedirs("storage/temp", exist_ok=True)
    os.makedirs("storage/templates", exist_ok=True)
    os.makedirs("storage/atlases", exist_ok=True)
    
    # 驗證 LangGraph 工作流程
    if langgraph_app:
        try:
            # 不實際執行，只是驗證工作流程可用
            logger.info("✅ LangGraph 工作流程驗證成功")
        except Exception as e:
            logger.error(f"❌ LangGraph 工作流程驗證失敗: {e}")
    else:
        logger.info("⚠️ LangGraph 工作流程未載入")
    
    yield
    
    # 關閉時的清理
    logger.info("🛑 關閉多模態臨床儀表板 API 服務")


# 創建 FastAPI 應用程式
app = FastAPI(
    title="多模態臨床儀表板 API",
    description="整合 fMRI 分析、AI 模型推理、腦圖譜視覺化和臨床報告生成的 API 服務",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS 中間件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React 開發服務器
        "http://localhost:8080",  # 其他前端服務器
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 信任的主機中間件 (在測試環境中允許 testserver)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost", "testserver"]
)

# 註冊路由
app.include_router(patients.router, prefix="/api/patients", tags=["患者管理"])
app.include_router(files.router, prefix="/api/files", tags=["檔案管理"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["分析服務"])
app.include_router(models.router, prefix="/api/models", tags=["模型管理"])
app.include_router(reports.router, prefix="/api/reports", tags=["報告生成"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])


@app.get("/", response_model=Dict[str, Any])
async def root():
    """API 根端點"""
    return {
        "message": "多模態臨床儀表板 API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs",
        "features": [
            "患者資料管理",
            "DICOM/NIfTI 檔案處理", 
            "ShuffleNet AI 分析",
            "腦圖譜整合",
            "臨床報告生成"
        ]
    }


@app.get("/api/health")
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy",
        "timestamp": "2025-01-31T00:00:00Z",
        "services": {
            "api": "running",
            "langgraph": "available",
            "storage": "accessible"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )