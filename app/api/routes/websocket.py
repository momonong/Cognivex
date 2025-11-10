"""
WebSocket API 路由
提供即時分析進度通知
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# WebSocket 連接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, analysis_id: str):
        """建立 WebSocket 連接"""
        await websocket.accept()
        if analysis_id not in self.active_connections:
            self.active_connections[analysis_id] = []
        self.active_connections[analysis_id].append(websocket)
        logger.info(f"WebSocket 連接建立: {analysis_id}")
    
    def disconnect(self, websocket: WebSocket, analysis_id: str):
        """斷開 WebSocket 連接"""
        if analysis_id in self.active_connections:
            if websocket in self.active_connections[analysis_id]:
                self.active_connections[analysis_id].remove(websocket)
            if not self.active_connections[analysis_id]:
                del self.active_connections[analysis_id]
        logger.info(f"WebSocket 連接斷開: {analysis_id}")
    
    async def send_personal_message(self, message: dict, analysis_id: str):
        """發送訊息給特定分析的所有連接"""
        if analysis_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[analysis_id]:
                try:
                    await connection.send_text(json.dumps(message, ensure_ascii=False))
                except Exception as e:
                    logger.error(f"發送訊息失敗: {e}")
                    disconnected.append(connection)
            
            # 清理斷開的連接
            for connection in disconnected:
                self.disconnect(connection, analysis_id)
    
    async def broadcast(self, message: dict):
        """廣播訊息給所有連接"""
        for analysis_id in self.active_connections:
            await self.send_personal_message(message, analysis_id)

manager = ConnectionManager()


@router.websocket("/analysis/{analysis_id}")
async def websocket_analysis_progress(websocket: WebSocket, analysis_id: str):
    """分析進度 WebSocket 端點"""
    await manager.connect(websocket, analysis_id)
    
    try:
        # 發送歡迎訊息
        welcome_message = {
            "event": "connection_established",
            "data": {
                "analysis_id": analysis_id,
                "message": "WebSocket 連接已建立",
                "timestamp": "2025-01-31T00:00:00Z"
            }
        }
        await websocket.send_text(json.dumps(welcome_message, ensure_ascii=False))
        
        # 保持連接並處理客戶端訊息
        while True:
            try:
                # 等待客戶端訊息
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # 處理客戶端請求
                if message.get("type") == "ping":
                    pong_message = {
                        "event": "pong",
                        "data": {
                            "timestamp": "2025-01-31T00:00:00Z"
                        }
                    }
                    await websocket.send_text(json.dumps(pong_message, ensure_ascii=False))
                
                elif message.get("type") == "get_status":
                    # 這裡可以查詢當前分析狀態並發送
                    status_message = {
                        "event": "status_update",
                        "data": {
                            "analysis_id": analysis_id,
                            "status": "running",
                            "progress": 0.5,
                            "current_step": "執行 ShuffleNet 推理",
                            "timestamp": "2025-01-31T00:00:00Z"
                        }
                    }
                    await websocket.send_text(json.dumps(status_message, ensure_ascii=False))
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                error_message = {
                    "event": "error",
                    "data": {
                        "message": "無效的 JSON 格式",
                        "timestamp": "2025-01-31T00:00:00Z"
                    }
                }
                await websocket.send_text(json.dumps(error_message, ensure_ascii=False))
            except Exception as e:
                logger.error(f"WebSocket 處理錯誤: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, analysis_id)


# 輔助函數：從其他服務發送進度更新
async def send_analysis_progress(analysis_id: str, progress: float, step: str, status: str = "running"):
    """發送分析進度更新"""
    message = {
        "event": "analysis_progress",
        "data": {
            "analysis_id": analysis_id,
            "progress": progress,
            "current_step": step,
            "status": status,
            "timestamp": "2025-01-31T00:00:00Z"
        }
    }
    await manager.send_personal_message(message, analysis_id)


async def send_analysis_complete(analysis_id: str, result: dict):
    """發送分析完成通知"""
    message = {
        "event": "analysis_complete",
        "data": {
            "analysis_id": analysis_id,
            "status": "completed",
            "result": result,
            "timestamp": "2025-01-31T00:00:00Z"
        }
    }
    await manager.send_personal_message(message, analysis_id)


async def send_analysis_error(analysis_id: str, error: str):
    """發送分析錯誤通知"""
    message = {
        "event": "analysis_error",
        "data": {
            "analysis_id": analysis_id,
            "status": "failed",
            "error": error,
            "timestamp": "2025-01-31T00:00:00Z"
        }
    }
    await manager.send_personal_message(message, analysis_id)


# 測試端點
@router.websocket("/test")
async def websocket_test(websocket: WebSocket):
    """WebSocket 測試端點"""
    await websocket.accept()
    
    try:
        # 發送測試訊息
        for i in range(10):
            test_message = {
                "event": "test_progress",
                "data": {
                    "step": i + 1,
                    "total": 10,
                    "progress": (i + 1) / 10,
                    "message": f"測試步驟 {i + 1}/10",
                    "timestamp": "2025-01-31T00:00:00Z"
                }
            }
            await websocket.send_text(json.dumps(test_message, ensure_ascii=False))
            await asyncio.sleep(1)
        
        # 發送完成訊息
        complete_message = {
            "event": "test_complete",
            "data": {
                "message": "WebSocket 測試完成",
                "timestamp": "2025-01-31T00:00:00Z"
            }
        }
        await websocket.send_text(json.dumps(complete_message, ensure_ascii=False))
        
    except WebSocketDisconnect:
        logger.info("WebSocket 測試連接斷開")
    except Exception as e:
        logger.error(f"WebSocket 測試錯誤: {e}")
    finally:
        await websocket.close()