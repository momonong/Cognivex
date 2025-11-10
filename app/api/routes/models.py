"""
模型管理 API 路由
處理 ShuffleNet 模型資訊和健康檢查
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import os

router = APIRouter()


@router.get("/", response_model=List[Dict[str, Any]])
async def get_available_models():
    """獲取可用的 AI 模型列表"""
    try:
        # 使用 ShuffleNet 服務獲取模型資訊
        from app.api.services.shufflenet_service_simple import simple_shufflenet_service
        model_info = simple_shufflenet_service.get_model_info()
        
        model_status = {
            "model_id": "shufflenet",
            "name": model_info.get("model_name", "ShuffleNet"),
            "version": "1.0",
            "description": model_info.get("description", "ShuffleNet 2D with ECA Attention"),
            "accuracy": model_info.get("accuracy", "80%+"),
            "input_format": model_info.get("input_format", "2D brain slices"),
            "output_classes": model_info.get("output_classes", ["AD", "NC"]),
            "model_file": model_info.get("model_path", ""),
            "architecture": model_info.get("model_type", "2D CNN"),
            "training_data": "ADNI fMRI dataset",
            "status": "available" if model_info.get("model_exists", False) else "missing",
            "file_exists": model_info.get("model_exists", False),
            "model_loaded": model_info.get("model_loaded", False)
        }
        
        # 獲取檔案大小
        if model_info.get("model_exists", False):
            try:
                file_size = os.path.getsize(model_info.get("model_path", ""))
                model_status["file_size"] = file_size
            except Exception:
                model_status["file_size"] = None
        else:
            model_status["file_size"] = None
        
        return [model_status]
        
    except Exception as e:
        # 如果 ShuffleNet 服務不可用，返回基本資訊
        return [{
            "model_id": "shufflenet",
            "name": "ShuffleNet 2D with ECA Attention",
            "status": "error",
            "error": str(e),
            "file_exists": False
        }]


@router.get("/{model_id}")
async def get_model_info(model_id: str):
    """獲取特定模型的詳細資訊"""
    if model_id != "shufflenet":
        raise HTTPException(status_code=404, detail="模型未找到")
    
    try:
        from app.api.services.shufflenet_service_simple import simple_shufflenet_service
        model_info = simple_shufflenet_service.get_model_info()
        model_path = model_info.get("model_path", "")
        
        # 檢查模型檔案
        file_exists = os.path.exists(model_path)
        
        result = {
            "model_id": model_id,
            "name": model_info.get("model_name", "ShuffleNet"),
            "description": model_info.get("description", ""),
            "version": "1.0",
            "accuracy": model_info.get("accuracy", "80%+"),
            "input_format": model_info.get("input_format", "2D brain slices"),
            "output_classes": model_info.get("output_classes", ["AD", "NC"]),
            "status": "available" if file_exists else "missing",
            "file_exists": file_exists,
            "file_path": model_path,
            "model_loaded": model_info.get("model_loaded", False),
            "device": model_info.get("device", "cpu"),
            "architecture_details": model_info.get("architecture", {}),
            "preprocessing": model_info.get("preprocessing", {}),
            "performance": {
                "accuracy": model_info.get("accuracy", "80%+"),
                "training_data": "ADNI fMRI dataset",
                "validation": "5-fold cross-validation"
            }
        }
        
        if file_exists:
            try:
                # 獲取檔案資訊
                file_stat = os.stat(model_path)
                result.update({
                    "file_size": file_stat.st_size,
                    "last_modified": file_stat.st_mtime,
                    "file_permissions": oct(file_stat.st_mode)[-3:]
                })
            except Exception as e:
                result["file_error"] = str(e)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取模型資訊失敗: {str(e)}")


@router.post("/{model_id}/health-check")
async def model_health_check(model_id: str):
    """執行模型健康檢查"""
    if model_id != "shufflenet":
        raise HTTPException(status_code=404, detail="模型未找到")
    
    try:
        from app.api.services.shufflenet_service_simple import simple_shufflenet_service
        health_status = simple_shufflenet_service.health_check()
        
        return {
            "model_id": model_id,
            "timestamp": "2025-01-31T00:00:00Z",
            "status": health_status.get("status", "unknown"),
            "checks": health_status.get("checks", []),
            "issues": health_status.get("issues", []),
            "model_info": health_status.get("model_info", {})
        }
        
    except Exception as e:
        return {
            "model_id": model_id,
            "timestamp": "2025-01-31T00:00:00Z",
            "status": "unhealthy",
            "checks": [],
            "issues": [f"健康檢查失敗: {str(e)}"],
            "model_info": {}
        }


@router.get("/{model_id}/config")
async def get_model_config(model_id: str):
    """獲取模型配置參數"""
    if model_id != "shufflenet":
        raise HTTPException(status_code=404, detail="模型未找到")
    
    try:
        from app.api.services.shufflenet_service_simple import simple_shufflenet_service
        model_info = simple_shufflenet_service.get_model_info()
        
        return {
            "model_id": model_id,
            "architecture": model_info.get("architecture", {}),
            "preprocessing": model_info.get("preprocessing", {}),
            "input_size": [128, 128],  # 2D slice size
            "num_classes": 2,
            "class_names": ["NC", "AD"],
            "inference_config": {
                "batch_size": 1,
                "device": model_info.get("device", "cpu"),
                "precision": "float32"
            },
            "performance": {
                "accuracy": model_info.get("accuracy", "80%+"),
                "inference_time": "< 1 second",
                "memory_usage": "< 2GB"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取模型配置失敗: {str(e)}")