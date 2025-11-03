"""
分析服務 API 路由
處理 ShuffleNet AI 分析請求和狀態管理
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any
from uuid import uuid4
from datetime import datetime
import asyncio
import logging

from app.api.models.analysis import (
    AnalysisRequest, AnalysisResponse, AnalysisResult,
    AnalysisStatus, AnalysisOptions
)

# 導入簡化版 ShuffleNet 分析服務
from app.api.services.shufflenet_service_simple import simple_shufflenet_service as shufflenet_service

router = APIRouter()
logger = logging.getLogger(__name__)

# 分析任務存儲 - 後續會替換為資料庫
analyses_db: dict = {}
analysis_progress: dict = {}


async def run_analysis_workflow(analysis_id: str, request: AnalysisRequest):
    """執行分析工作流程的背景任務"""
    try:
        # 更新進度
        analysis_progress[analysis_id] = {
            "progress": 0.1,
            "current_step": "初始化分析",
            "status": "running"
        }
        
        # 更新進度
        analysis_progress[analysis_id]["progress"] = 0.2
        analysis_progress[analysis_id]["current_step"] = "載入 ShuffleNet 模型"
        
        # 執行 ShuffleNet 分析
        logger.info(f"開始執行 ShuffleNet 分析 {analysis_id}")
        
        # 更新進度
        analysis_progress[analysis_id]["progress"] = 0.3
        analysis_progress[analysis_id]["current_step"] = "執行 ShuffleNet 推理"
        
        # 使用真實的 ShuffleNet 服務
        analysis_result = await shufflenet_service.run_analysis(
            patient_id=request.patient_id,
            fmri_file_path=request.fmri_file_path,
            analysis_options=request.analysis_options.dict() if request.analysis_options else None
        )
        
        # 更新進度
        analysis_progress[analysis_id]["progress"] = 0.8
        analysis_progress[analysis_id]["current_step"] = "處理分析結果"
        
        # 檢查分析是否成功
        if not analysis_result["success"]:
            raise Exception(analysis_result.get("error", "分析失敗"))
        
        # 構建 final_state 格式以保持兼容性
        final_state = {
            "classification_result": analysis_result["prediction"],
            "confidence_score": analysis_result["confidence"],
            "processing_time": analysis_result["processing_time"],
            "visualization_paths": analysis_result["visualization_paths"],
            "generated_reports": analysis_result["generated_reports"],
            "activated_regions": analysis_result["brain_regions"],
            "trace_log": analysis_result["trace_log"]
        }
        
        # 更新進度
        analysis_progress[analysis_id]["progress"] = 0.9
        analysis_progress[analysis_id]["current_step"] = "處理分析結果"
        
        # 處理結果
        if final_state:
            # 創建默認的品質指標
            from app.api.models.analysis import QualityMetrics
            default_quality_metrics = QualityMetrics(
                overall_score=0.85,
                motion_score=0.90,
                signal_noise_ratio=15.0,
                temporal_stability=0.88
            )
            
            result = AnalysisResult(
                id=analysis_id,
                patient_id=request.patient_id,
                model_name="shufflenet",
                prediction=final_state.get("classification_result", "Unknown"),
                confidence=final_state.get("confidence_score", 0.0),
                processing_time=final_state.get("processing_time", 0.0),
                visualization_paths=final_state.get("visualization_paths", []),
                generated_reports=final_state.get("generated_reports", {}),
                brain_regions=[],  # 後續整合腦圖譜時填充
                functional_networks=[],  # 後續整合功能網路時填充
                quality_metrics=default_quality_metrics,
                status=AnalysisStatus.COMPLETED,
                created_at=datetime.now(),
                completed_at=datetime.now()
            )
            
            analyses_db[analysis_id]["result"] = result
            analyses_db[analysis_id]["status"] = AnalysisStatus.COMPLETED
            
            # 完成進度
            analysis_progress[analysis_id] = {
                "progress": 1.0,
                "current_step": "分析完成",
                "status": "completed"
            }
            
            logger.info(f"分析 {analysis_id} 完成成功")
        else:
            raise Exception("LangGraph 工作流程返回空結果")
            
    except Exception as e:
        logger.error(f"分析 {analysis_id} 失敗: {str(e)}")
        
        # 更新錯誤狀態
        analyses_db[analysis_id]["status"] = AnalysisStatus.FAILED
        analyses_db[analysis_id]["error"] = str(e)
        
        analysis_progress[analysis_id] = {
            "progress": 0.0,
            "current_step": f"分析失敗: {str(e)}",
            "status": "failed"
        }


@router.post("/start", response_model=AnalysisResponse)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """開始新的分析任務"""
    
    analysis_id = str(uuid4())
    
    # 創建分析記錄
    analysis_record = {
        "id": analysis_id,
        "patient_id": request.patient_id,
        "request": request,
        "status": AnalysisStatus.PENDING,
        "created_at": datetime.now(),
        "result": None,
        "error": None
    }
    
    analyses_db[analysis_id] = analysis_record
    
    # 初始化進度
    analysis_progress[analysis_id] = {
        "progress": 0.0,
        "current_step": "排隊等待",
        "status": "pending"
    }
    
    # 添加背景任務
    background_tasks.add_task(run_analysis_workflow, analysis_id, request)
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.PENDING,
        message="分析任務已創建並開始執行",
        created_at=datetime.now()
    )


@router.get("/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    """獲取分析狀態"""
    if analysis_id not in analyses_db:
        raise HTTPException(status_code=404, detail="分析任務未找到")
    
    analysis = analyses_db[analysis_id]
    progress_info = analysis_progress.get(analysis_id, {})
    
    return {
        "analysis_id": analysis_id,
        "status": analysis["status"],
        "progress": progress_info.get("progress", 0.0),
        "current_step": progress_info.get("current_step", "未知"),
        "created_at": analysis["created_at"],
        "error": analysis.get("error")
    }


@router.get("/{analysis_id}/results", response_model=AnalysisResult)
async def get_analysis_results(analysis_id: str):
    """獲取分析結果"""
    if analysis_id not in analyses_db:
        raise HTTPException(status_code=404, detail="分析任務未找到")
    
    analysis = analyses_db[analysis_id]
    
    if analysis["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"分析尚未完成，當前狀態: {analysis['status']}"
        )
    
    if not analysis["result"]:
        raise HTTPException(status_code=500, detail="分析結果不可用")
    
    return analysis["result"]


@router.get("/patient/{patient_id}")
async def get_patient_analyses(patient_id: str):
    """獲取患者的所有分析記錄"""
    patient_analyses = [
        {
            "analysis_id": analysis["id"],
            "status": analysis["status"],
            "created_at": analysis["created_at"],
            "progress": analysis_progress.get(analysis["id"], {}).get("progress", 0.0)
        }
        for analysis in analyses_db.values()
        if analysis["patient_id"] == patient_id
    ]
    
    return {
        "patient_id": patient_id,
        "analyses": patient_analyses,
        "total_count": len(patient_analyses)
    }


@router.delete("/{analysis_id}")
async def cancel_analysis(analysis_id: str):
    """取消或刪除分析任務"""
    if analysis_id not in analyses_db:
        raise HTTPException(status_code=404, detail="分析任務未找到")
    
    analysis = analyses_db[analysis_id]
    
    # 如果正在執行，標記為取消
    if analysis["status"] in [AnalysisStatus.PENDING, AnalysisStatus.RUNNING]:
        analysis["status"] = AnalysisStatus.CANCELLED
        if analysis_id in analysis_progress:
            analysis_progress[analysis_id]["status"] = "cancelled"
    
    # 刪除記錄
    del analyses_db[analysis_id]
    if analysis_id in analysis_progress:
        del analysis_progress[analysis_id]
    
    return {
        "message": "分析任務已取消/刪除",
        "success": True
    }