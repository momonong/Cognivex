"""
分析服務資料模型
定義分析相關的 Pydantic 模型
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AnalysisStatus(str, Enum):
    """分析狀態枚舉"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisOptions(BaseModel):
    """分析選項模型"""
    include_grad_cam: bool = Field(True, description="是否包含 Grad-CAM 視覺化")
    include_network_analysis: bool = Field(True, description="是否包含功能網路分析")
    atlas_type: str = Field("aal3", description="腦圖譜類型")
    network_type: str = Field("yeo7", description="功能網路類型")
    quality_threshold: float = Field(0.8, ge=0.0, le=1.0, description="品質閾值")
    
    class Config:
        json_schema_extra = {
            "example": {
                "include_grad_cam": True,
                "include_network_analysis": True,
                "atlas_type": "aal3",
                "network_type": "yeo7",
                "quality_threshold": 0.8
            }
        }


class AnalysisRequest(BaseModel):
    """分析請求模型"""
    patient_id: str = Field(..., description="患者 ID")
    fmri_file_path: str = Field(..., description="fMRI 檔案路徑")
    model_path: Optional[str] = Field(None, description="模型檔案路徑")
    analysis_options: AnalysisOptions = Field(default_factory=AnalysisOptions, description="分析選項")
    priority: int = Field(1, ge=1, le=5, description="優先級 (1-5)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "patient_123",
                "fmri_file_path": "/storage/patients/patient_123/raw/fmri_scan.nii.gz",
                "model_path": "model/shufflenet/fold_3_best_model.pth",
                "analysis_options": {
                    "include_grad_cam": True,
                    "include_network_analysis": True,
                    "atlas_type": "aal3",
                    "network_type": "yeo7",
                    "quality_threshold": 0.8
                },
                "priority": 1
            }
        }


class BrainRegion(BaseModel):
    """腦區模型"""
    aal3_label: str = Field(..., description="AAL3 標籤")
    region_name: str = Field(..., description="腦區名稱")
    activation_score: float = Field(..., ge=0.0, le=1.0, description="活化分數")
    yeo_network: str = Field(..., description="Yeo 功能網路")
    hemisphere: str = Field(..., description="大腦半球")
    coordinates: List[float] = Field(..., description="MNI 座標 [x, y, z]")
    
    class Config:
        json_schema_extra = {
            "example": {
                "aal3_label": "Hippocampus_L",
                "region_name": "左側海馬迴",
                "activation_score": 0.75,
                "yeo_network": "Default Mode Network",
                "hemisphere": "Left",
                "coordinates": [-25.0, -15.0, -20.0]
            }
        }


class FunctionalNetwork(BaseModel):
    """功能網路模型"""
    network_name: str = Field(..., description="網路名稱")
    yeo_id: int = Field(..., description="Yeo 網路 ID")
    activation_strength: float = Field(..., ge=0.0, le=1.0, description="活化強度")
    connectivity_score: float = Field(..., ge=0.0, le=1.0, description="連接性分數")
    clinical_significance: str = Field(..., description="臨床意義")
    
    class Config:
        json_schema_extra = {
            "example": {
                "network_name": "Default Mode Network",
                "yeo_id": 7,
                "activation_strength": 0.68,
                "connectivity_score": 0.72,
                "clinical_significance": "與自我參照和記憶相關，AD 患者常見異常"
            }
        }


class QualityMetrics(BaseModel):
    """品質指標模型"""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="整體品質分數")
    motion_score: float = Field(..., ge=0.0, le=1.0, description="運動偽影分數")
    signal_noise_ratio: float = Field(..., ge=0.0, description="信噪比")
    temporal_stability: float = Field(..., ge=0.0, le=1.0, description="時間穩定性")
    
    class Config:
        json_schema_extra = {
            "example": {
                "overall_score": 0.85,
                "motion_score": 0.92,
                "signal_noise_ratio": 15.6,
                "temporal_stability": 0.88
            }
        }


class AnalysisResult(BaseModel):
    """分析結果模型"""
    id: str = Field(..., description="分析 ID")
    patient_id: str = Field(..., description="患者 ID")
    model_name: str = Field(..., description="使用的模型名稱")
    prediction: str = Field(..., description="預測結果")
    confidence: float = Field(..., ge=0.0, le=1.0, description="信心分數")
    processing_time: float = Field(..., ge=0.0, description="處理時間 (秒)")
    visualization_paths: List[str] = Field(default_factory=list, description="視覺化檔案路徑")
    generated_reports: Dict[str, str] = Field(default_factory=dict, description="生成的報告")
    brain_regions: List[BrainRegion] = Field(default_factory=list, description="腦區分析結果")
    functional_networks: List[FunctionalNetwork] = Field(default_factory=list, description="功能網路分析結果")
    quality_metrics: QualityMetrics = Field(..., description="品質指標")
    status: AnalysisStatus = Field(..., description="分析狀態")
    created_at: datetime = Field(..., description="創建時間")
    completed_at: Optional[datetime] = Field(None, description="完成時間")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "analysis_123",
                "patient_id": "patient_456",
                "model_name": "shufflenet",
                "prediction": "AD",
                "confidence": 0.85,
                "processing_time": 45.2,
                "visualization_paths": [
                    "/storage/patients/patient_456/analysis/heatmap.png"
                ],
                "generated_reports": {
                    "zh": "中文報告內容...",
                    "en": "English report content..."
                },
                "brain_regions": [],
                "functional_networks": [],
                "quality_metrics": {
                    "overall_score": 0.85,
                    "motion_score": 0.92,
                    "signal_noise_ratio": 15.6,
                    "temporal_stability": 0.88
                },
                "status": "completed",
                "created_at": "2025-01-30T10:00:00Z",
                "completed_at": "2025-01-30T10:45:00Z"
            }
        }


class AnalysisResponse(BaseModel):
    """分析響應模型"""
    analysis_id: str = Field(..., description="分析 ID")
    status: AnalysisStatus = Field(..., description="分析狀態")
    message: str = Field(..., description="響應訊息")
    created_at: datetime = Field(..., description="創建時間")
    estimated_completion: Optional[datetime] = Field(None, description="預估完成時間")
    
    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": "analysis_123",
                "status": "pending",
                "message": "分析任務已創建並開始執行",
                "created_at": "2025-01-30T10:00:00Z",
                "estimated_completion": "2025-01-30T10:45:00Z"
            }
        }