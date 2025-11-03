"""
檔案管理資料模型
定義檔案相關的 Pydantic 模型
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class FileType(str, Enum):
    """檔案類型枚舉"""
    NIFTI = "nifti"
    DICOM = "dicom"
    METADATA = "metadata"
    UNKNOWN = "unknown"


class PatientFile(BaseModel):
    """患者檔案模型"""
    id: str = Field(..., description="檔案 ID")
    patient_id: str = Field(..., description="患者 ID")
    original_filename: str = Field(..., description="原始檔案名")
    stored_filename: str = Field(..., description="存儲檔案名")
    file_type: FileType = Field(..., description="檔案類型")
    file_path: str = Field(..., description="檔案路徑")
    file_size: int = Field(..., ge=0, description="檔案大小 (bytes)")
    description: Optional[str] = Field(None, description="檔案描述")
    metadata: Optional[Dict[str, Any]] = Field(None, description="檔案元數據")
    uploaded_at: datetime = Field(..., description="上傳時間")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "file_123",
                "patient_id": "patient_456",
                "original_filename": "fmri_scan.nii.gz",
                "stored_filename": "abc123_fmri_scan.nii.gz",
                "file_type": "nifti",
                "file_path": "/storage/patients/patient_456/raw/abc123_fmri_scan.nii.gz",
                "file_size": 52428800,
                "description": "功能性 MRI 掃描",
                "metadata": {
                    "dimensions": [64, 64, 30, 200],
                    "voxel_size": [3.0, 3.0, 4.0, 2.0],
                    "data_type": "float32"
                },
                "uploaded_at": "2025-01-30T10:30:00Z"
            }
        }


class FileMetadata(BaseModel):
    """檔案元數據模型"""
    dimensions: Optional[List[int]] = Field(None, description="影像維度")
    voxel_size: Optional[List[float]] = Field(None, description="體素大小")
    data_type: Optional[str] = Field(None, description="資料類型")
    scanner_info: Optional[Dict[str, Any]] = Field(None, description="掃描儀資訊")
    acquisition_params: Optional[Dict[str, Any]] = Field(None, description="採集參數")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dimensions": [64, 64, 30, 200],
                "voxel_size": [3.0, 3.0, 4.0, 2.0],
                "data_type": "float32",
                "scanner_info": {
                    "manufacturer": "Siemens",
                    "model": "Magnetom Prisma",
                    "field_strength": 3.0
                },
                "acquisition_params": {
                    "tr": 2000,
                    "te": 30,
                    "flip_angle": 90
                }
            }
        }


class FileUploadResponse(BaseModel):
    """檔案上傳響應模型"""
    uploaded_files: List[PatientFile] = Field(..., description="成功上傳的檔案")
    errors: List[str] = Field(default_factory=list, description="上傳錯誤")
    success: bool = Field(..., description="上傳是否成功")
    message: str = Field(..., description="響應訊息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "uploaded_files": [
                    {
                        "id": "file_123",
                        "patient_id": "patient_456",
                        "original_filename": "fmri_scan.nii.gz",
                        "file_type": "nifti",
                        "file_size": 52428800
                    }
                ],
                "errors": [],
                "success": True,
                "message": "成功上傳 1 個檔案"
            }
        }


class FileValidationResult(BaseModel):
    """檔案驗證結果模型"""
    file_id: str = Field(..., description="檔案 ID")
    is_valid: bool = Field(..., description="是否有效")
    checks: List[str] = Field(default_factory=list, description="執行的檢查")
    warnings: List[str] = Field(default_factory=list, description="警告訊息")
    errors: List[str] = Field(default_factory=list, description="錯誤訊息")
    metadata: Optional[Dict[str, Any]] = Field(None, description="提取的元數據")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "file_123",
                "is_valid": True,
                "checks": [
                    "檔案存在性檢查",
                    "檔案大小檢查",
                    "NIfTI 格式驗證"
                ],
                "warnings": [],
                "errors": [],
                "metadata": {
                    "shape": [64, 64, 30, 200],
                    "data_type": "float32",
                    "header_info": "有效"
                }
            }
        }