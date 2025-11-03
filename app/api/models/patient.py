"""
患者資料模型
定義患者相關的 Pydantic 模型
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class Gender(str, Enum):
    """性別枚舉"""
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


class HospitalInfo(BaseModel):
    """醫院資訊模型"""
    institution_name: str = Field(..., description="醫療機構名稱")
    department: str = Field(..., description="科室")
    scanner_model: str = Field(..., description="掃描儀型號")
    magnetic_field_strength: float = Field(..., description="磁場強度 (Tesla)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "institution_name": "台北榮民總醫院",
                "department": "神經內科",
                "scanner_model": "Siemens Magnetom Prisma",
                "magnetic_field_strength": 3.0
            }
        }


class PatientBase(BaseModel):
    """患者基礎模型"""
    name: str = Field(..., min_length=1, max_length=100, description="患者姓名")
    age: int = Field(..., ge=0, le=150, description="年齡")
    gender: Gender = Field(..., description="性別")
    diagnosis: Optional[str] = Field(None, max_length=200, description="診斷")
    scan_date: datetime = Field(..., description="掃描日期")
    hospital_info: Optional[HospitalInfo] = Field(None, description="醫院資訊")
    clinical_notes: Optional[str] = Field(None, max_length=1000, description="臨床備註")


class PatientCreate(PatientBase):
    """創建患者請求模型"""
    pass


class PatientUpdate(BaseModel):
    """更新患者請求模型"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[Gender] = None
    diagnosis: Optional[str] = Field(None, max_length=200)
    scan_date: Optional[datetime] = None
    hospital_info: Optional[HospitalInfo] = None
    clinical_notes: Optional[str] = Field(None, max_length=1000)


class Patient(PatientBase):
    """完整患者模型"""
    id: str = Field(..., description="患者 ID")
    created_at: datetime = Field(..., description="創建時間")
    updated_at: datetime = Field(..., description="更新時間")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "王小明",
                "age": 65,
                "gender": "M",
                "diagnosis": "AD",
                "scan_date": "2025-01-30T10:00:00Z",
                "hospital_info": {
                    "institution_name": "台北榮民總醫院",
                    "department": "神經內科",
                    "scanner_model": "Siemens Magnetom Prisma",
                    "magnetic_field_strength": 3.0
                },
                "clinical_notes": "輕度認知功能障礙，建議追蹤",
                "created_at": "2025-01-30T09:00:00Z",
                "updated_at": "2025-01-30T09:00:00Z"
            }
        }


class PatientResponse(BaseModel):
    """患者操作響應模型"""
    patient: Patient
    message: str
    success: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "name": "王小明",
                    "age": 65,
                    "gender": "M",
                    "diagnosis": "AD"
                },
                "message": "患者記錄創建成功",
                "success": True
            }
        }