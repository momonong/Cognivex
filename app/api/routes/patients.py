"""
患者管理 API 路由
處理患者資料的 CRUD 操作
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import List, Optional
from uuid import uuid4
from datetime import datetime

from app.api.models.patient import (
    Patient, PatientCreate, PatientUpdate, PatientResponse,
    HospitalInfo
)

router = APIRouter()

# 臨時存儲 - 後續會替換為資料庫
patients_db: dict = {}


@router.post("/", response_model=PatientResponse)
async def create_patient(patient: PatientCreate):
    """創建新患者記錄"""
    patient_id = str(uuid4())
    
    new_patient = Patient(
        id=patient_id,
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
        diagnosis=patient.diagnosis,
        scan_date=patient.scan_date,
        hospital_info=patient.hospital_info,
        clinical_notes=patient.clinical_notes,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    patients_db[patient_id] = new_patient
    
    return PatientResponse(
        patient=new_patient,
        message="患者記錄創建成功",
        success=True
    )


@router.get("/", response_model=List[Patient])
async def get_patients(
    skip: int = 0,
    limit: int = 100,
    diagnosis: Optional[str] = None
):
    """獲取患者列表"""
    patients = list(patients_db.values())
    
    # 按診斷篩選
    if diagnosis:
        patients = [p for p in patients if p.diagnosis == diagnosis]
    
    # 分頁
    return patients[skip:skip + limit]


@router.get("/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    """獲取特定患者詳情"""
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="患者記錄未找到")
    
    return patients_db[patient_id]


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(patient_id: str, patient_update: PatientUpdate):
    """更新患者資訊"""
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="患者記錄未找到")
    
    existing_patient = patients_db[patient_id]
    
    # 更新欄位
    update_data = patient_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(existing_patient, field, value)
    
    existing_patient.updated_at = datetime.now()
    patients_db[patient_id] = existing_patient
    
    return PatientResponse(
        patient=existing_patient,
        message="患者記錄更新成功",
        success=True
    )


@router.delete("/{patient_id}")
async def delete_patient(patient_id: str):
    """刪除患者記錄"""
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="患者記錄未找到")
    
    del patients_db[patient_id]
    
    return {
        "message": "患者記錄刪除成功",
        "success": True
    }


@router.get("/{patient_id}/summary")
async def get_patient_summary(patient_id: str):
    """獲取患者摘要資訊"""
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="患者記錄未找到")
    
    patient = patients_db[patient_id]
    
    return {
        "patient_id": patient.id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "diagnosis": patient.diagnosis,
        "scan_date": patient.scan_date,
        "hospital": patient.hospital_info.institution_name if patient.hospital_info else None,
        "files_count": 0,  # 後續整合檔案管理時更新
        "analyses_count": 0,  # 後續整合分析服務時更新
        "last_updated": patient.updated_at
    }