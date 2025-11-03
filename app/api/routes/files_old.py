"""
檔案管理 API 路由
處理 DICOM/NIfTI 檔案上傳和管理
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import os
import shutil
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from app.api.models.file import (
    PatientFile, FileUploadResponse, FileMetadata
)
from app.api.services.file_processor import file_processor

router = APIRouter()

# 檔案存儲配置
STORAGE_BASE = "storage/patients"
ALLOWED_EXTENSIONS = {".nii", ".nii.gz", ".dcm", ".json"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# 臨時存儲 - 後續會替換為資料庫
files_db: dict = {}


def validate_file_extension(filename: str) -> bool:
    """驗證檔案副檔名"""
    return file_processor.validate_file_format(filename)


def get_file_type(filename: str) -> str:
    """根據檔案名判斷檔案類型"""
    return file_processor.get_file_type(filename)


@router.post("/upload/{patient_id}", response_model=FileUploadResponse)
async def upload_patient_files(
    patient_id: str,
    files: List[UploadFile] = File(...),
    description: Optional[str] = Form(None)
):
    """上傳患者檔案"""
    
    if not files:
        raise HTTPException(status_code=400, detail="沒有選擇檔案")
    
    uploaded_files = []
    errors = []
    
    # 創建患者目錄
    patient_dir = Path(STORAGE_BASE) / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        try:
            # 驗證檔案
            if not validate_file_extension(file.filename):
                errors.append(f"不支援的檔案格式: {file.filename}")
                continue
            
            if file.size > MAX_FILE_SIZE:
                errors.append(f"檔案過大: {file.filename} ({file.size} bytes)")
                continue
            
            # 生成唯一檔案名
            file_id = str(uuid4())
            file_extension = Path(file.filename).suffix
            stored_filename = f"{file_id}{file_extension}"
            file_path = patient_dir / "raw" / stored_filename
            
            # 創建目錄
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存檔案
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 提取檔案元數據
            file_type = get_file_type(file.filename)
            metadata = file_processor.extract_metadata(str(file_path), file_type)
            
            # 創建檔案記錄
            patient_file = PatientFile(
                id=file_id,
                patient_id=patient_id,
                original_filename=file.filename,
                stored_filename=stored_filename,
                file_type=file_type,
                file_path=str(file_path),
                file_size=file.size,
                description=description,
                metadata=metadata,
                uploaded_at=datetime.now()
            )
            
            files_db[file_id] = patient_file
            uploaded_files.append(patient_file)
            
        except Exception as e:
            errors.append(f"上傳失敗 {file.filename}: {str(e)}")
    
    return FileUploadResponse(
        uploaded_files=uploaded_files,
        errors=errors,
        success=len(uploaded_files) > 0,
        message=f"成功上傳 {len(uploaded_files)} 個檔案"
    )


@router.get("/patient/{patient_id}", response_model=List[PatientFile])
async def get_patient_files(patient_id: str):
    """獲取患者的所有檔案"""
    patient_files = [
        file for file in files_db.values() 
        if file.patient_id == patient_id
    ]
    return patient_files


@router.get("/{file_id}", response_model=PatientFile)
async def get_file_info(file_id: str):
    """獲取檔案詳細資訊"""
    if file_id not in files_db:
        raise HTTPException(status_code=404, detail="檔案未找到")
    
    return files_db[file_id]


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """刪除檔案"""
    if file_id not in files_db:
        raise HTTPException(status_code=404, detail="檔案未找到")
    
    file_record = files_db[file_id]
    
    # 刪除實際檔案
    try:
        if os.path.exists(file_record.file_path):
            os.remove(file_record.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除檔案失敗: {str(e)}")
    
    # 刪除記錄
    del files_db[file_id]
    
    return {
        "message": "檔案刪除成功",
        "success": True
    }


@router.post("/{file_id}/validate")
async def validate_file(file_id: str):
    """驗證檔案完整性和格式"""
    if file_id not in files_db:
        raise HTTPException(status_code=404, detail="檔案未找到")
    
    file_record = files_db[file_id]
    
    validation_result = {
        "file_id": file_id,
        "is_valid": True,
        "checks": [],
        "warnings": [],
        "errors": []
    }
    
    # 檢查檔案是否存在
    if not os.path.exists(file_record.file_path):
        validation_result["is_valid"] = False
        validation_result["errors"].append("檔案不存在")
        return validation_result
    
    # 檢查檔案大小
    actual_size = os.path.getsize(file_record.file_path)
    if actual_size != file_record.file_size:
        validation_result["warnings"].append(
            f"檔案大小不符: 預期 {file_record.file_size}, 實際 {actual_size}"
        )
    
    validation_result["checks"].append("檔案存在性檢查")
    validation_result["checks"].append("檔案大小檢查")
    
    # 根據檔案類型進行特定驗證
    if file_record.file_type == "nifti":
        nifti_validation = file_processor.validate_nifti_file(file_record.file_path)
        validation_result.update(nifti_validation)
    elif file_record.file_type == "dicom":
        validation_result["checks"].append("DICOM 格式檢查")
        validation_result["warnings"].append("DICOM 詳細驗證待實作")
    elif file_record.file_type == "metadata":
        validation_result["checks"].append("JSON 格式檢查")
        try:
            import json
            with open(file_record.file_path, 'r') as f:
                json.load(f)
            validation_result["checks"].append("JSON 格式有效")
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"JSON 格式錯誤: {str(e)}")
    
    return validation_result

@
router.post("/{file_id}/convert-to-nifti")
async def convert_dicom_to_nifti(file_id: str):
    """將 DICOM 檔案轉換為 NIfTI 格式"""
    if file_id not in files_db:
        raise HTTPException(status_code=404, detail="檔案未找到")
    
    file_record = files_db[file_id]
    
    if file_record.file_type != "dicom":
        raise HTTPException(status_code=400, detail="只能轉換 DICOM 檔案")
    
    # 生成輸出路徑
    input_path = file_record.file_path
    output_dir = Path(input_path).parent.parent / "processed"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{Path(input_path).stem}.nii.gz"
    
    # 執行轉換
    conversion_result = file_processor.convert_dicom_to_nifti(
        str(input_path), 
        str(output_path)
    )
    
    if conversion_result["success"]:
        # 創建新的 NIfTI 檔案記錄
        nifti_file_id = str(uuid4())
        nifti_file = PatientFile(
            id=nifti_file_id,
            patient_id=file_record.patient_id,
            original_filename=f"{Path(file_record.original_filename).stem}.nii.gz",
            stored_filename=Path(conversion_result["output_path"]).name,
            file_type="nifti",
            file_path=conversion_result["output_path"],
            file_size=os.path.getsize(conversion_result["output_path"]),
            description=f"從 DICOM 轉換: {file_record.original_filename}",
            uploaded_at=datetime.now()
        )
        
        # 提取 NIfTI 元數據
        nifti_metadata = file_processor.extract_metadata(
            conversion_result["output_path"], 
            "nifti"
        )
        nifti_file.metadata = nifti_metadata
        
        files_db[nifti_file_id] = nifti_file
        
        return {
            "success": True,
            "message": "DICOM 轉換為 NIfTI 成功",
            "original_file_id": file_id,
            "converted_file_id": nifti_file_id,
            "converted_file": nifti_file,
            "conversion_details": conversion_result
        }
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"轉換失敗: {conversion_result['message']}"
        )


@router.get("/{file_id}/metadata")
async def get_file_metadata(file_id: str):
    """獲取檔案詳細元數據"""
    if file_id not in files_db:
        raise HTTPException(status_code=404, detail="檔案未找到")
    
    file_record = files_db[file_id]
    
    # 重新提取最新元數據
    fresh_metadata = file_processor.extract_metadata(
        file_record.file_path, 
        file_record.file_type
    )
    
    # 獲取預處理建議
    recommendations = file_processor.get_preprocessing_recommendations(fresh_metadata)
    
    return {
        "file_id": file_id,
        "file_info": {
            "original_filename": file_record.original_filename,
            "file_type": file_record.file_type,
            "file_size": file_record.file_size,
            "uploaded_at": file_record.uploaded_at
        },
        "metadata": fresh_metadata,
        "preprocessing_recommendations": recommendations
    }