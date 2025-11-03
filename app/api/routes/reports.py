"""
報告生成 API 路由
處理臨床報告的生成和匯出
"""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from typing import Optional, Dict, Any
import json
import os
from datetime import datetime
from pathlib import Path

router = APIRouter()

# 報告存儲目錄
REPORTS_DIR = "storage/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# 臨時存儲 - 後續會替換為資料庫
reports_db: dict = {}


@router.post("/generate/{analysis_id}")
async def generate_report(
    analysis_id: str,
    format: str = "json",
    language: str = "zh"
):
    """基於分析結果生成臨床報告"""
    
    # 這裡需要從分析服務獲取結果
    # 暫時使用模擬數據
    
    if format not in ["json", "pdf"]:
        raise HTTPException(status_code=400, detail="不支援的報告格式")
    
    if language not in ["zh", "en"]:
        raise HTTPException(status_code=400, detail="不支援的語言")
    
    # 模擬報告數據
    report_data = {
        "report_id": f"report_{analysis_id}",
        "analysis_id": analysis_id,
        "generated_at": datetime.now().isoformat(),
        "language": language,
        "format": format,
        "patient_info": {
            "patient_id": "patient_123",
            "name": "測試患者",
            "age": 65,
            "gender": "M",
            "scan_date": "2025-01-30"
        },
        "analysis_results": {
            "model_used": "ShuffleNet 2D with ECA Attention",
            "prediction": "AD",
            "confidence": 0.85,
            "processing_time": 45.2
        },
        "brain_regions": [
            {
                "region": "海馬迴",
                "aal3_label": "Hippocampus_L",
                "activation_score": 0.72,
                "yeo_network": "Default Mode Network",
                "clinical_significance": "與記憶功能相關的重要區域"
            }
        ],
        "clinical_interpretation": {
            "zh": "基於 ShuffleNet 模型分析，患者腦部影像顯示阿茲海默症的特徵性改變。海馬迴區域活化異常，與認知功能下降一致。建議進一步臨床評估。",
            "en": "Based on ShuffleNet model analysis, the patient's brain imaging shows characteristic changes of Alzheimer's disease. Abnormal hippocampal activation is consistent with cognitive decline. Further clinical evaluation is recommended."
        },
        "recommendations": {
            "zh": [
                "建議進行詳細的神經心理學評估",
                "考慮進行 PET 掃描以確認澱粉樣蛋白沉積",
                "定期追蹤認知功能變化",
                "評估藥物治療選項"
            ],
            "en": [
                "Detailed neuropsychological assessment recommended",
                "Consider PET scan to confirm amyloid deposition",
                "Regular monitoring of cognitive function changes",
                "Evaluate medication treatment options"
            ]
        }
    }
    
    # 生成報告檔案
    report_filename = f"report_{analysis_id}_{language}.{format}"
    report_path = Path(REPORTS_DIR) / report_filename
    
    if format == "json":
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    elif format == "pdf":
        # 這裡應該整合 PDF 生成庫 (如 reportlab)
        # 暫時創建一個文本檔案作為佔位符
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("PDF 報告生成功能待實作\n")
            f.write(f"分析 ID: {analysis_id}\n")
            f.write(f"語言: {language}\n")
            f.write(json.dumps(report_data, ensure_ascii=False, indent=2))
    
    # 保存報告記錄
    report_record = {
        "report_id": report_data["report_id"],
        "analysis_id": analysis_id,
        "format": format,
        "language": language,
        "file_path": str(report_path),
        "generated_at": datetime.now(),
        "data": report_data
    }
    
    reports_db[report_data["report_id"]] = report_record
    
    return {
        "report_id": report_data["report_id"],
        "message": "報告生成成功",
        "format": format,
        "language": language,
        "file_path": str(report_path),
        "download_url": f"/api/reports/{report_data['report_id']}/download"
    }


@router.get("/{report_id}")
async def get_report_info(report_id: str):
    """獲取報告資訊"""
    if report_id not in reports_db:
        raise HTTPException(status_code=404, detail="報告未找到")
    
    report = reports_db[report_id]
    
    return {
        "report_id": report["report_id"],
        "analysis_id": report["analysis_id"],
        "format": report["format"],
        "language": report["language"],
        "generated_at": report["generated_at"],
        "file_exists": os.path.exists(report["file_path"]),
        "file_size": os.path.getsize(report["file_path"]) if os.path.exists(report["file_path"]) else 0
    }


@router.get("/{report_id}/content")
async def get_report_content(report_id: str):
    """獲取報告內容 (JSON 格式)"""
    if report_id not in reports_db:
        raise HTTPException(status_code=404, detail="報告未找到")
    
    report = reports_db[report_id]
    return report["data"]


@router.get("/{report_id}/download")
async def download_report(report_id: str):
    """下載報告檔案"""
    if report_id not in reports_db:
        raise HTTPException(status_code=404, detail="報告未找到")
    
    report = reports_db[report_id]
    file_path = report["file_path"]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="報告檔案不存在")
    
    filename = f"clinical_report_{report_id}.{report['format']}"
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@router.get("/analysis/{analysis_id}")
async def get_analysis_reports(analysis_id: str):
    """獲取特定分析的所有報告"""
    analysis_reports = [
        {
            "report_id": report["report_id"],
            "format": report["format"],
            "language": report["language"],
            "generated_at": report["generated_at"]
        }
        for report in reports_db.values()
        if report["analysis_id"] == analysis_id
    ]
    
    return {
        "analysis_id": analysis_id,
        "reports": analysis_reports,
        "total_count": len(analysis_reports)
    }


@router.delete("/{report_id}")
async def delete_report(report_id: str):
    """刪除報告"""
    if report_id not in reports_db:
        raise HTTPException(status_code=404, detail="報告未找到")
    
    report = reports_db[report_id]
    
    # 刪除檔案
    try:
        if os.path.exists(report["file_path"]):
            os.remove(report["file_path"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除報告檔案失敗: {str(e)}")
    
    # 刪除記錄
    del reports_db[report_id]
    
    return {
        "message": "報告刪除成功",
        "success": True
    }