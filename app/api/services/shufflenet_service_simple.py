"""
簡化版 ShuffleNet AI 分析服務
不依賴 LangGraph，直接使用 ShuffleNet 模型
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class SimpleShuffleNetService:
    """簡化版 ShuffleNet 分析服務類"""
    
    def __init__(self):
        self.model_path = "model/shufflenet/fold_3_best_model.pth"
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化 ShuffleNet 模型"""
        try:
            # 檢查模型文件是否存在
            if not os.path.exists(self.model_path):
                logger.warning(f"ShuffleNet 模型文件不存在: {self.model_path}")
                return
            
            # 嘗試載入模型類別
            try:
                from model.shufflenet.model import PaperModel
                self.model = PaperModel(num_classes=2)
                
                # 載入預訓練權重
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                self.model.eval()
                
                logger.info("✅ ShuffleNet 模型載入成功")
                
            except Exception as e:
                logger.warning(f"⚠️ ShuffleNet 模型載入失敗: {e}")
                self.model = None
            
        except Exception as e:
            logger.error(f"❌ ShuffleNet 模型初始化失敗: {e}")
    
    def validate_input_file(self, file_path: str) -> Dict[str, Any]:
        """驗證輸入檔案"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {}
        }
        
        try:
            # 檢查檔案是否存在
            if not os.path.exists(file_path):
                validation_result["is_valid"] = False
                validation_result["errors"].append("檔案不存在")
                return validation_result
            
            # 檢查檔案格式
            if not file_path.lower().endswith(('.nii', '.nii.gz')):
                validation_result["is_valid"] = False
                validation_result["errors"].append("不支援的檔案格式，需要 NIfTI 格式")
                return validation_result
            
            # 檢查檔案大小
            file_size = os.path.getsize(file_path)
            validation_result["file_info"]["size_mb"] = round(file_size / (1024 * 1024), 2)
            
            if file_size == 0:
                validation_result["is_valid"] = False
                validation_result["errors"].append("檔案大小為 0")
                return validation_result
            
            # 嘗試載入 NIfTI 檔案進行基本驗證
            try:
                import nibabel as nib
                img = nib.load(file_path)
                shape = img.shape
                
                validation_result["file_info"]["dimensions"] = list(shape)
                validation_result["file_info"]["data_type"] = str(img.get_data_dtype())
                
                # 檢查維度
                if len(shape) < 3:
                    validation_result["warnings"].append("影像維度可能不正確")
                elif len(shape) == 4:
                    validation_result["file_info"]["is_4d"] = True
                    validation_result["file_info"]["time_points"] = shape[3]
                else:
                    validation_result["file_info"]["is_4d"] = False
                
            except Exception as e:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"NIfTI 檔案載入失敗: {str(e)}")
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"檔案驗證失敗: {str(e)}")
        
        return validation_result
    
    def preprocess_nifti_data(self, file_path: str) -> torch.Tensor:
        """預處理 NIfTI 數據為 ShuffleNet 輸入格式"""
        try:
            import nibabel as nib
            import cv2
            
            # 載入 NIfTI 數據
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # 處理 4D fMRI 數據
            if len(data.shape) == 4:
                # 取時間平均
                data = np.mean(data, axis=3)
            
            # 提取矢狀面切片 (10 張中央切片)
            sagittal_dim = 0
            num_total_slices = data.shape[sagittal_dim]
            
            if num_total_slices < 10:
                raise ValueError(f"矢狀面切片數不足: {num_total_slices} < 10")
            
            # 找到中央 10 張切片
            center_slice_index = num_total_slices // 2
            start_index = center_slice_index - 5
            end_index = start_index + 10
            
            selected_slices_data = data[start_index:end_index, :, :]
            
            # 處理每張切片
            processed_slices = []
            for i in range(10):
                slice_2d = selected_slices_data[i, :, :]
                
                # 旋轉影像
                slice_2d = np.rot90(slice_2d)
                
                # 標準化到 0-255
                if np.max(slice_2d) > 0:
                    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))
                slice_2d_uint8 = (slice_2d * 255).astype(np.uint8)
                
                # 縮放到 128x128
                resized_slice = cv2.resize(slice_2d_uint8, (128, 128), interpolation=cv2.INTER_CUBIC)
                processed_slices.append(resized_slice)
            
            # 堆疊並轉換為 tensor
            stacked_slices = np.stack(processed_slices)  # (10, 128, 128)
            stacked_slices = stacked_slices[:, np.newaxis, :, :]  # (10, 1, 128, 128)
            
            # 轉換為 tensor 並標準化到 0-1
            slices_tensor = torch.tensor(stacked_slices, dtype=torch.float32) / 255.0
            
            # 添加 batch 維度: (1, 10, 1, 128, 128)
            input_tensor = slices_tensor.unsqueeze(0)
            
            return input_tensor
            
        except Exception as e:
            raise ValueError(f"數據預處理失敗: {e}")
    
    async def run_analysis(
        self, 
        patient_id: str, 
        fmri_file_path: str,
        analysis_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """執行 ShuffleNet 分析"""
        
        analysis_result = {
            "success": False,
            "patient_id": patient_id,
            "model_name": "shufflenet",
            "prediction": None,
            "confidence": 0.0,
            "processing_time": 0.0,
            "visualization_paths": [],
            "generated_reports": {},
            "brain_regions": [],
            "functional_networks": [],
            "quality_metrics": {},
            "error": None,
            "trace_log": []
        }
        
        start_time = datetime.now()
        
        try:
            # 1. 驗證輸入檔案
            logger.info(f"🔍 驗證輸入檔案: {fmri_file_path}")
            validation = self.validate_input_file(fmri_file_path)
            
            if not validation["is_valid"]:
                analysis_result["error"] = f"檔案驗證失敗: {', '.join(validation['errors'])}"
                return analysis_result
            
            analysis_result["trace_log"].append("檔案驗證通過")
            
            # 2. 檢查模型是否可用
            if self.model is None:
                analysis_result["error"] = "ShuffleNet 模型未載入"
                return analysis_result
            
            analysis_result["trace_log"].append("模型檢查通過")
            
            # 3. 預處理數據
            logger.info(f"🔄 預處理 fMRI 數據")
            input_tensor = self.preprocess_nifti_data(fmri_file_path)
            input_tensor = input_tensor.to(self.device)
            
            analysis_result["trace_log"].append("數據預處理完成")
            
            # 4. 執行推理
            logger.info(f"🧠 執行 ShuffleNet 推理")
            
            with torch.no_grad():
                logits, embeddings = self.model(input_tensor)
                
                # 計算預測結果
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
                
                # 轉換為類別名稱
                prediction = "AD" if predicted_class == 1 else "NC"
            
            analysis_result["prediction"] = prediction
            analysis_result["confidence"] = confidence
            analysis_result["success"] = True
            
            analysis_result["trace_log"].append(f"推理完成: {prediction} (信心度: {confidence:.3f})")
            
            # 5. 生成簡單報告
            analysis_result["generated_reports"] = {
                "zh": f"基於 ShuffleNet 模型分析，預測結果為 {prediction}，信心分數為 {confidence:.3f}。",
                "en": f"Based on ShuffleNet analysis, prediction is {prediction} with confidence {confidence:.3f}."
            }
            
            logger.info(f"✅ 分析完成: {prediction} (信心度: {confidence:.3f})")
            
        except Exception as e:
            logger.error(f"❌ ShuffleNet 分析失敗: {e}")
            analysis_result["error"] = str(e)
            analysis_result["trace_log"].append(f"分析失敗: {str(e)}")
        
        # 計算處理時間
        end_time = datetime.now()
        analysis_result["processing_time"] = (end_time - start_time).total_seconds()
        
        return analysis_result
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取 ShuffleNet 模型資訊"""
        return {
            "model_name": "ShuffleNet 2D with ECA Attention",
            "model_type": "2D CNN",
            "model_path": self.model_path,
            "model_exists": os.path.exists(self.model_path),
            "model_loaded": self.model is not None,
            "description": "高準確度 2D CNN，具有注意力機制，用於基於切片的 fMRI 分析",
            "accuracy": "80%+",
            "input_format": "2D brain slices (10 sagittal slices)",
            "output_classes": ["AD", "NC"],
            "preprocessing": {
                "slice_selection": "10 consecutive sagittal slices from center",
                "slice_size": "128x128 pixels",
                "normalization": "0-255 to 0-1",
                "augmentation": False
            },
            "architecture": {
                "backbone": "ShuffleNet V1 with modifications (2,4,2)",
                "attention": "ECA (Efficient Channel Attention)",
                "groups": 3,
                "dropout": 0.2
            },
            "device": str(self.device)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """執行模型健康檢查"""
        health_status = {
            "status": "healthy",
            "checks": [],
            "issues": [],
            "model_info": {}
        }
        
        try:
            # 檢查模型文件
            if os.path.exists(self.model_path):
                health_status["checks"].append("✅ 模型文件存在")
                
                # 檢查文件大小
                file_size = os.path.getsize(self.model_path)
                health_status["model_info"]["file_size_mb"] = round(file_size / (1024 * 1024), 2)
                
                if file_size > 0:
                    health_status["checks"].append("✅ 模型文件大小正常")
                else:
                    health_status["issues"].append("模型文件大小為 0")
                    health_status["status"] = "unhealthy"
            else:
                health_status["checks"].append("❌ 模型文件不存在")
                health_status["issues"].append("模型文件缺失")
                health_status["status"] = "unhealthy"
            
            # 檢查模型載入狀態
            if self.model is not None:
                health_status["checks"].append("✅ 模型載入成功")
            else:
                health_status["checks"].append("❌ 模型未載入")
                health_status["issues"].append("模型載入失敗")
                health_status["status"] = "unhealthy"
            
            # 檢查 PyTorch 和設備
            health_status["model_info"]["pytorch_version"] = torch.__version__
            health_status["model_info"]["device"] = str(self.device)
            health_status["model_info"]["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                health_status["checks"].append("✅ CUDA 可用")
                health_status["model_info"]["cuda_device_count"] = torch.cuda.device_count()
                health_status["model_info"]["cuda_device_name"] = torch.cuda.get_device_name()
            else:
                health_status["checks"].append("⚠️ CUDA 不可用，使用 CPU")
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["issues"].append(f"健康檢查失敗: {str(e)}")
        
        return health_status


# 全域簡化版 ShuffleNet 服務實例
simple_shufflenet_service = SimpleShuffleNetService()