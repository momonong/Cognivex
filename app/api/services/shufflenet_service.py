"""
ShuffleNet AI 分析服務
整合 ShuffleNet 模型進行 fMRI 分析
"""

import os
import logging
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import time
from datetime import datetime

# 導入 ShuffleNet 模型
from model.shufflenet.model import PaperModel, preprocess_nii_to_slices

logger = logging.getLogger(__name__)


class ShuffleNetService:
    """ShuffleNet 分析服務類"""
    
    def __init__(self, model_path: str = "model/shufflenet/fold_3_best_model.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_loaded = False
        
        # 模型配置
        self.num_classes = 2
        self.class_names = ["NC", "AD"]  # 0: NC (Normal Control), 1: AD (Alzheimer's Disease)
        
        logger.info(f"ShuffleNet 服務初始化，使用設備: {self.device}")
    
    def load_model(self) -> bool:
        """載入 ShuffleNet 模型"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"模型檔案不存在: {self.model_path}")
                return False
            
            # 創建模型實例
            self.model = PaperModel(num_classes=self.num_classes)
            
            # 載入權重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 處理不同的權重格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info("ShuffleNet 模型載入成功")
            return True
            
        except Exception as e:
            logger.error(f"載入 ShuffleNet 模型失敗: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_fmri(self, nii_path: str) -> Optional[torch.Tensor]:
        """預處理 fMRI 數據"""
        try:
            logger.info(f"開始預處理 fMRI 數據: {nii_path}")
            
            # 使用 ShuffleNet 模型的預處理函數
            slices_array = preprocess_nii_to_slices(nii_path)
            
            if slices_array is None:
                logger.error("fMRI 預處理失敗")
                return None
            
            # 轉換為 PyTorch tensor
            slices_tensor = torch.tensor(slices_array, dtype=torch.float32) / 255.0
            
            # 添加 batch 維度: (10, 1, 128, 128) -> (1, 10, 1, 128, 128)
            slices_tensor = slices_tensor.unsqueeze(0)
            
            logger.info(f"fMRI 預處理完成，張量形狀: {slices_tensor.shape}")
            return slices_tensor
            
        except Exception as e:
            logger.error(f"fMRI 預處理錯誤: {e}")
            return None
    
    def run_inference(self, preprocessed_data: torch.Tensor) -> Dict[str, Any]:
        """執行 ShuffleNet 推理"""
        try:
            if not self.is_loaded:
                if not self.load_model():
                    return {"error": "模型載入失敗"}
            
            start_time = time.time()
            
            # 移動數據到設備
            preprocessed_data = preprocessed_data.to(self.device)
            
            # 執行推理
            with torch.no_grad():
                logits, embeddings = self.model(preprocessed_data)
                
                # 獲取預測結果
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                # 獲取各類別的機率
                class_probabilities = {
                    self.class_names[i]: probabilities[0, i].item() 
                    for i in range(self.num_classes)
                }
            
            processing_time = time.time() - start_time
            
            result = {
                "prediction": self.class_names[predicted_class],
                "predicted_class_index": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "processing_time": processing_time,
                "embeddings": embeddings.cpu().numpy().tolist(),
                "logits": logits.cpu().numpy().tolist()
            }
            
            logger.info(f"推理完成: {result['prediction']} (信心度: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"ShuffleNet 推理錯誤: {e}")
            return {"error": str(e)}
    
    def analyze_fmri(self, nii_path: str, patient_id: str = None) -> Dict[str, Any]:
        """完整的 fMRI 分析流程"""
        try:
            analysis_start_time = time.time()
            
            logger.info(f"開始 ShuffleNet fMRI 分析: {nii_path}")
            
            # 1. 預處理
            preprocessed_data = self.preprocess_fmri(nii_path)
            if preprocessed_data is None:
                return {"error": "fMRI 預處理失敗"}
            
            # 2. 推理
            inference_result = self.run_inference(preprocessed_data)
            if "error" in inference_result:
                return inference_result
            
            # 3. 生成分析報告
            total_time = time.time() - analysis_start_time
            
            analysis_result = {
                "patient_id": patient_id,
                "model_name": "ShuffleNet 2D with ECA Attention",
                "analysis_timestamp": datetime.now().isoformat(),
                "input_file": nii_path,
                "prediction": inference_result["prediction"],
                "confidence": inference_result["confidence"],
                "class_probabilities": inference_result["class_probabilities"],
                "processing_time": total_time,
                "model_details": {
                    "architecture": "ShuffleNet v1 + ECA Attention",
                    "input_slices": 10,
                    "slice_size": "128x128",
                    "classes": self.class_names
                },
                "quality_metrics": self._calculate_quality_metrics(preprocessed_data),
                "success": True
            }
            
            logger.info(f"ShuffleNet 分析完成: {analysis_result['prediction']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"ShuffleNet 分析錯誤: {e}")
            return {
                "error": str(e),
                "success": False,
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    def _calculate_quality_metrics(self, preprocessed_data: torch.Tensor) -> Dict[str, float]:
        """計算影像品質指標"""
        try:
            # 移除 batch 維度進行計算
            data = preprocessed_data.squeeze(0).cpu().numpy()  # (10, 1, 128, 128)
            
            # 計算基本統計指標
            mean_intensity = float(np.mean(data))
            std_intensity = float(np.std(data))
            snr = mean_intensity / std_intensity if std_intensity > 0 else 0.0
            
            # 計算對比度
            contrast = float(np.std(data))
            
            # 計算切片間一致性
            slice_correlations = []
            for i in range(data.shape[0] - 1):
                slice1 = data[i, 0].flatten()
                slice2 = data[i + 1, 0].flatten()
                corr = np.corrcoef(slice1, slice2)[0, 1]
                if not np.isnan(corr):
                    slice_correlations.append(corr)
            
            temporal_consistency = float(np.mean(slice_correlations)) if slice_correlations else 0.0
            
            # 整體品質分數 (簡化計算)
            quality_score = min(1.0, (snr * 0.4 + contrast * 0.3 + temporal_consistency * 0.3))
            
            return {
                "overall_quality": quality_score,
                "signal_noise_ratio": snr,
                "contrast": contrast,
                "temporal_consistency": temporal_consistency,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity
            }
            
        except Exception as e:
            logger.warning(f"品質指標計算失敗: {e}")
            return {
                "overall_quality": 0.5,
                "signal_noise_ratio": 0.0,
                "contrast": 0.0,
                "temporal_consistency": 0.0,
                "mean_intensity": 0.0,
                "std_intensity": 0.0
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型資訊"""
        return {
            "model_name": "ShuffleNet 2D with ECA Attention",
            "model_path": self.model_path,
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "architecture_details": {
                "backbone": "ShuffleNet v1",
                "attention": "ECA (Efficient Channel Attention)",
                "input_format": "10 consecutive sagittal slices",
                "slice_size": "128x128",
                "preprocessing": "Rotation, normalization, resizing"
            },
            "performance": {
                "accuracy": "80%+",
                "training_data": "ADNI fMRI dataset",
                "validation": "5-fold cross-validation"
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """模型健康檢查"""
        health_status = {
            "status": "healthy",
            "checks": [],
            "warnings": [],
            "errors": []
        }
        
        try:
            # 檢查模型檔案
            if os.path.exists(self.model_path):
                health_status["checks"].append("✅ 模型檔案存在")
                file_size = os.path.getsize(self.model_path)
                health_status["model_file_size"] = file_size
            else:
                health_status["errors"].append("❌ 模型檔案不存在")
                health_status["status"] = "unhealthy"
            
            # 檢查設備可用性
            if torch.cuda.is_available():
                health_status["checks"].append("✅ CUDA 可用")
                health_status["gpu_info"] = {
                    "device_name": torch.cuda.get_device_name(),
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_reserved": torch.cuda.memory_reserved()
                }
            else:
                health_status["warnings"].append("⚠️ CUDA 不可用，使用 CPU")
            
            # 檢查模型載入
            if self.is_loaded:
                health_status["checks"].append("✅ 模型已載入")
            else:
                # 嘗試載入模型
                if self.load_model():
                    health_status["checks"].append("✅ 模型載入成功")
                else:
                    health_status["errors"].append("❌ 模型載入失敗")
                    health_status["status"] = "unhealthy"
            
            # 簡單推理測試
            if self.is_loaded:
                test_input = torch.randn(1, 10, 1, 128, 128).to(self.device)
                with torch.no_grad():
                    logits, embeddings = self.model(test_input)
                    health_status["checks"].append("✅ 推理測試通過")
            
        except Exception as e:
            health_status["errors"].append(f"❌ 健康檢查錯誤: {str(e)}")
            health_status["status"] = "unhealthy"
        
        return health_status


# 全域 ShuffleNet 服務實例
shufflenet_service = ShuffleNetService()