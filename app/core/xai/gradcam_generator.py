# app/core/xai/gradcam_generator.py

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from monai.visualize import GradCAM
import scipy.ndimage


class GradCAMGenerator:
    """
    生成 Grad-CAM 熱圖的類別，支援單一模型和集成模型
    
    Attributes:
        models: 模型列表 (用於集成)
        device: 計算裝置 (cuda 或 cpu)
        target_layer_name: 目標層名稱 (預設 'block4')
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: torch.device,
        target_layer_name: str = "block4"
    ):
        """
        初始化 GradCAMGenerator
        
        Args:
            models: 模型列表，用於集成分析
            device: PyTorch 裝置 (cuda:0 或 cpu)
            target_layer_name: 要分析的目標層名稱
        """
        self.models = models
        self.device = device
        self.target_layer_name = target_layer_name
        
        # 驗證模型
        if not models:
            raise ValueError("模型列表不能為空")
        
        # 確保所有模型都在正確的裝置上並處於評估模式
        for model in self.models:
            model.to(self.device)
            model.eval()
    
    def generate_single_model(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """
        為單一模型生成 Grad-CAM 熱圖
        
        Args:
            model: 3D CNN 模型
            input_tensor: 輸入張量，shape (1, 1, H, W, D)
            target_class: 目標類別索引 (0=NC, 1=AD)
            
        Returns:
            3D numpy array，shape (H, W, D)
        """
        try:
            # 取得目標層
            target_layer = self._get_target_layer(model)
            
            # 建立 GradCAM 物件
            gradcam = GradCAM(
                nn_module=model,
                target_layers=target_layer,
                device=self.device
            )
            
            # 計算 Grad-CAM
            with torch.set_grad_enabled(True):
                heatmap_tensor = gradcam(x=input_tensor, class_idx=target_class)
            
            # 轉換為 numpy
            heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
            
            return heatmap_np
            
        except Exception as e:
            raise RuntimeError(f"生成 Grad-CAM 失敗: {e}")
    
    def generate_ensemble(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        threshold_percentile: float = 95.0,
        aggregation_method: str = "mean"
    ) -> np.ndarray:
        """
        生成集成 Grad-CAM (平均多個模型的熱圖)
        
        Args:
            input_tensor: 輸入張量，shape (1, 1, H, W, D)
            target_class: 目標類別索引 (0=NC, 1=AD)
            threshold_percentile: 閾值百分位數 (0-100)，用於過濾低激活值
            aggregation_method: 聚合方法 ('mean', 'max', 'weighted')
            
        Returns:
            3D numpy array，shape (H, W, D)，已標準化和閾值處理
        """
        if not 0 <= threshold_percentile <= 100:
            raise ValueError("threshold_percentile 必須在 0-100 之間")
        
        # 收集所有模型的熱圖
        all_heatmaps = []
        
        for model in self.models:
            heatmap = self.generate_single_model(model, input_tensor, target_class)
            all_heatmaps.append(heatmap)
        
        # 聚合熱圖
        heatmap_stack = np.stack(all_heatmaps, axis=0)
        
        if aggregation_method == "mean":
            aggregated_heatmap = np.mean(heatmap_stack, axis=0)
        elif aggregation_method == "max":
            aggregated_heatmap = np.max(heatmap_stack, axis=0)
        elif aggregation_method == "weighted":
            # 加權平均 (可以根據模型準確度調整權重)
            # 目前使用均等權重
            weights = np.ones(len(self.models)) / len(self.models)
            aggregated_heatmap = np.average(heatmap_stack, axis=0, weights=weights)
        else:
            raise ValueError(f"不支援的聚合方法: {aggregation_method}")
        
        # 標準化到 [0, 1]
        normalized_heatmap = self._normalize_heatmap(aggregated_heatmap)
        
        # 應用閾值過濾
        filtered_heatmap = self._apply_threshold(
            normalized_heatmap,
            threshold_percentile
        )
        
        return filtered_heatmap
    
    def save_as_nifti(
        self,
        heatmap: np.ndarray,
        affine: np.ndarray,
        output_path: str,
        subject_id: Optional[str] = None,
        target_class: Optional[str] = None
    ) -> str:
        """
        將熱圖儲存為 NIfTI 格式
        
        Args:
            heatmap: 3D numpy array
            affine: 4x4 affine 矩陣
            output_path: 輸出目錄或完整檔案路徑
            subject_id: 受試者 ID (可選)
            target_class: 目標類別名稱 (可選，如 'AD' 或 'NC')
            
        Returns:
            儲存的檔案完整路徑
        """
        # 確保輸出目錄存在
        if os.path.isdir(output_path) or not output_path.endswith('.nii.gz'):
            # 如果是目錄，建立檔名
            os.makedirs(output_path, exist_ok=True)
            
            filename_parts = []
            if subject_id:
                filename_parts.append(subject_id)
            filename_parts.append("gradcam_ensemble")
            if target_class:
                filename_parts.append(target_class.upper())
            
            filename = "_".join(filename_parts) + ".nii.gz"
            full_path = os.path.join(output_path, filename)
        else:
            # 如果是完整路徑，確保目錄存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            full_path = output_path
        
        # 建立 NIfTI 影像
        nii_image = nib.Nifti1Image(heatmap.astype(np.float32), affine)
        
        # 儲存
        nib.save(nii_image, full_path)
        
        return full_path
    
    def upsample_to_original(
        self,
        heatmap: np.ndarray,
        target_shape: Tuple[int, int, int],
        order: int = 1
    ) -> np.ndarray:
        """
        將熱圖上採樣到原始影像解析度
        
        Args:
            heatmap: 3D numpy array
            target_shape: 目標形狀 (H, W, D)
            order: 插值順序 (0=最近鄰, 1=線性, 3=三次)
            
        Returns:
            上採樣後的 3D numpy array
        """
        current_shape = heatmap.shape
        zoom_factors = [
            t / c for t, c in zip(target_shape, current_shape)
        ]
        
        upsampled = scipy.ndimage.zoom(heatmap, zoom_factors, order=order)
        
        return upsampled
    
    def update_affine_after_upsample(
        self,
        affine: np.ndarray,
        zoom_factors: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        更新上採樣後的 affine 矩陣
        
        Args:
            affine: 原始 4x4 affine 矩陣
            zoom_factors: 縮放因子 (x, y, z)
            
        Returns:
            更新後的 affine 矩陣
        """
        new_affine = np.copy(affine)
        zoom_factors_4d = np.append(zoom_factors, 1)
        np.fill_diagonal(new_affine, new_affine.diagonal() / zoom_factors_4d)
        return new_affine
    
    def _get_target_layer(self, model: nn.Module) -> nn.Module:
        """
        取得模型的目標層
        
        Args:
            model: 3D CNN 模型
            
        Returns:
            目標層模組
        """
        try:
            # 嘗試取得指定的層
            layer = getattr(model, self.target_layer_name)
            
            # 如果是 Sequential，取第一個子模組
            if isinstance(layer, nn.Sequential):
                return layer[0]
            
            return layer
            
        except AttributeError:
            raise ValueError(
                f"模型中找不到層 '{self.target_layer_name}'"
            )
    
    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        將熱圖標準化到 [0, 1] 範圍
        
        Args:
            heatmap: 原始熱圖
            
        Returns:
            標準化後的熱圖
        """
        min_val = np.min(heatmap)
        max_val = np.max(heatmap)
        
        if max_val - min_val < 1e-8:
            # 避免除以零
            return np.zeros_like(heatmap)
        
        normalized = (heatmap - min_val) / (max_val - min_val)
        return normalized
    
    def _apply_threshold(
        self,
        heatmap: np.ndarray,
        percentile: float
    ) -> np.ndarray:
        """
        應用百分位數閾值過濾
        
        Args:
            heatmap: 標準化後的熱圖
            percentile: 百分位數閾值
            
        Returns:
            過濾後的熱圖
        """
        threshold = np.percentile(heatmap, percentile)
        filtered = np.where(heatmap >= threshold, heatmap, 0)
        return filtered
    
    def get_statistics(self, heatmap: np.ndarray) -> Dict[str, float]:
        """
        計算熱圖的統計資訊
        
        Args:
            heatmap: 3D numpy array
            
        Returns:
            包含統計資訊的字典
        """
        non_zero_values = heatmap[heatmap > 0]
        
        stats = {
            "min": float(np.min(heatmap)),
            "max": float(np.max(heatmap)),
            "mean": float(np.mean(heatmap)),
            "std": float(np.std(heatmap)),
            "non_zero_count": int(np.sum(heatmap > 0)),
            "total_voxels": int(heatmap.size),
            "non_zero_percentage": float(np.sum(heatmap > 0) / heatmap.size * 100)
        }
        
        if len(non_zero_values) > 0:
            stats["non_zero_mean"] = float(np.mean(non_zero_values))
            stats["non_zero_std"] = float(np.std(non_zero_values))
        else:
            stats["non_zero_mean"] = 0.0
            stats["non_zero_std"] = 0.0
        
        return stats
