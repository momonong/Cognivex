# app/core/xai/activation_extractor.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from datetime import datetime
import os


class ActivationExtractor:
    """
    從 3D CNN 模型的指定層擷取 activation 和 gradient。
    
    使用 PyTorch hooks 機制在前向和反向傳播時擷取中間層的資料。
    支援同時擷取多個層的 activation 和 gradient。
    
    Attributes:
        model: PyTorch 模型實例
        target_layers: 要擷取的層名稱或模組列表
        device: 計算裝置 (cuda 或 cpu)
        activations: 儲存擷取的 activation 資料
        gradients: 儲存擷取的 gradient 資料
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        target_layers: List[str],
        device: Optional[torch.device] = None
    ):
        """
        初始化 ActivationExtractor。
        
        Args:
            model: 3D CNN 模型實例
            target_layers: 要擷取的層名稱列表，如 ['block4', 'block3']
            device: 計算裝置，如果為 None 則自動偵測
        """
        self.model = model
        self.target_layers = target_layers
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 儲存擷取的資料
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        
        # 儲存 hook 句柄以便後續移除
        self.hook_handles: List[Any] = []
        
        # 將模型移到指定裝置
        self.model.to(self.device)
        self.model.eval()
    
    def _get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """
        根據層名稱取得模型中的對應模組。
        
        Args:
            layer_name: 層名稱，如 'block4' 或 'block4.0'
            
        Returns:
            對應的 nn.Module，如果找不到則返回 None
        """
        # 支援巢狀層名稱，如 'block4.0'
        parts = layer_name.split('.')
        module = self.model
        
        try:
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            print(f"⚠️ 警告: 找不到層 '{layer_name}'")
            return None
    
    def _forward_hook(self, layer_name: str):
        """
        建立 forward hook 函式以擷取 activation。
        
        Args:
            layer_name: 層名稱
            
        Returns:
            Hook 函式
        """
        def hook(module, input, output):
            # 儲存 activation (detach 以避免計算圖累積)
            self.activations[layer_name] = output.detach()
        return hook
    
    def _backward_hook(self, layer_name: str):
        """
        建立 backward hook 函式以擷取 gradient。
        
        Args:
            layer_name: 層名稱
            
        Returns:
            Hook 函式
        """
        def hook(module, grad_input, grad_output):
            # 儲存 gradient (detach 以避免計算圖累積)
            # grad_output 是對該層輸出的梯度
            self.gradients[layer_name] = grad_output[0].detach()
        return hook
    
    def register_hooks(self) -> None:
        """
        為所有目標層註冊 forward 和 backward hooks。
        
        這個方法會為每個指定的層註冊兩個 hooks：
        - Forward hook: 擷取 activation
        - Backward hook: 擷取 gradient
        """
        # 清除之前的 hooks
        self.remove_hooks()
        
        for layer_name in self.target_layers:
            layer = self._get_layer_by_name(layer_name)
            
            if layer is None:
                continue
            
            # 註冊 forward hook
            forward_handle = layer.register_forward_hook(
                self._forward_hook(layer_name)
            )
            self.hook_handles.append(forward_handle)
            
            # 註冊 backward hook
            backward_handle = layer.register_full_backward_hook(
                self._backward_hook(layer_name)
            )
            self.hook_handles.append(backward_handle)
        
        print(f"✅ 已為 {len(self.target_layers)} 個層註冊 hooks")
    
    def remove_hooks(self) -> None:
        """移除所有已註冊的 hooks。"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def extract(
        self, 
        input_tensor: torch.Tensor, 
        target_class: int,
        subject_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        執行前向和反向傳播，擷取 activations 和 gradients。
        
        Args:
            input_tensor: 輸入張量，shape 應為 (batch, channels, H, W, D)
            target_class: 目標類別索引 (用於計算梯度)
            subject_id: 受試者 ID (可選，用於 metadata)
            
        Returns:
            字典，格式為:
            {
                'layer_name': {
                    'activation': torch.Tensor,
                    'gradient': torch.Tensor,
                    'metadata': {
                        'subject_id': str,
                        'layer_name': str,
                        'activation_shape': tuple,
                        'gradient_shape': tuple,
                        'timestamp': str,
                        'target_class': int
                    }
                }
            }
        """
        # 清空之前的資料
        self.activations.clear()
        self.gradients.clear()
        
        # 確保輸入在正確的裝置上
        input_tensor = input_tensor.to(self.device)
        
        # 確保模型在評估模式
        self.model.eval()
        
        # 前向傳播
        output = self.model(input_tensor)
        
        # 計算目標類別的分數
        target_score = output[0, target_class]
        
        # 反向傳播以計算梯度
        self.model.zero_grad()
        target_score.backward()
        
        # 組織結果
        results = {}
        timestamp = datetime.now().isoformat()
        
        for layer_name in self.target_layers:
            if layer_name in self.activations and layer_name in self.gradients:
                activation = self.activations[layer_name]
                gradient = self.gradients[layer_name]
                
                results[layer_name] = {
                    'activation': activation.cpu(),
                    'gradient': gradient.cpu(),
                    'metadata': {
                        'subject_id': subject_id if subject_id else 'unknown',
                        'layer_name': layer_name,
                        'activation_shape': tuple(activation.shape),
                        'gradient_shape': tuple(gradient.shape),
                        'timestamp': timestamp,
                        'target_class': target_class,
                        'target_score': target_score.item()
                    }
                }
            else:
                print(f"⚠️ 警告: 層 '{layer_name}' 的資料未完整擷取")
        
        return results
    
    def save_to_disk(
        self, 
        data: Dict[str, Dict[str, Any]], 
        output_path: str,
        subject_id: Optional[str] = None
    ) -> None:
        """
        將擷取的資料儲存為 .pt 檔案。
        
        Args:
            data: extract() 方法返回的資料字典
            output_path: 輸出目錄路徑
            subject_id: 受試者 ID (用於檔名)
        """
        # 建立輸出目錄
        os.makedirs(output_path, exist_ok=True)
        
        # 為每個層儲存獨立的檔案
        for layer_name, layer_data in data.items():
            # 從 metadata 取得 subject_id (如果有的話)
            if subject_id is None:
                subject_id = layer_data['metadata'].get('subject_id', 'unknown')
            
            # 建立檔名
            filename = f"{subject_id}_{layer_name}_activations.pt"
            filepath = os.path.join(output_path, filename)
            
            # 儲存資料
            torch.save(layer_data, filepath)
            print(f"✅ 已儲存: {filepath}")
    
    def load_from_disk(self, filepath: str) -> Dict[str, Any]:
        """
        從 .pt 檔案載入擷取的資料。
        
        Args:
            filepath: .pt 檔案路徑
            
        Returns:
            包含 activation, gradient 和 metadata 的字典
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到檔案: {filepath}")
        
        data = torch.load(filepath, map_location='cpu')
        
        # 驗證資料格式
        required_keys = ['activation', 'gradient', 'metadata']
        if not all(key in data for key in required_keys):
            raise ValueError(f"檔案格式不正確，缺少必要的鍵: {required_keys}")
        
        print(f"✅ 已載入: {filepath}")
        return data
    
    def __del__(self):
        """解構函式，確保 hooks 被移除。"""
        self.remove_hooks()
