"""
CNN-RF 模型推理接口
Inference Interface for CNN-RF Model
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
import nibabel as nib
from typing import Dict, List, Tuple, Optional

class CNNRF_Predictor:
    """
    CNN-RF 模型預測器
    
    功能：
    1. 載入訓練好的 RF 模型
    2. 對新樣本進行預測
    3. 提取重要特徵和腦區
    4. 生成可視化地圖
    """
    
    def __init__(
        self,
        model_path: str = "model/cnn_rf/rf_model_NC_vs_AD.joblib",
        atlas_path: str = "data/aal3/AAL3v1_1mm.nii.gz",
        atlas_labels_path: str = "data/aal3/AAL3v1.json"
    ):
        """
        初始化預測器
        
        Parameters:
        -----------
        model_path : str
            訓練好的模型路徑
        atlas_path : str
            AAL3 圖譜路徑
        atlas_labels_path : str
            AAL3 標籤 JSON 路徑
        """
        self.model_path = Path(model_path)
        self.atlas_path = Path(atlas_path)
        self.atlas_labels_path = Path(atlas_labels_path)
        
        # 載入模型
        self.model = self._load_model()
        
        # 載入圖譜標籤
        self.atlas_labels = self._load_atlas_labels()
        
        # 提取類別名稱
        self.class_names = self._extract_class_names()
        
        print(f"[OK] CNN-RF Predictor initialized")
        print(f"   Model: {self.model_path.name}")
        print(f"   Classes: {self.class_names}")
    
    def _load_model(self):
        """載入訓練好的模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        model = joblib.load(self.model_path)
        print(f"[OK] Model loaded: {self.model_path}")
        return model
    
    def _load_atlas_labels(self) -> Dict[str, int]:
        """載入 AAL3 標籤"""
        if not self.atlas_labels_path.exists():
            print(f"[WARN] Atlas labels not found: {self.atlas_labels_path}")
            return {}
        
        with open(self.atlas_labels_path, 'r', encoding='utf-8') as f:
            labels_raw = json.load(f)
        
        # 建立名稱 -> 索引的映射
        name_to_index = {name: int(idx) for idx, name in labels_raw.items()}
        return name_to_index
    
    def _extract_class_names(self) -> List[str]:
        """從模型路徑提取類別名稱"""
        model_name = self.model_path.stem
        
        if "NC_vs_AD" in model_name:
            return ['AD', 'NC']  # 按字母排序
        elif "NC_MCI_AD" in model_name:
            return ['AD', 'MCI', 'NC']  # 按字母排序
        else:
            return ['Class_0', 'Class_1']
    
    def predict(
        self, 
        features: pd.DataFrame,
        return_proba: bool = True
    ) -> Dict:
        """
        對新樣本進行預測
        
        Parameters:
        -----------
        features : pd.DataFrame
            特徵數據框，包含 ROI 特徵
        return_proba : bool
            是否返回概率
        
        Returns:
        --------
        results : dict
            包含預測結果的字典
        """
        # 移除非特徵列
        X = features.drop(columns=['Subject_ID', 'Group'], errors='ignore')
        
        # 預測
        y_pred = self.model.predict(X)
        
        results = {
            'predictions': y_pred,
            'predicted_labels': [self.class_names[p] for p in y_pred]
        }
        
        # 如果需要概率
        if return_proba:
            y_proba = self.model.predict_proba(X)
            results['probabilities'] = y_proba
            
            # 為每個樣本添加詳細信息
            results['detailed'] = []
            for i in range(len(y_pred)):
                detail = {
                    'predicted_class': self.class_names[y_pred[i]],
                    'confidence': y_proba[i][y_pred[i]],
                    'probabilities': {
                        cls: prob for cls, prob in zip(self.class_names, y_proba[i])
                    }
                }
                results['detailed'].append(detail)
        
        return results
    
    def get_feature_importance(
        self, 
        top_n: int = 30
    ) -> pd.DataFrame:
        """
        獲取特徵重要性
        
        Parameters:
        -----------
        top_n : int
            返回前 N 個重要特徵
        
        Returns:
        --------
        importance_df : pd.DataFrame
            特徵重要性數據框
        """
        # 從 pipeline 中提取 RandomForest 模型
        rf_model = self.model.named_steps['model']
        
        # 獲取特徵重要性
        importances = rf_model.feature_importances_
        
        # 獲取選擇後的特徵名稱
        selector = self.model.named_steps['select']
        selected_mask = selector.get_support()
        
        # 需要原始特徵名稱（這需要在訓練時保存）
        # 這裡我們假設特徵名稱已經保存在模型中
        # 如果沒有，我們需要從訓練數據中獲取
        
        # 創建重要性數據框
        importance_df = pd.DataFrame({
            'feature_index': range(len(importances)),
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def extract_important_rois(
        self, 
        top_n: int = 10
    ) -> List[str]:
        """
        提取最重要的腦區
        
        Parameters:
        -----------
        top_n : int
            返回前 N 個重要腦區
        
        Returns:
        --------
        roi_names : list
            重要腦區名稱列表
        """
        importance_df = self.get_feature_importance(top_n=top_n)
        
        # 從特徵名稱中提取 ROI 名稱
        # 假設特徵名稱格式為 "ROI_Name_GM" 或 "ROI_Name_FA"
        roi_names = []
        for idx in importance_df['feature_index']:
            # 這裡需要實際的特徵名稱
            # 暫時返回索引
            roi_names.append(f"ROI_{idx}")
        
        return roi_names
    
    def create_brain_map(
        self,
        important_rois: List[str],
        output_path: str = "output/cnn_rf/important_rois_map.nii.gz",
        template_path: str = "data/templates/MNI152_T1_1mm_brain.nii.gz"
    ) -> str:
        """
        創建重要腦區的 3D 地圖
        
        Parameters:
        -----------
        important_rois : list
            重要腦區名稱列表
        output_path : str
            輸出 NIfTI 文件路徑
        template_path : str
            MNI 模板路徑
        
        Returns:
        --------
        output_path : str
            生成的地圖路徑
        """
        import ants
        
        # 載入圖譜和模板
        atlas_img = nib.load(self.atlas_path)
        atlas_data = atlas_img.get_fdata().astype(int)
        
        template_img = nib.load(template_path)
        
        # 重新採樣圖譜到 MNI 空間
        atlas_img_ants = ants.image_read(str(self.atlas_path))
        template_img_ants = ants.image_read(template_path)
        
        atlas_resampled = ants.resample_image_to_target(
            atlas_img_ants,
            template_img_ants,
            interp_type='nearestNeighbor'
        )
        atlas_data = atlas_resampled.numpy().astype(int)
        
        # 創建特徵地圖
        feature_map = np.zeros(atlas_data.shape, dtype=np.int16)
        
        # 標記重要腦區
        for i, roi_name in enumerate(important_rois):
            if roi_name in self.atlas_labels:
                roi_index = self.atlas_labels[roi_name]
                feature_map[atlas_data == roi_index] = i + 1
        
        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_img = nib.Nifti1Image(
            feature_map,
            template_img.affine,
            template_img.header
        )
        nib.save(output_img, output_path)
        
        print(f"[OK] Brain map saved: {output_path}")
        return str(output_path)


def load_roi_features(csv_path: str = "data/roi_features.csv") -> pd.DataFrame:
    """
    載入 ROI 特徵
    
    Parameters:
    -----------
    csv_path : str
        特徵 CSV 文件路徑
    
    Returns:
    --------
    features : pd.DataFrame
        特徵數據框
    """
    features = pd.read_csv(csv_path)
    print(f"[OK] Loaded {len(features)} subjects from {csv_path}")
    return features


def main():
    """示例：使用 CNN-RF 模型進行預測"""
    print("="*80)
    print("CNN-RF Model Inference Example")
    print("="*80)
    
    # 1. 初始化預測器
    predictor = CNNRF_Predictor(
        model_path="model/cnn_rf/rf_model_NC_vs_AD.joblib"
    )
    
    # 2. 載入特徵
    features = load_roi_features("data/roi_features.csv")
    
    # 3. 過濾 NC 和 AD 樣本
    features_filtered = features[features['Group'].isin(['NC', 'AD'])]
    
    # 4. 進行預測
    print(f"\n[*] Predicting {len(features_filtered)} samples...")
    results = predictor.predict(features_filtered)
    
    # 5. 顯示結果
    print(f"\n[*] Prediction Results:")
    for i, detail in enumerate(results['detailed'][:5]):  # 只顯示前 5 個
        print(f"\nSample {i+1}:")
        print(f"  Predicted: {detail['predicted_class']}")
        print(f"  Confidence: {detail['confidence']:.3f}")
        print(f"  Probabilities: {detail['probabilities']}")
    
    # 6. 提取重要特徵
    print(f"\n[*] Top 10 Important Features:")
    importance_df = predictor.get_feature_importance(top_n=10)
    print(importance_df)
    
    print("\n" + "="*80)
    print("[SUCCESS] Inference completed!")
    print("="*80)


if __name__ == "__main__":
    main()
