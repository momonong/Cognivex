"""
Inference Pipeline for Multi-modal ROI Classification
多模態 ROI 分類推理 Pipeline
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import *
from resnet3d_mini import MultiModalFeatureExtractor
from patch_extractor import AAL116PatchExtractor


class MultiModalROIPredictor:
    """
    Predictor for multi-modal MRI classification
    
    Pipeline:
    1. Extract 116 ROI patches from each modality
    2. Extract features using trained Mini-CNNs
    3. Classify using XGBoost
    4. Provide interpretability analysis
    """
    
    def __init__(
        self,
        feature_extractor_path,
        xgboost_path,
        device='cuda'
    ):
        """
        Parameters:
        -----------
        feature_extractor_path : str or Path
            Path to trained feature extractor checkpoint
        xgboost_path : str or Path
            Path to trained XGBoost model
        device : str
            Device for inference ('cpu' or 'cuda')
        """
        self.device = device
        
        # Load feature extractor
        print("Loading feature extractor...")
        self.feature_extractor = MultiModalFeatureExtractor(
            feature_dim=FEATURE_DIM_PER_ROI,
            initial_filters=RESNET_CONFIG['initial_filters']
        ).to(device)
        
        checkpoint = torch.load(feature_extractor_path, map_location=device)
        self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
        self.feature_extractor.eval()
        
        # Load XGBoost
        print("Loading XGBoost classifier...")
        self.xgboost = joblib.load(xgboost_path)
        
        # Initialize patch extractor
        print("Initializing patch extractor...")
        self.patch_extractor = AAL116PatchExtractor(
            target_patch_size=PATCH_CONFIG['target_patch_size'],
            padding=PATCH_CONFIG['padding'],
            min_patch_size=PATCH_CONFIG['min_patch_size'],
            device='cpu'
        )
        
        # Get ROI labels
        self.roi_labels = self.patch_extractor.get_roi_labels()
        
        print("[OK] Predictor initialized successfully!")
    
    def predict(self, t1_path, t2_path, dwi_path, return_features=False):
        """
        Predict class for one subject
        
        Parameters:
        -----------
        t1_path : str or Path
            Path to T1 image
        t2_path : str or Path
            Path to T2-FLAIR image
        dwi_path : str or Path
            Path to DWI image
        return_features : bool
            Whether to return extracted features
        
        Returns:
        --------
        result : dict
            Dictionary containing:
            - prediction: Predicted class (0: NC, 1: MCI, 2: AD)
            - probabilities: Class probabilities
            - confidence: Confidence score
            - features: Extracted features (if return_features=True)
        """
        # Extract patches
        print("Extracting ROI patches...")
        patches = self.patch_extractor.extract_patches_from_subject(
            t1_path, t2_path, dwi_path
        )
        
        # Move to device and add batch dimension
        t1_patches = patches['T1'].unsqueeze(0).to(self.device)
        t2_patches = patches['T2_FLAIR'].unsqueeze(0).to(self.device)
        dwi_patches = patches['DWI'].unsqueeze(0).to(self.device)
        
        # Extract features
        print("Extracting features...")
        with torch.no_grad():
            features = self.feature_extractor(t1_patches, t2_patches, dwi_patches)
            features = features.cpu().numpy()
        
        # Predict with XGBoost
        print("Classifying...")
        prediction = self.xgboost.predict(features)[0]
        probabilities = self.xgboost.predict_proba(features)[0]
        confidence = probabilities.max()
        
        result = {
            'prediction': int(prediction),
            'prediction_label': ['NC', 'MCI', 'AD'][prediction],
            'probabilities': {
                'NC': float(probabilities[0]),
                'MCI': float(probabilities[1]),
                'AD': float(probabilities[2])
            },
            'confidence': float(confidence)
        }
        
        if return_features:
            result['features'] = features
        
        return result
    
    def analyze_feature_importance(self, t1_path, t2_path, dwi_path, top_k=30):
        """
        Analyze which ROIs and modalities are most important for prediction
        
        Parameters:
        -----------
        t1_path, t2_path, dwi_path : str or Path
            Paths to MRI images
        top_k : int
            Number of top features to return
        
        Returns:
        --------
        importance_df : pd.DataFrame
            DataFrame with feature importance analysis
        """
        # Get prediction and features
        result = self.predict(t1_path, t2_path, dwi_path, return_features=True)
        features = result['features'].flatten()
        
        # Get feature importance from XGBoost
        feature_importance = self.xgboost.feature_importances_
        
        # Calculate contribution (feature value × importance)
        contribution = np.abs(features) * feature_importance
        
        # Map to ROI and modality
        importance_data = []
        
        for idx in range(len(contribution)):
            # Decode feature index
            # Total features: 116 ROIs × 3 modalities × 64 features
            roi_idx = idx // (3 * FEATURE_DIM_PER_ROI)
            modality_idx = (idx // FEATURE_DIM_PER_ROI) % 3
            feature_idx = idx % FEATURE_DIM_PER_ROI
            
            modality_name = MODALITIES[modality_idx]
            roi_name = self.roi_labels[roi_idx] if roi_idx < len(self.roi_labels) else f"ROI_{roi_idx}"
            
            importance_data.append({
                'feature_idx': idx,
                'roi_idx': roi_idx,
                'roi_name': roi_name,
                'modality': modality_name,
                'feature_in_roi': feature_idx,
                'feature_value': features[idx],
                'importance': feature_importance[idx],
                'contribution': contribution[idx]
            })
        
        # Create DataFrame
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('contribution', ascending=False)
        
        # Aggregate by ROI
        roi_importance = importance_df.groupby(['roi_idx', 'roi_name', 'modality'])['contribution'].sum()
        roi_importance = roi_importance.reset_index().sort_values('contribution', ascending=False)
        
        return {
            'feature_importance': importance_df.head(top_k),
            'roi_importance': roi_importance.head(top_k),
            'prediction': result['prediction_label'],
            'probabilities': result['probabilities'],
            'confidence': result['confidence']
        }
    
    def batch_predict(self, subject_list, output_path=None):
        """
        Batch prediction for multiple subjects
        
        Parameters:
        -----------
        subject_list : list of dict
            List of subjects with keys: 't1_path', 't2_path', 'dwi_path', 'subject_id'
        output_path : str or Path
            Path to save results CSV
        
        Returns:
        --------
        results_df : pd.DataFrame
            DataFrame with predictions for all subjects
        """
        results = []
        
        for subject in subject_list:
            print(f"\nProcessing {subject['subject_id']}...")
            
            try:
                result = self.predict(
                    subject['t1_path'],
                    subject['t2_path'],
                    subject['dwi_path']
                )
                
                results.append({
                    'subject_id': subject['subject_id'],
                    'prediction': result['prediction_label'],
                    'prob_NC': result['probabilities']['NC'],
                    'prob_MCI': result['probabilities']['MCI'],
                    'prob_AD': result['probabilities']['AD'],
                    'confidence': result['confidence']
                })
                
            except Exception as e:
                print(f"[WARN] Error processing {subject['subject_id']}: {e}")
                results.append({
                    'subject_id': subject['subject_id'],
                    'prediction': 'ERROR',
                    'prob_NC': 0,
                    'prob_MCI': 0,
                    'prob_AD': 0,
                    'confidence': 0
                })
        
        results_df = pd.DataFrame(results)
        
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"\n[OK] Results saved to: {output_path}")
        
        return results_df


def main():
    """Example usage"""
    print("="*80)
    print("Multi-modal ROI Classification - Inference")
    print("="*80)
    
    # Initialize predictor
    predictor = MultiModalROIPredictor(
        feature_extractor_path=MODEL_DIR / 'best_feature_extractor.pth',
        xgboost_path=MODEL_DIR / 'xgboost_classifier.pkl',
        device=DEVICE
    )
    
    # Example: Find first test subject
    data_root = Path(DATA_ROOT)
    test_subjects = []
    
    for label_name in ['NC', 'MCI', 'AD']:
        class_dir = data_root / label_name
        if class_dir.exists():
            t1_files = list(class_dir.glob("*_T1.nii.gz"))[:1]  # Take first subject
            
            for t1_path in t1_files:
                base_name = str(t1_path).replace("_T1.nii.gz", "")
                t2_path = Path(base_name + "_T2_FLAIR.nii.gz")
                dwi_path = Path(base_name + "_DWI.nii.gz")
                
                if t2_path.exists() and dwi_path.exists():
                    test_subjects.append({
                        'subject_id': t1_path.stem.replace("_T1", ""),
                        't1_path': t1_path,
                        't2_path': t2_path,
                        'dwi_path': dwi_path,
                        'true_label': label_name
                    })
    
    if len(test_subjects) == 0:
        print("[WARN] No test subjects found")
        return
    
    # Test single prediction
    print(f"\n{'='*80}")
    print("Testing single prediction")
    print("="*80)
    
    subject = test_subjects[0]
    print(f"Subject: {subject['subject_id']}")
    print(f"True label: {subject['true_label']}")
    
    result = predictor.predict(
        subject['t1_path'],
        subject['t2_path'],
        subject['dwi_path']
    )
    
    print(f"\n[OK] Prediction completed!")
    print(f"   Predicted: {result['prediction_label']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"     {label}: {prob:.2%}")
    
    # Test feature importance analysis
    print(f"\n{'='*80}")
    print("Analyzing feature importance")
    print("="*80)
    
    analysis = predictor.analyze_feature_importance(
        subject['t1_path'],
        subject['t2_path'],
        subject['dwi_path'],
        top_k=20
    )
    
    print(f"\nTop 10 most important ROIs:")
    print(analysis['roi_importance'].head(10).to_string(index=False))
    
    # Save analysis
    output_dir = OUTPUT_DIR / 'inference'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis['roi_importance'].to_csv(
        output_dir / f"{subject['subject_id']}_roi_importance.csv",
        index=False
    )
    
    print(f"\n[OK] Analysis saved to: {output_dir}")


if __name__ == "__main__":
    main()
