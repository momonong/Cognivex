"""
End-to-End CNN-RF Inference Pipeline

Complete pipeline from raw MRI images to prediction:
1. Load raw MRI images (GM, FA, MD)
2. Extract ROI features using AAL3 atlas
3. Load trained CNN-RF model
4. Make prediction
5. Generate report

Usage:
    python scripts/cnn_rf/end_to_end_inference.py --subject sub-0005
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.cnn_rf.extract_roi_features import ROIFeatureExtractor

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not installed. Local explainability will not be available.")
    print("[WARN] Install with: pip install shap")


class EndToEndPredictor:
    """End-to-end predictor from MRI images to diagnosis"""
    
    def __init__(
        self,
        model_path="model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        atlas_path="data/aal3/AAL3v1_1mm.nii.gz",
        atlas_labels_path="data/aal3/AAL3v1.json",
        data_root="data/MRI_processed"
    ):
        """
        Initialize end-to-end predictor
        
        Args:
            model_path: Path to trained CNN-RF model
            atlas_path: Path to AAL3 atlas
            atlas_labels_path: Path to AAL3 labels
            data_root: Root directory for MRI data
        """
        self.model_path = Path(model_path)
        self.data_root = Path(data_root)
        
        # Load model
        print(f"\n[1/3] Loading CNN-RF model...")
        self.model = joblib.load(self.model_path)
        
        # Extract classes from model
        if hasattr(self.model, 'classes_'):
            self.classes = self.model.classes_
        elif hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps['model'], 'classes_'):
            self.classes = self.model.named_steps['model'].classes_
        else:
            # Fallback: infer from model name
            if 'NC_vs_AD' in self.model_path.name:
                self.classes = ['AD', 'NC']
            else:
                self.classes = ['Class_0', 'Class_1']
        
        print(f"[OK] Model loaded: {self.model_path.name}")
        print(f"[OK] Classes: {self.classes}")
        
        # Initialize feature extractor
        print(f"\n[2/3] Initializing ROI feature extractor...")
        self.extractor = ROIFeatureExtractor(
            atlas_path=atlas_path,
            atlas_labels_path=atlas_labels_path
        )
        print(f"[OK] Feature extractor ready")
        
        # Initialize SHAP explainer if available
        self.shap_explainer = None
        if SHAP_AVAILABLE:
            try:
                print(f"\n[3/3] Initializing SHAP explainer...")
                # Extract the actual RandomForest model from pipeline
                if hasattr(self.model, 'named_steps'):
                    if 'model' in self.model.named_steps:
                        rf_model = self.model.named_steps['model']
                        self.shap_explainer = shap.TreeExplainer(rf_model)
                        print(f"[OK] SHAP explainer initialized for local explanations")
                        print(f"[INFO] SHAP will use TRANSFORMED data (after scaling & selection)")
                    else:
                        print(f"[WARN] Could not find 'model' step in pipeline")
                else:
                    # Not a pipeline, use the model directly
                    self.shap_explainer = shap.TreeExplainer(self.model)
                    print(f"[OK] SHAP explainer initialized (no pipeline)")
            except Exception as e:
                print(f"[WARN] Could not initialize SHAP explainer: {e}")
                self.shap_explainer = None
    
    def find_subject_directory(self, subject_id):
        """
        Find subject directory in data_root
        
        Args:
            subject_id: Subject identifier (e.g., 'sub-0005')
        
        Returns:
            Path to subject directory
        """
        # Search in all group directories
        for group in ['NC', 'MCI', 'AD']:
            subject_dir = self.data_root / group / subject_id
            if subject_dir.exists():
                return subject_dir, group
        
        raise FileNotFoundError(
            f"Subject {subject_id} not found in {self.data_root}. "
            f"Searched in: NC/, MCI/, AD/"
        )
    
    def get_feature_names(self, features_dict: Dict) -> List[str]:
        """
        Get feature names from features dictionary
        
        Args:
            features_dict: Dictionary of features
        
        Returns:
            List of feature names
        """
        return list(features_dict.keys())
    
    def calculate_shap_values(
        self, 
        feature_df: pd.DataFrame,
        verbose: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate SHAP values for local explainability
        
        IMPORTANT: SHAP needs the TRANSFORMED data (after scaling and selection),
        not the raw data. We must manually apply the pipeline transformations.
        
        Args:
            feature_df: DataFrame with RAW features
            verbose: Print information
        
        Returns:
            Tuple of (shap_values, selected_feature_names)
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            if verbose:
                print(f"[WARN] SHAP not available, using global feature importances")
            return None, None
        
        try:
            if verbose:
                print(f"\n[SHAP] Calculating local feature importance...")
            
            # CRITICAL: Extract pipeline components
            if hasattr(self.model, 'named_steps'):
                # Get scaler
                scaler = None
                if 'scaler' in self.model.named_steps:
                    scaler = self.model.named_steps['scaler']
                elif 'scale' in self.model.named_steps:
                    scaler = self.model.named_steps['scale']
                
                # Get selector
                selector = None
                if 'select' in self.model.named_steps:
                    selector = self.model.named_steps['select']
                elif 'selector' in self.model.named_steps:
                    selector = self.model.named_steps['selector']
                
                # Get RF model
                rf_model = None
                if 'model' in self.model.named_steps:
                    rf_model = self.model.named_steps['model']
                
                if verbose:
                    print(f"[INFO] Pipeline components:")
                    print(f"  - Scaler: {type(scaler).__name__ if scaler else 'None'}")
                    print(f"  - Selector: {type(selector).__name__ if selector else 'None'}")
                    print(f"  - Model: {type(rf_model).__name__ if rf_model else 'None'}")
                
                # STEP 1: Apply scaling (if exists)
                if scaler:
                    X_scaled = scaler.transform(feature_df)
                    if verbose:
                        print(f"[OK] Applied scaling: {feature_df.shape} -> {X_scaled.shape}")
                else:
                    X_scaled = feature_df.values
                
                # STEP 2: Apply feature selection (if exists)
                if selector:
                    X_selected = selector.transform(X_scaled)
                    
                    # Get selected feature names
                    selected_mask = selector.get_support()
                    original_feature_names = list(feature_df.columns)
                    selected_feature_names = [name for name, selected in zip(original_feature_names, selected_mask) if selected]
                    
                    if verbose:
                        print(f"[OK] Applied feature selection: {X_scaled.shape} -> {X_selected.shape}")
                        print(f"[OK] Selected {len(selected_feature_names)} features")
                else:
                    X_selected = X_scaled
                    selected_feature_names = list(feature_df.columns)
                
                # STEP 3: Calculate SHAP on the TRANSFORMED data
                if rf_model and self.shap_explainer:
                    shap_values = self.shap_explainer.shap_values(X_selected)
                    
                    if verbose:
                        print(f"[DEBUG] SHAP raw output type: {type(shap_values)}")
                        if isinstance(shap_values, list):
                            print(f"[DEBUG] SHAP is list with {len(shap_values)} elements")
                            for i, arr in enumerate(shap_values):
                                print(f"[DEBUG]   Class {i} shape: {arr.shape}")
                        else:
                            print(f"[DEBUG] SHAP shape: {shap_values.shape}")
                    
                    # [FIX] Handle Binary Classification Output
                    # Random Forest SHAP returns a list: [array_class_0, array_class_1]
                    # We need to select the POSITIVE CLASS (AD)
                    # In sklearn, classes are sorted: [0, 1] where 0=AD, 1=NC
                    # But we want SHAP values that push TOWARDS AD (positive class)
                    
                    if isinstance(shap_values, list):
                        if verbose:
                            print(f"[INFO] SHAP output is list. Selecting Class 1 (positive/AD direction)...")
                        
                        # For binary classification, we want class 1 (positive class)
                        # This gives us SHAP values that are positive when pushing towards AD
                        shap_values_selected = shap_values[1]  # Class 1
                        
                        if verbose:
                            print(f"[DEBUG] Selected class 1 shape: {shap_values_selected.shape}")
                        
                        # Get first sample
                        shap_values_ad = shap_values_selected[0]
                        
                    elif len(shap_values.shape) == 3:
                        # If it returns a 3D array (samples, features, classes)
                        if verbose:
                            print(f"[INFO] SHAP output is 3D array. Selecting Class 1...")
                        shap_values_ad = shap_values[0, :, 1]  # First sample, all features, class 1
                        
                    else:
                        # Single output, shape: (n_samples, n_features)
                        if verbose:
                            print(f"[INFO] SHAP output is 2D array (single class)")
                        shap_values_ad = shap_values[0]
                    
                    # Ensure it's 1D
                    shap_values_ad = np.array(shap_values_ad).flatten()
                    
                    if verbose:
                        print(f"[DEBUG] Final SHAP shape: {shap_values_ad.shape}")
                        print(f"[OK] SHAP values: {len(shap_values_ad)} values for {len(selected_feature_names)} features")
                    
                    # Validate lengths
                    if len(shap_values_ad) != len(selected_feature_names):
                        if verbose:
                            print(f"[ERROR] SHAP values length mismatch: {len(shap_values_ad)} vs {len(selected_feature_names)}")
                        raise ValueError(
                            f"SHAP values length ({len(shap_values_ad)}) does not match "
                            f"feature names length ({len(selected_feature_names)})"
                        )
                    
                    if verbose:
                        print(f"[OK] SHAP values and feature names aligned correctly")
                    
                    return shap_values_ad, selected_feature_names
                else:
                    if verbose:
                        print(f"[WARN] Could not extract RF model from pipeline")
                    return None, None
            else:
                # Not a pipeline, use original logic
                feature_names = list(feature_df.columns)
                shap_values = self.shap_explainer.shap_values(feature_df)
                
                if isinstance(shap_values, list):
                    shap_values_ad = shap_values[0][0]
                else:
                    shap_values_ad = shap_values[0]
                
                return shap_values_ad, feature_names
            
        except Exception as e:
            if verbose:
                print(f"[WARN] SHAP calculation failed: {e}")
                import traceback
                traceback.print_exc()
            return None, None
    
    def get_top_shap_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_n: int = 10,
        class_name: str = "AD"
    ) -> List[Dict]:
        """
        Get top features by SHAP value
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            top_n: Number of top features to return
            class_name: Class name for interpretation
        
        Returns:
            List of dicts with feature info
        """
        if shap_values is None or feature_names is None:
            return []
        
        # Ensure shap_values and feature_names are 1D
        shap_values = np.array(shap_values).flatten()
        abs_shap = np.abs(shap_values)
        
        # Validate lengths match
        if len(shap_values) != len(feature_names):
            raise ValueError(
                f"SHAP values length ({len(shap_values)}) does not match "
                f"feature names length ({len(feature_names)})"
            )
        
        # Adjust top_n if it's larger than available features
        actual_top_n = min(top_n, len(shap_values))
        
        # Get top indices
        top_indices = np.argsort(abs_shap)[-actual_top_n:][::-1]
        
        # Create feature info list
        top_features = []
        for i in range(len(top_indices)):
            idx = int(top_indices[i])  # Ensure it's a Python int
            
            # Safety check
            if idx >= len(feature_names):
                print(f"[WARN] Index {idx} out of range for {len(feature_names)} features, skipping")
                continue
            
            feature_info = {
                'name': feature_names[idx],
                'shap_value': float(shap_values[idx]),
                'abs_shap_value': float(abs_shap[idx]),
                'direction': 'towards AD' if shap_values[idx] > 0 else 'towards NC',
                'impact': 'High' if abs_shap[idx] > np.mean(abs_shap) else 'Medium'
            }
            top_features.append(feature_info)
        
        return top_features
    
    def predict_subject(self, subject_id, verbose=True):
        """
        Predict diagnosis for a subject
        
        Args:
            subject_id: Subject identifier
            verbose: Print detailed information
        
        Returns:
            Dictionary with prediction results
        """
        if verbose:
            print("\n" + "="*80)
            print(f"End-to-End Prediction: {subject_id}")
            print("="*80)
        
        # Step 1: Find subject directory
        if verbose:
            print(f"\n[Step 1/4] Locating subject data...")
        
        subject_dir, true_group = self.find_subject_directory(subject_id)
        
        if verbose:
            print(f"[OK] Found: {subject_dir}")
            print(f"[OK] True label: {true_group}")
        
        # Step 2: Extract ROI features
        if verbose:
            print(f"\n[Step 2/4] Extracting ROI features from MRI images...")
        
        features = self.extractor.extract_subject_features(subject_dir)
        
        if verbose:
            print(f"[OK] Extracted {len(features)} features")
        
        # Convert to DataFrame (model expects this format)
        feature_df = pd.DataFrame([features])
        
        # [FIX] Filter features based on model type
        # GM-only models expect only GM features
        if 'GM_only' in str(self.model_path) or 'GM' in str(self.model_path).upper():
            if verbose:
                print(f"[INFO] Detected GM-only model, filtering to GM features only...")
            
            # Keep only GM features
            gm_features = [col for col in feature_df.columns if col.endswith('_GM')]
            feature_df = feature_df[gm_features]
            
            if verbose:
                print(f"[OK] Filtered: {len(features)} -> {len(gm_features)} GM features")
        
        # Step 3: Make prediction
        if verbose:
            print(f"\n[Step 3/4] Making prediction...")
        
        prediction_idx = self.model.predict(feature_df)[0]
        probabilities = self.model.predict_proba(feature_df)[0]
        
        # Convert prediction index to class name
        if isinstance(self.classes[0], str):
            predicted_class = self.classes[prediction_idx]
        else:
            # Classes are numeric, map to names
            class_names = ['AD', 'NC'] if len(self.classes) == 2 else ['AD', 'MCI', 'NC']
            predicted_class = class_names[prediction_idx]
        
        confidence = probabilities[prediction_idx]
        
        if verbose:
            print(f"[OK] Prediction: {predicted_class}")
            print(f"[OK] Confidence: {confidence:.1%}")
            print(f"\n[Probabilities]")
            # Map class indices to names for display
            if isinstance(self.classes[0], str):
                class_names = self.classes
            else:
                class_names = ['AD', 'NC'] if len(self.classes) == 2 else ['AD', 'MCI', 'NC']
            
            for cls, prob in zip(class_names, probabilities):
                print(f"  {cls}: {prob:.1%}")
        
        # Step 4: Calculate SHAP values for local explainability
        if verbose:
            print(f"\n[Step 4/4] Analyzing local feature importance (SHAP)...")
        
        shap_values, feature_names = self.calculate_shap_values(feature_df, verbose=verbose)
        
        if shap_values is not None:
            # Get top SHAP features
            top_shap_features = self.get_top_shap_features(
                shap_values, 
                feature_names, 
                top_n=10,
                class_name=predicted_class
            )
            
            if verbose:
                print(f"\n[SHAP] Top 5 Features for This Subject:")
                for i, feat in enumerate(top_shap_features[:5], 1):
                    direction_symbol = "→" if feat['direction'] == 'towards AD' else "←"
                    print(f"      {i}. {feat['name']}")
                    print(f"         SHAP: {feat['shap_value']:+.4f} {direction_symbol} {feat['direction']}")
        else:
            # Fallback to global feature importances
            top_shap_features = []
            if verbose:
                print(f"[INFO] Using global feature importances (SHAP not available)")
                try:
                    if hasattr(self.model, 'named_steps'):
                        rf_model = self.model.named_steps['model']
                        feature_importances = rf_model.feature_importances_
                        top_indices = np.argsort(feature_importances)[-5:][::-1]
                        print(f"\n[Global] Top 5 Important Features:")
                        for i, idx in enumerate(top_indices, 1):
                            print(f"      {i}. Feature {idx}: {feature_importances[idx]:.4f}")
                except Exception as e:
                    print(f"[WARN] Could not extract feature importances: {e}")
        
        # Compile results
        results = {
            'subject_id': subject_id,
            'subject_dir': str(subject_dir),
            'true_label': true_group,
            'predicted_label': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                cls: float(prob) for cls, prob in zip(self.classes, probabilities)
            },
            'correct': predicted_class == true_group,
            'features': features,
            'shap_features': top_shap_features if shap_values is not None else [],
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def generate_report(self, results):
        """
        Generate human-readable report
        
        Args:
            results: Prediction results dictionary
        
        Returns:
            Report string
        """
        report = f"""
{'='*80}
CNN-RF Diagnosis Report
{'='*80}

Subject Information:
  Subject ID: {results['subject_id']}
  Data Location: {results['subject_dir']}
  Analysis Time: {results['timestamp']}

Diagnosis:
  Predicted: {results['predicted_label']}
  Confidence: {results['confidence']:.1%}
  
  Probability Distribution:
"""
        
        for cls, prob in results['probabilities'].items():
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            report += f"    {cls:<5} {bar} {prob:.1%}\n"
        
        report += f"""
Ground Truth:
  True Label: {results['true_label']}
  Prediction: {'CORRECT' if results['correct'] else 'INCORRECT'}

Model Information:
  Model: {self.model_path.name}
  Classes: {', '.join([str(c) for c in self.classes])}
  Total Features: {len(results['features'])}

{'='*80}
"""
        
        return report
    
    def batch_predict(self, subject_ids, save_results=True):
        """
        Predict for multiple subjects
        
        Args:
            subject_ids: List of subject IDs
            save_results: Save results to CSV
        
        Returns:
            DataFrame with all results
        """
        print("\n" + "="*80)
        print(f"Batch Prediction: {len(subject_ids)} subjects")
        print("="*80)
        
        all_results = []
        
        for i, subject_id in enumerate(subject_ids, 1):
            print(f"\n[{i}/{len(subject_ids)}] Processing: {subject_id}")
            
            try:
                results = self.predict_subject(subject_id, verbose=False)
                all_results.append({
                    'Subject_ID': results['subject_id'],
                    'True_Label': results['true_label'],
                    'Predicted_Label': results['predicted_label'],
                    'Confidence': results['confidence'],
                    'Correct': results['correct']
                })
                
                status = "[OK]" if results['correct'] else "[X]"
                print(f"  {status} {results['predicted_label']} (confidence: {results['confidence']:.1%})")
                
            except Exception as e:
                print(f"  [ERROR] {e}")
                all_results.append({
                    'Subject_ID': subject_id,
                    'True_Label': 'ERROR',
                    'Predicted_Label': 'ERROR',
                    'Confidence': 0.0,
                    'Correct': False
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Calculate accuracy
        accuracy = df['Correct'].mean()
        
        print("\n" + "="*80)
        print("Batch Prediction Summary")
        print("="*80)
        print(f"Total subjects: {len(df)}")
        print(f"Correct: {df['Correct'].sum()}")
        print(f"Incorrect: {(~df['Correct']).sum()}")
        print(f"Accuracy: {accuracy:.1%}")
        
        # Per-class accuracy
        print(f"\nPer-class accuracy:")
        for true_label in df['True_Label'].unique():
            if true_label != 'ERROR':
                class_df = df[df['True_Label'] == true_label]
                class_acc = class_df['Correct'].mean()
                print(f"  {true_label}: {class_acc:.1%} ({class_df['Correct'].sum()}/{len(class_df)})")
        
        # Save results
        if save_results:
            output_path = Path("output/cnn_rf/batch_predictions.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"\n[OK] Results saved: {output_path}")
        
        return df


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="End-to-end CNN-RF inference from MRI images"
    )
    parser.add_argument(
        '--subject',
        help='Subject ID to predict (e.g., sub-0005)'
    )
    parser.add_argument(
        '--batch',
        nargs='+',
        help='Multiple subject IDs for batch prediction'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Predict all subjects in dataset'
    )
    parser.add_argument(
        '--model',
        default='model/cnn_rf/rf_model_NC_vs_AD.joblib',
        help='Path to trained model'
    )
    parser.add_argument(
        '--data-root',
        default='data/MRI_processed',
        help='Root directory for MRI data'
    )
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save report to file'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = EndToEndPredictor(
        model_path=args.model,
        data_root=args.data_root
    )
    
    if args.subject:
        # Single subject prediction
        results = predictor.predict_subject(args.subject)
        
        # Generate and print report
        report = predictor.generate_report(results)
        print(report)
        
        # Save report if requested
        if args.save_report:
            output_path = Path(f"output/cnn_rf/reports/{args.subject}_report.txt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"[OK] Report saved: {output_path}")
    
    elif args.batch:
        # Batch prediction
        predictor.batch_predict(args.batch)
    
    elif args.all:
        # Predict all subjects
        data_root = Path(args.data_root)
        all_subjects = []
        
        for group in ['NC', 'MCI', 'AD']:
            group_dir = data_root / group
            if group_dir.exists():
                subjects = [d.name for d in group_dir.iterdir() if d.is_dir()]
                all_subjects.extend(subjects)
        
        print(f"[INFO] Found {len(all_subjects)} subjects")
        predictor.batch_predict(all_subjects)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
