"""
CDDA Framework - Core Tools (Layer 1 + Layer 2)

This module implements the formalized Tool API as specified in CDDA_Architecture_Spec.md:
- Tool 1: get_diagnostic_report(subject_id)
- Tool 2: simulate_counterfactual(subject_id, features_to_mask)

These tools combine:
- Layer 1: Tool Kit (RF/SHAP)
- Layer 2: Trust/Calibration (UQ/Z-Score Logic)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.cnn_rf.end_to_end_inference import EndToEndPredictor


class CDDAToolKit:
    """
    CDDA Tool Kit - Implements Layer 1 (RF/SHAP) + Layer 2 (UQ/Z-Score)
    
    This class provides the two core tools for the CDDA agent:
    1. get_diagnostic_report() - Complete diagnostic data with UQ and anomaly detection
    2. simulate_counterfactual() - What-if analysis by masking features
    """
    
    def __init__(
        self,
        model_path: str = "model/cnn_rf/rf_model_NC_MCI_AD.joblib",
        data_root: str = "data/MRI_processed",
        uq_threshold: float = 0.8,
        z_score_threshold: float = 2.5
    ):
        """
        Initialize CDDA Tool Kit
        
        Args:
            model_path: Path to trained CNN-RF model (default: 3-class model)
            data_root: Root directory for MRI data
            uq_threshold: Threshold for high uncertainty (default: 0.8)
            z_score_threshold: Threshold for anomaly detection (default: 2.5)
        """
        self.model_path = Path(model_path)
        self.data_root = Path(data_root)
        self.uq_threshold = uq_threshold
        self.z_score_threshold = z_score_threshold
        
        # Define class mapping for 3-class model
        self.classes = {0: 'NC', 1: 'MCI', 2: 'AD'}
        
        # Initialize end-to-end predictor (Layer 1)
        print(f"\n[CDDA] Initializing Tool Kit...")
        self.predictor = EndToEndPredictor(
            model_path=str(model_path),
            data_root=str(data_root)
        )
        
        # Load population statistics for z-score calculation (Layer 2)
        self.population_stats = self._load_population_statistics()
        
        print(f"[OK] CDDA Tool Kit ready")
        print(f"   Model: {model_path}")
        print(f"   Classes: {list(self.classes.values())}")
        print(f"   UQ Threshold: {uq_threshold}")
        print(f"   Z-Score Threshold: ±{z_score_threshold}")
    
    def _load_population_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Load population mean and std for z-score calculation
        
        Returns:
            Dictionary with 'mean' and 'std' for each feature
        """
        try:
            # Load ROI features CSV
            roi_csv = Path("data/roi_features.csv")
            if not roi_csv.exists():
                print(f"[WARN] ROI features not found: {roi_csv}")
                return {}
            
            df = pd.read_csv(roi_csv)
            
            # Remove non-feature columns
            feature_cols = [col for col in df.columns if col not in ['Subject_ID', 'Group']]
            
            # Filter to GM features only if using GM model
            if 'GM_only' in str(self.model_path) or 'GM' in str(self.model_path).upper():
                feature_cols = [col for col in feature_cols if col.endswith('_GM')]
            
            # Calculate population statistics
            stats = {
                'mean': df[feature_cols].mean().to_dict(),
                'std': df[feature_cols].std().to_dict()
            }
            
            print(f"[OK] Loaded population statistics for {len(feature_cols)} features")
            return stats
            
        except Exception as e:
            print(f"[WARN] Could not load population statistics: {e}")
            return {}
    
    def _calculate_uq_score(
        self, 
        probabilities: Dict[str, float],
        confidence: float
    ) -> float:
        """
        Calculate Uncertainty Quantification (UQ) score
        
        UQ score is based on:
        1. Entropy of probability distribution (higher entropy = more uncertain)
        2. Confidence margin (difference between top 2 classes)
        
        Args:
            probabilities: Dictionary of class probabilities (supports 2 or 3 classes)
            confidence: Confidence of predicted class
        
        Returns:
            UQ score (0.0 to 1.0, higher = more uncertain)
        """
        # Convert to array
        probs = np.array(list(probabilities.values()))
        
        # Calculate entropy (normalized)
        epsilon = 1e-10  # Avoid log(0)
        entropy = -np.sum(probs * np.log(probs + epsilon))
        max_entropy = np.log(len(probs))  # Maximum possible entropy (log(2) or log(3))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate confidence margin (difference between top 2)
        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) >= 2:
            margin = sorted_probs[0] - sorted_probs[1]
            margin_uncertainty = 1.0 - margin  # Low margin = high uncertainty
        else:
            margin_uncertainty = 0.0
        
        # Combine metrics (weighted average)
        uq_score = 0.6 * normalized_entropy + 0.4 * margin_uncertainty
        
        return float(uq_score)
    
    def _calculate_z_scores(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate z-scores for all features
        
        Z-score = (value - population_mean) / population_std
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Dictionary of z-scores
        """
        if not self.population_stats:
            return {}
        
        z_scores = {}
        
        for feature_name, value in features.items():
            if feature_name in self.population_stats['mean']:
                mean = self.population_stats['mean'][feature_name]
                std = self.population_stats['std'][feature_name]
                
                if std > 0:
                    z_score = (value - mean) / std
                    z_scores[feature_name] = float(z_score)
                else:
                    z_scores[feature_name] = 0.0
        
        return z_scores
    
    def _detect_anomalies(
        self,
        z_scores: Dict[str, float]
    ) -> Dict:
        """
        Detect anomalous features based on z-scores
        
        Args:
            z_scores: Dictionary of z-scores
        
        Returns:
            Anomaly status dictionary
        """
        anomalous_regions = []
        
        for feature_name, z_score in z_scores.items():
            if abs(z_score) > self.z_score_threshold:
                # Extract ROI name (remove _GM, _FA, _MD suffix)
                roi_name = feature_name.rsplit('_', 1)[0]
                anomalous_regions.append(roi_name)
        
        return {
            'has_anomaly': len(anomalous_regions) > 0,
            'anomalous_regions': anomalous_regions,
            'anomaly_type': 'statistical_outlier' if anomalous_regions else None
        }
    
    def get_diagnostic_report(
        self,
        subject_id: str,
        verbose: bool = True
    ) -> Dict:
        """
        Tool 1: Get Diagnostic Report
        
        Provides all factual and contextual data for the CDDA Agent.
        
        This tool combines:
        - Layer 1: RF prediction + SHAP explainability
        - Layer 2: UQ scoring + Z-score anomaly detection
        
        Args:
            subject_id: Unique patient identifier (e.g., 'sub-0005')
            verbose: Print detailed information
        
        Returns:
            Dictionary with mandatory fields as per CDDA spec:
            - subject_id: str
            - prediction_result: str (AD, NC, or MCI)
            - confidence: float (0.0 to 1.0)
            - uq_score: float (0.0 to 1.0)
            - top_features: list of dicts with:
                - roi_name: str
                - feature_value: float
                - z_score: float
                - shap_value: float
                - rank: int
            - anomaly_status: dict with:
                - has_anomaly: bool
                - anomalous_regions: list of str
                - anomaly_type: str
            - metadata: dict
        """
        if verbose:
            print("\n" + "="*80)
            print("CDDA Tool 1: get_diagnostic_report()")
            print("="*80)
            print(f"Subject: {subject_id}")
        
        # Step 1: Run end-to-end prediction (Layer 1)
        if verbose:
            print(f"\n[Layer 1] Running RF prediction + SHAP analysis...")
        
        prediction_results = self.predictor.predict_subject(
            subject_id, 
            verbose=verbose
        )
        
        # Extract core prediction data
        prediction_result = prediction_results['predicted_label']
        confidence = prediction_results['confidence']
        probabilities = prediction_results['probabilities']
        features = prediction_results['features']
        shap_features = prediction_results.get('shap_features', [])
        
        # Step 2: Calculate UQ score (Layer 2)
        if verbose:
            print(f"\n[Layer 2] Calculating uncertainty quantification...")
        
        uq_score = self._calculate_uq_score(probabilities, confidence)
        
        if verbose:
            print(f"[OK] UQ Score: {uq_score:.3f}")
            if uq_score > self.uq_threshold:
                print(f"[ALERT] High uncertainty detected (> {self.uq_threshold})")
        
        # Step 3: Calculate z-scores (Layer 2)
        if verbose:
            print(f"\n[Layer 2] Calculating z-scores for anomaly detection...")
        
        z_scores = self._calculate_z_scores(features)
        
        # Step 4: Detect anomalies (Layer 2)
        anomaly_status = self._detect_anomalies(z_scores)
        
        if verbose:
            print(f"[OK] Anomaly detection complete")
            if anomaly_status['has_anomaly']:
                print(f"[ALERT] Anomalies detected in {len(anomaly_status['anomalous_regions'])} regions:")
                for region in anomaly_status['anomalous_regions'][:5]:
                    print(f"   - {region}")
        
        # Step 5: Compile top features with all metrics
        if verbose:
            print(f"\n[Integration] Compiling top features with SHAP + Z-scores...")
        
        top_features = []
        
        # Use SHAP features if available
        if shap_features:
            for i, shap_feat in enumerate(shap_features[:10], 1):
                feature_name = shap_feat['name']
                
                # Extract ROI name
                roi_name = feature_name.rsplit('_', 1)[0]
                
                top_features.append({
                    'roi_name': roi_name,
                    'feature_name': feature_name,
                    'feature_value': float(features.get(feature_name, 0.0)),
                    'z_score': float(z_scores.get(feature_name, 0.0)),
                    'shap_value': float(shap_feat['shap_value']),
                    'rank': i
                })
        else:
            # Fallback: use z-scores to rank features
            if verbose:
                print(f"[WARN] SHAP not available, ranking by z-score magnitude")
            
            sorted_z = sorted(
                z_scores.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:10]
            
            for i, (feature_name, z_score) in enumerate(sorted_z, 1):
                roi_name = feature_name.rsplit('_', 1)[0]
                
                top_features.append({
                    'roi_name': roi_name,
                    'feature_name': feature_name,
                    'feature_value': float(features.get(feature_name, 0.0)),
                    'z_score': float(z_score),
                    'shap_value': 0.0,  # Not available
                    'rank': i
                })
        
        # Step 6: Compile final report
        report = {
            'subject_id': subject_id,
            'prediction_result': prediction_result,
            'confidence': float(confidence),
            'uq_score': float(uq_score),
            'top_features': top_features,
            'anomaly_status': anomaly_status,
            'metadata': {
                'model_version': self.model_path.name,
                'timestamp': datetime.now().isoformat(),
                'true_label': prediction_results.get('true_label', 'unknown'),
                'correct_prediction': prediction_results.get('correct', None)
            }
        }
        
        if verbose:
            print(f"\n[SUCCESS] Diagnostic report generated")
            print(f"   Prediction: {prediction_result} ({confidence:.1%})")
            print(f"   UQ Score: {uq_score:.3f}")
            print(f"   Top Features: {len(top_features)}")
            print(f"   Anomalies: {len(anomaly_status['anomalous_regions'])}")
            print("="*80 + "\n")
        
        return report
    
    def simulate_counterfactual(
        self,
        subject_id: str,
        features_to_mask: List[str],
        verbose: bool = True
    ) -> Dict:
        """
        Tool 2: Simulate Counterfactual
        
        Execute a "What-If" prediction experiment by masking/adjusting features.
        
        This simulates what would happen if certain brain regions were "normal"
        (i.e., at population mean values).
        
        Args:
            subject_id: Unique patient identifier
            features_to_mask: List of ROI names or feature names to neutralize
            verbose: Print detailed information
        
        Returns:
            Dictionary with mandatory fields as per CDDA spec:
            - subject_id: str
            - original_prediction: str
            - original_confidence: float
            - new_prediction: str
            - new_confidence: float
            - confidence_delta: float
            - masked_features: list of dicts with:
                - roi_name: str
                - original_value: float
                - masked_value: float
                - impact: float
            - interpretation: str
        """
        if verbose:
            print("\n" + "="*80)
            print("CDDA Tool 2: simulate_counterfactual()")
            print("="*80)
            print(f"Subject: {subject_id}")
            print(f"Features to mask: {features_to_mask}")
        
        # Step 1: Get original prediction
        if verbose:
            print(f"\n[Step 1/3] Getting original prediction...")
        
        original_results = self.predictor.predict_subject(
            subject_id,
            verbose=False
        )
        
        original_prediction = original_results['predicted_label']
        original_confidence = original_results['confidence']
        original_features = original_results['features']
        
        if verbose:
            print(f"[OK] Original: {original_prediction} ({original_confidence:.1%})")
        
        # Step 2: Create counterfactual features
        if verbose:
            print(f"\n[Step 2/3] Creating counterfactual scenario...")
        
        counterfactual_features = original_features.copy()
        masked_feature_info = []
        
        for roi_or_feature in features_to_mask:
            # Find matching feature names
            matching_features = [
                fname for fname in original_features.keys()
                if roi_or_feature in fname
            ]
            
            for feature_name in matching_features:
                original_value = original_features[feature_name]
                
                # Mask with population mean
                if feature_name in self.population_stats['mean']:
                    masked_value = self.population_stats['mean'][feature_name]
                else:
                    # Fallback: use median of all features
                    masked_value = np.median(list(original_features.values()))
                
                counterfactual_features[feature_name] = masked_value
                
                masked_feature_info.append({
                    'roi_name': roi_or_feature,
                    'feature_name': feature_name,
                    'original_value': float(original_value),
                    'masked_value': float(masked_value),
                    'impact': 0.0  # Will be calculated after prediction
                })
                
                if verbose:
                    print(f"   Masked: {feature_name}")
                    print(f"      {original_value:.2f} → {masked_value:.2f}")
        
        # Step 3: Run counterfactual prediction
        if verbose:
            print(f"\n[Step 3/3] Running counterfactual prediction...")
        
        # Create DataFrame for prediction
        cf_df = pd.DataFrame([counterfactual_features])
        
        # Filter to GM features if needed
        if 'GM_only' in str(self.model_path) or 'GM' in str(self.model_path).upper():
            gm_features = [col for col in cf_df.columns if col.endswith('_GM')]
            cf_df = cf_df[gm_features]
        
        # Predict
        new_prediction_idx = self.predictor.model.predict(cf_df)[0]
        new_probabilities = self.predictor.model.predict_proba(cf_df)[0]
        
        # Convert prediction to class name (ensure it's a string)
        if isinstance(self.predictor.classes[0], str):
            new_prediction = self.predictor.classes[new_prediction_idx]
        else:
            # Classes are numeric, map to names
            class_names = ['AD', 'NC'] if len(self.predictor.classes) == 2 else ['AD', 'MCI', 'NC']
            new_prediction = class_names[new_prediction_idx]
        
        new_confidence = new_probabilities[new_prediction_idx]
        
        confidence_delta = new_confidence - original_confidence
        
        if verbose:
            print(f"[OK] Counterfactual: {new_prediction} ({new_confidence:.1%})")
            print(f"[OK] Confidence change: {confidence_delta:+.1%}")
        
        # Calculate impact for each masked feature
        for feat_info in masked_feature_info:
            feat_info['impact'] = float(confidence_delta / len(masked_feature_info))
        
        # Generate interpretation
        if abs(confidence_delta) < 0.05:
            interpretation = (
                f"Masking {len(features_to_mask)} feature(s) had minimal impact "
                f"({confidence_delta:+.1%}), suggesting these regions are not primary "
                f"drivers of the {original_prediction} diagnosis."
            )
        elif confidence_delta < 0:
            interpretation = (
                f"Masking {len(features_to_mask)} feature(s) reduced {original_prediction} "
                f"confidence by {abs(confidence_delta):.1%}, indicating these regions "
                f"are significant contributors to the diagnosis."
            )
        else:
            interpretation = (
                f"Masking {len(features_to_mask)} feature(s) increased confidence by "
                f"{confidence_delta:.1%}, suggesting these regions may be protective "
                f"or confounding factors."
            )
        
        # Compile results
        results = {
            'subject_id': subject_id,
            'original_prediction': original_prediction,
            'original_confidence': float(original_confidence),
            'new_prediction': new_prediction,
            'new_confidence': float(new_confidence),
            'confidence_delta': float(confidence_delta),
            'masked_features': masked_feature_info,
            'interpretation': interpretation
        }
        
        if verbose:
            print(f"\n[Interpretation]")
            print(f"   {interpretation}")
            print("="*80 + "\n")
        
        return results


def demo_tool_1():
    """Demo: Tool 1 - get_diagnostic_report()"""
    print("\n" + "="*80)
    print("DEMO: CDDA Tool 1 - get_diagnostic_report()")
    print("="*80)
    
    toolkit = CDDAToolKit()
    
    # Test with a subject
    report = toolkit.get_diagnostic_report('sub-0005', verbose=True)
    
    # Display formatted report
    print("\n[Formatted Report]")
    print(f"Subject: {report['subject_id']}")
    print(f"Prediction: {report['prediction_result']} ({report['confidence']:.1%})")
    print(f"UQ Score: {report['uq_score']:.3f}")
    print(f"\nTop 5 Features:")
    for feat in report['top_features'][:5]:
        print(f"  {feat['rank']}. {feat['roi_name']}")
        print(f"     Z-score: {feat['z_score']:+.2f}, SHAP: {feat['shap_value']:+.4f}")
    
    print(f"\nAnomaly Status:")
    print(f"  Has Anomaly: {report['anomaly_status']['has_anomaly']}")
    if report['anomaly_status']['has_anomaly']:
        print(f"  Regions: {', '.join(report['anomaly_status']['anomalous_regions'][:5])}")


def demo_tool_2():
    """Demo: Tool 2 - simulate_counterfactual()"""
    print("\n" + "="*80)
    print("DEMO: CDDA Tool 2 - simulate_counterfactual()")
    print("="*80)
    
    toolkit = CDDAToolKit()
    
    # First get diagnostic report to identify top features
    report = toolkit.get_diagnostic_report('sub-0005', verbose=False)
    
    # Mask top 3 features
    top_rois = [feat['roi_name'] for feat in report['top_features'][:3]]
    
    print(f"\nMasking top 3 ROIs: {top_rois}")
    
    cf_results = toolkit.simulate_counterfactual(
        'sub-0005',
        top_rois,
        verbose=True
    )
    
    print("\n[Counterfactual Summary]")
    print(f"Original: {cf_results['original_prediction']} ({cf_results['original_confidence']:.1%})")
    print(f"Counterfactual: {cf_results['new_prediction']} ({cf_results['new_confidence']:.1%})")
    print(f"Change: {cf_results['confidence_delta']:+.1%}")


if __name__ == "__main__":
    # Run demos
    demo_tool_1()
    print("\n\n")
    demo_tool_2()
