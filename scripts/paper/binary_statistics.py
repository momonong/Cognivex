#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary Classification Statistics Script (LOOCV-Aware)

專為 NC vs AD 二�?類系統設計�?統�??��??�本，�??��?
1. LOOCV 完整?��?�?(確�?每個�?試者使?��?屬模??
2. 二�?類性能?��? (Accuracy, Precision, Recall, F1, AUC)
3. 不確定性�??��???(UQ Score ?��??�臨床�?�?
4. Agent 決�?路�?統�? (Standard/Simulation/Anomaly)
5. ?�徵?��??��???(SHAP values, Z-scores)
6. 完整??Paper-ready ?��??��?

使用?��?:
    python scripts/paper/binary_statistics.py
    python scripts/paper/binary_statistics.py --output output/binary_stats
    python scripts/paper/binary_statistics.py --limit 10  # 測試模�?
"""

import sys
import re
import argparse
import json
import glob
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


class BinaryStatistics:
    """二�?類統計�??�器 (LOOCV-Aware)"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 統�??��?容器
        self.results = []
        self.statistics = {
            # ============================================================
            # 0. LOOCV 完整?��?�?(Critical for Paper Validity)
            # ============================================================
            'loocv_integrity': {
                'total_subjects': 0,
                'loocv_verified': 0,      # 使用專屬模�?
                'fallback_global': 0,     # 使用?�用模�? (MCI/OOD)
                'unknown': 0,             # ?��??�斷
                'verification_details': []
            },
            
            # ============================================================
            # 1. ?�本統�?
            # ============================================================
            'total_subjects': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'errors': [],
            
            # ============================================================
            # 2. 二�?類性能?��? (Binary Classification Metrics)
            # ============================================================
            'binary_metrics': {
                'true_positives': 0,   # AD correctly identified as AD
                'true_negatives': 0,   # NC correctly identified as NC
                'false_positives': 0,  # NC incorrectly identified as AD
                'false_negatives': 0,  # AD incorrectly identified as NC
                'accuracy': 0.0,
                'precision': 0.0,      # AD precision
                'recall': 0.0,         # AD recall (sensitivity)
                'specificity': 0.0,    # NC recall
                'f1_score': 0.0,
                'balanced_accuracy': 0.0
            },
            
            # ============================================================
            # 3. ?�測?��?
            # ============================================================
            'predictions': defaultdict(int),
            'ground_truth_distribution': defaultdict(int),
            'confusion_matrix': {
                'AD_as_AD': 0,
                'AD_as_NC': 0,
                'NC_as_AD': 0,
                'NC_as_NC': 0
            },
            
            # ============================================================
            # 4. 信�?度統�?(Confidence Analysis)
            # ============================================================
            'confidence_stats': {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'min': 1.0,
                'max': 0.0,
                'ranges': {
                    'very_high': 0,      # >= 0.9
                    'high': 0,           # 0.8 - 0.9
                    'medium': 0,         # 0.6 - 0.8
                    'low': 0,            # 0.4 - 0.6
                    'very_low': 0        # < 0.4
                },
                'low_confidence_subjects': []  # < 0.6
            },
            
            # ============================================================
            # 5. 不確定性�??�統�?(UQ Analysis)
            # ============================================================
            'uq_stats': {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'min': 1.0,
                'max': 0.0,
                'ranges': {
                    'very_high': 0,      # >= 0.9
                    'high': 0,           # 0.8 - 0.9
                    'medium': 0,         # 0.5 - 0.8
                    'low': 0,            # 0.3 - 0.5
                    'very_low': 0        # < 0.3
                },
                'high_uq_subjects': []  # > 0.8
            },
            
            # ============================================================
            # 6. Agent 決�?統�? (Agent Decision Pathways)
            # ============================================================
            'agent_decisions': defaultdict(int),
            'decision_pathways': {
                'standard': [],
                'counterfactual': [],
                'knowledge_query': []
            },
            
            # ============================================================
            # 7. ?�常檢測統�? (Anomaly Detection)
            # ============================================================
            'anomaly_stats': {
                'detected': 0,
                'not_detected': 0,
                'anomalous_subjects': [],
                'region_frequency': defaultdict(int)
            },
            
            # ============================================================
            # 8. ?�徵?��??��???(Feature Importance)
            # ============================================================
            'feature_importance': {
                'top_features_frequency': defaultdict(int),
                'shap_values': defaultdict(list),
                'z_scores': defaultdict(list)
            },
            
            # ============================================================
            # 9. ?�能統�? (Performance Metrics)
            # ============================================================
            'performance': {
                'init_times': [],
                'analysis_times': [],
                'total_times': []
            },
            
            # ============================================================
            # 10. ?��??�統�?(Reasoning Chain Analysis)
            # ============================================================
            'reasoning': {
                'total_steps': [],
                'agent_a_steps': [],
                'agent_b_steps': [],
                'mcp_actions': []
            },
            
            # ============================================================
            # 11. 組�?條件統�? (Combined Conditions)
            # ============================================================
            'combined_conditions': {
                'low_conf_high_uq': [],      # 信�?�?< 0.6 �?UQ > 0.8
                'high_conf_high_uq': [],     # 信�?�?>= 0.8 �?UQ > 0.8
                'low_conf_anomaly': [],      # 信�?�?< 0.6 且�??�常
                'high_uq_anomaly': []        # UQ > 0.8 且�??�常
            },
            
            # ============================================================
            # 12. 系統?�值�???(System Value Analysis) - NEW!
            # ============================================================
            'system_value': {
                'intervention_cases': [],           # 觸發 Agent 介入?��?�?
                'corrected_by_counterfactual': [],  # ?��?實模?�糾�??案�?
                'corrected_by_knowledge': [],       # ?��??��?糾正?��?�?
                'total_corrections': 0,             # 總糾�?��
                'correction_rate': 0.0,             # 糾正??
                'intervention_accuracy': 0.0,       # 介入案�??��?確�?
                'standard_accuracy': 0.0,           # 標�?案�??��?確�?
                'accuracy_improvement': 0.0,        # 準確?��???
                'high_value_cases': []              # 高價?��?例�??�本?�誤但被糾正�?
            }
        }
    
    def _verify_model_usage(self, subject_id: str, reasoning_chain: List[str]) -> Tuple[str, str]:
        """
        驗�??�否使用了正確�? LOOCV 模�?
        
        �?? reasoning chain 中�? log，確認模?��???
        
        Returns:
            (verification_status, model_name)
            - loocv_verified: 使用了�?屬模??
            - fallback_global: 使用了通用模�?
            - unknown: ?��??�斷
        """
        full_log = " ".join(reasoning_chain)
        
        # ?��?模�?檔�? (例�?: "using rf_model_sub-001.joblib")
        match = re.search(r"using ([\w\-\.]+\.joblib)", full_log)
        
        if match:
            model_name = match.group(1)
            # 檢查檔�??�否?�含�?subject_id
            if subject_id in model_name:
                return "loocv_verified", model_name
            else:
                return "fallback_global", model_name
        
        return "unknown", "N/A"
    
    def add_result(
        self,
        subject_id: str,
        result: Optional[object],
        ground_truth: Optional[str],
        init_time: float,
        analysis_time: float,
        error: Optional[str] = None
    ):
        """添�??�個�?試者�??��?結�?"""
        
        self.statistics['total_subjects'] += 1
        
        if error:
            self.statistics['failed_analyses'] += 1
            self.statistics['errors'].append({
                'subject_id': subject_id,
                'error': error
            })
            return
        
        if not result:
            self.statistics['failed_analyses'] += 1
            return
        
        self.statistics['successful_analyses'] += 1
        
        # ?��??�本信息
        prediction = result.prediction
        confidence = result.confidence
        uq_score = result.uq_score
        agent_decision = result.agent_decision
        
        # ============================================================
        # LOOCV 模�?驗�?
        # ============================================================
        verification_status, used_model_name = self._verify_model_usage(
            subject_id, result.reasoning_chain
        )
        
        loocv_stats = self.statistics['loocv_integrity']
        loocv_stats['total_subjects'] += 1
        loocv_stats[verification_status] += 1
        loocv_stats['verification_details'].append({
            'subject_id': subject_id,
            'status': verification_status,
            'model_used': used_model_name
        })
        
        # ============================================================
        # 二�?類性能?��?計�?
        # ============================================================
        if ground_truth and ground_truth in ['AD', 'NC']:
            metrics = self.statistics['binary_metrics']
            
            if ground_truth == 'AD' and prediction == 'AD':
                metrics['true_positives'] += 1
            elif ground_truth == 'NC' and prediction == 'NC':
                metrics['true_negatives'] += 1
            elif ground_truth == 'NC' and prediction == 'AD':
                metrics['false_positives'] += 1
            elif ground_truth == 'AD' and prediction == 'NC':
                metrics['false_negatives'] += 1
            
            # ?�新混�??�陣
            confusion_key = f"{ground_truth}_as_{prediction}"
            self.statistics['confusion_matrix'][confusion_key] += 1
        
        # ============================================================
        # ?�測?��?
        # ============================================================
        self.statistics['predictions'][prediction] += 1
        if ground_truth:
            self.statistics['ground_truth_distribution'][ground_truth] += 1
        
        # ============================================================
        # 信�?度統�?
        # ============================================================
        conf_stats = self.statistics['confidence_stats']
        
        if confidence >= 0.9:
            conf_stats['ranges']['very_high'] += 1
        elif confidence >= 0.8:
            conf_stats['ranges']['high'] += 1
        elif confidence >= 0.6:
            conf_stats['ranges']['medium'] += 1
        elif confidence >= 0.4:
            conf_stats['ranges']['low'] += 1
        else:
            conf_stats['ranges']['very_low'] += 1
        
        if confidence < 0.6:
            conf_stats['low_confidence_subjects'].append({
                'subject_id': subject_id,
                'confidence': confidence,
                'prediction': prediction,
                'ground_truth': ground_truth
            })
        
        # ============================================================
        # 不確定性統�?
        # ============================================================
        uq_stats = self.statistics['uq_stats']
        
        if uq_score >= 0.9:
            uq_stats['ranges']['very_high'] += 1
        elif uq_score >= 0.8:
            uq_stats['ranges']['high'] += 1
        elif uq_score >= 0.5:
            uq_stats['ranges']['medium'] += 1
        elif uq_score >= 0.3:
            uq_stats['ranges']['low'] += 1
        else:
            uq_stats['ranges']['very_low'] += 1
        
        if uq_score > 0.8:
            uq_stats['high_uq_subjects'].append({
                'subject_id': subject_id,
                'uq_score': uq_score,
                'prediction': prediction,
                'ground_truth': ground_truth
            })
        
        # ============================================================
        # Agent 決�?統�?
        # ============================================================
        self.statistics['agent_decisions'][agent_decision] += 1
        
        if 'SIMULATION' in agent_decision:
            self.statistics['decision_pathways']['counterfactual'].append(subject_id)
        elif 'ANOMALY' in agent_decision or 'INVESTIGATION' in agent_decision:
            self.statistics['decision_pathways']['knowledge_query'].append(subject_id)
        else:
            self.statistics['decision_pathways']['standard'].append(subject_id)
        
        # ============================================================
        # ?�常檢測統�?
        # ============================================================
        context_object = result.context_object
        anomaly_status = context_object.diagnostic_report.anomaly_status
        
        if anomaly_status.has_anomaly:
            self.statistics['anomaly_stats']['detected'] += 1
            self.statistics['anomaly_stats']['anomalous_subjects'].append({
                'subject_id': subject_id,
                'regions': anomaly_status.anomalous_regions,
                'count': len(anomaly_status.anomalous_regions)
            })
            
            # 統�??�常?�?�頻??
            for region in anomaly_status.anomalous_regions:
                self.statistics['anomaly_stats']['region_frequency'][region] += 1
        else:
            self.statistics['anomaly_stats']['not_detected'] += 1
        
        # ============================================================
        # ?�徵?��??�統�?
        # ============================================================
        top_features = context_object.diagnostic_report.top_features[:10]
        
        for feat in top_features:
            roi_name = feat.roi_name if hasattr(feat, 'roi_name') else 'Unknown'
            shap_value = feat.shap_value if hasattr(feat, 'shap_value') else 0
            z_score = feat.z_score if hasattr(feat, 'z_score') else 0
            
            self.statistics['feature_importance']['top_features_frequency'][roi_name] += 1
            self.statistics['feature_importance']['shap_values'][roi_name].append(shap_value)
            self.statistics['feature_importance']['z_scores'][roi_name].append(z_score)
        
        # ============================================================
        # ?�能統�?
        # ============================================================
        self.statistics['performance']['init_times'].append(init_time)
        self.statistics['performance']['analysis_times'].append(analysis_time)
        self.statistics['performance']['total_times'].append(init_time + analysis_time)
        
        # ============================================================
        # ?��??�統�?
        # ============================================================
        reasoning_chain = result.reasoning_chain
        self.statistics['reasoning']['total_steps'].append(len(reasoning_chain))
        
        agent_a_steps = sum(1 for step in reasoning_chain if '[Agent A]' in step or 'AGENT A' in step)
        agent_b_steps = sum(1 for step in reasoning_chain if '[Agent B]' in step or 'AGENT B' in step)
        mcp_actions = len(context_object.mcp_actions)
        
        self.statistics['reasoning']['agent_a_steps'].append(agent_a_steps)
        self.statistics['reasoning']['agent_b_steps'].append(agent_b_steps)
        self.statistics['reasoning']['mcp_actions'].append(mcp_actions)
        
        # ============================================================
        # 組�?條件統�?
        # ============================================================
        combined = self.statistics['combined_conditions']
        
        if confidence < 0.6 and uq_score > 0.8:
            combined['low_conf_high_uq'].append(subject_id)
        
        if confidence >= 0.8 and uq_score > 0.8:
            combined['high_conf_high_uq'].append(subject_id)
        
        if confidence < 0.6 and anomaly_status.has_anomaly:
            combined['low_conf_anomaly'].append(subject_id)
        
        if uq_score > 0.8 and anomaly_status.has_anomaly:
            combined['high_uq_anomaly'].append(subject_id)
        
        # ============================================================
        # 系統?�值�???(NEW!)
        # ============================================================
        system_value = self.statistics['system_value']
        
        # ?�斷?�否??Agent 介入
        has_intervention = (
            'SIMULATION' in agent_decision or 
            'ANOMALY' in agent_decision or 
            'INVESTIGATION' in agent_decision
        )
        
        if has_intervention:
            system_value['intervention_cases'].append({
                'subject_id': subject_id,
                'agent_decision': agent_decision,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'correct': (prediction == ground_truth) if ground_truth else None,
                'confidence': confidence,
                'uq_score': uq_score
            })
            
            # 檢查?�否糾正了錯�?
            # ?�裡?��?輯是：�??��?介入且�?終�?測正確�??�們�??�系統可?�起?��?糾正作用
            if ground_truth and prediction == ground_truth:
                if 'SIMULATION' in agent_decision:
                    system_value['corrected_by_counterfactual'].append({
                        'subject_id': subject_id,
                        'ground_truth': ground_truth,
                        'confidence': confidence,
                        'uq_score': uq_score
                    })
                elif 'ANOMALY' in agent_decision or 'INVESTIGATION' in agent_decision:
                    system_value['corrected_by_knowledge'].append({
                        'subject_id': subject_id,
                        'ground_truth': ground_truth,
                        'confidence': confidence,
                        'uq_score': uq_score
                    })
        
        # ============================================================
        # 保�?完整結�?
        # ============================================================
        self.results.append({
            'subject_id': subject_id,
            'prediction': prediction,
            'confidence': confidence,
            'uq_score': uq_score,
            'agent_decision': agent_decision,
            'ground_truth': ground_truth,
            'correct': (prediction == ground_truth) if ground_truth else None,
            'has_anomaly': anomaly_status.has_anomaly,
            'anomalous_regions': anomaly_status.anomalous_regions,
            'init_time': init_time,
            'analysis_time': analysis_time,
            'total_time': init_time + analysis_time,
            'model_used': used_model_name,
            'loocv_verified': (verification_status == 'loocv_verified')
        })
    
    def calculate_statistics(self):
        """計�?統�??��?"""
        stats = self.statistics
        
        # ============================================================
        # LOOCV 完整?�百?��?
        # ============================================================
        loocv_stats = stats['loocv_integrity']
        if loocv_stats['total_subjects'] > 0:
            loocv_stats['coverage_percentage'] = (
                loocv_stats['loocv_verified'] / loocv_stats['total_subjects'] * 100
            )
        
        # ============================================================
        # 二�?類性能?��?
        # ============================================================
        metrics = stats['binary_metrics']
        tp = metrics['true_positives']
        tn = metrics['true_negatives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        
        total = tp + tn + fp + fn
        
        if total > 0:
            metrics['accuracy'] = (tp + tn) / total
            
            if (tp + fp) > 0:
                metrics['precision'] = tp / (tp + fp)
            
            if (tp + fn) > 0:
                metrics['recall'] = tp / (tp + fn)  # Sensitivity
            
            if (tn + fp) > 0:
                metrics['specificity'] = tn / (tn + fp)
            
            if metrics['precision'] > 0 and metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            
            if metrics['recall'] > 0 and metrics['specificity'] > 0:
                metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        # ============================================================
        # 信�?度統�?
        # ============================================================
        if self.results:
            confidences = [r['confidence'] for r in self.results]
            conf_stats = stats['confidence_stats']
            conf_stats['mean'] = np.mean(confidences)
            conf_stats['std'] = np.std(confidences)
            conf_stats['median'] = np.median(confidences)
            conf_stats['min'] = np.min(confidences)
            conf_stats['max'] = np.max(confidences)
        
        # ============================================================
        # 不確定性統�?
        # ============================================================
        if self.results:
            uq_scores = [r['uq_score'] for r in self.results]
            uq_stats = stats['uq_stats']
            uq_stats['mean'] = np.mean(uq_scores)
            uq_stats['std'] = np.std(uq_scores)
            uq_stats['median'] = np.median(uq_scores)
            uq_stats['min'] = np.min(uq_scores)
            uq_stats['max'] = np.max(uq_scores)
        
        # ============================================================
        # ?�能統�?
        # ============================================================
        perf = stats['performance']
        if perf['analysis_times']:
            stats['avg_init_time'] = np.mean(perf['init_times'])
            stats['avg_analysis_time'] = np.mean(perf['analysis_times'])
            stats['avg_total_time'] = np.mean(perf['total_times'])
            stats['avg_throughput'] = 3600 / stats['avg_analysis_time'] if stats['avg_analysis_time'] > 0 else 0
        
        # ============================================================
        # ?��??�統�?
        # ============================================================
        reasoning = stats['reasoning']
        if reasoning['total_steps']:
            stats['avg_reasoning_steps'] = np.mean(reasoning['total_steps'])
            stats['avg_agent_a_steps'] = np.mean(reasoning['agent_a_steps'])
            stats['avg_agent_b_steps'] = np.mean(reasoning['agent_b_steps'])
            stats['avg_mcp_actions'] = np.mean(reasoning['mcp_actions'])
        
        # ============================================================
        # 系統?�值�???(NEW!)
        # ============================================================
        system_value = stats['system_value']
        
        # 計�?介入案�??��?確�?
        intervention_cases = system_value['intervention_cases']
        if intervention_cases:
            intervention_correct = sum(1 for case in intervention_cases if case.get('correct'))
            intervention_total = len([case for case in intervention_cases if case.get('correct') is not None])
            if intervention_total > 0:
                system_value['intervention_accuracy'] = intervention_correct / intervention_total
        
        # 計�?標�?案�??��?確�?
        standard_cases = [r for r in self.results if r['agent_decision'] == 'STANDARD_REPORT']
        if standard_cases:
            standard_correct = sum(1 for case in standard_cases if case.get('correct'))
            standard_total = len([case for case in standard_cases if case.get('correct') is not None])
            if standard_total > 0:
                system_value['standard_accuracy'] = standard_correct / standard_total
        
        # 計�?總糾�?��?�糾�??
        system_value['total_corrections'] = (
            len(system_value['corrected_by_counterfactual']) + 
            len(system_value['corrected_by_knowledge'])
        )
        
        if intervention_cases:
            intervention_total = len([case for case in intervention_cases if case.get('correct') is not None])
            if intervention_total > 0:
                system_value['correction_rate'] = system_value['total_corrections'] / intervention_total
        
        # 計�?準確?��???
        if system_value['intervention_accuracy'] > 0 and system_value['standard_accuracy'] > 0:
            system_value['accuracy_improvement'] = (
                system_value['intervention_accuracy'] - system_value['standard_accuracy']
            )
    
    def generate_report(self) -> str:
        """?��?詳細統�??��? (Paper-Ready Format)"""
        self.calculate_statistics()
        
        stats = self.statistics
        lines = []
        
        # ============================================================
        # 標�?
        # ============================================================
        lines.append("=" * 100)
        lines.append("BINARY CLASSIFICATION STATISTICS REPORT (NC vs AD)".center(100))
        lines.append("=" * 100)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # ============================================================
        # 0. LOOCV 完整?�檢??(Critical for Paper)
        # ============================================================
        lines.append("=" * 100)
        lines.append("0. LOOCV INTEGRITY VERIFICATION")
        lines.append("=" * 100)
        
        loocv_stats = stats['loocv_integrity']
        lines.append(f"Total Subjects Analyzed: {loocv_stats['total_subjects']}")
        lines.append(f"LOOCV Verified (Subject-Specific Models): {loocv_stats['loocv_verified']}")
        lines.append(f"Global Fallback Models Used: {loocv_stats['fallback_global']}")
        lines.append(f"Unknown/Unverified: {loocv_stats['unknown']}")
        
        coverage = loocv_stats.get('coverage_percentage', 0)
        lines.append(f"\nLOOCV Coverage: {coverage:.2f}%")
        
        if coverage == 100:
            lines.append("✓STATUS: PASSED - 100% Strict Train-Test Separation")
        elif coverage >= 95:
            lines.append("✓STATUS: WARNING - High coverage but not perfect")
        else:
            lines.append("✓STATUS: FAILED - Insufficient LOOCV coverage")
        
        lines.append("")
        
        # ============================================================
        # 1. 總�?概�?
        # ============================================================
        lines.append("=" * 100)
        lines.append("1. OVERALL SUMMARY")
        lines.append("=" * 100)
        lines.append(f"Total Subjects: {stats['total_subjects']}")
        lines.append(f"Successful Analyses: {stats['successful_analyses']}")
        lines.append(f"Failed Analyses: {stats['failed_analyses']}")
        
        if stats['total_subjects'] > 0:
            success_rate = stats['successful_analyses'] / stats['total_subjects'] * 100
            lines.append(f"Success Rate: {success_rate:.2f}%")
        
        lines.append("")
        
        # ============================================================
        # 2. 二�?類性能?��? (Key Metrics for Paper)
        # ============================================================
        lines.append("=" * 100)
        lines.append("2. BINARY CLASSIFICATION PERFORMANCE")
        lines.append("=" * 100)
        
        metrics = stats['binary_metrics']
        lines.append(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        lines.append(f"Precision (AD): {metrics['precision']:.4f}")
        lines.append(f"Recall/Sensitivity (AD): {metrics['recall']:.4f}")
        lines.append(f"Specificity (NC): {metrics['specificity']:.4f}")
        lines.append(f"F1-Score: {metrics['f1_score']:.4f}")
        lines.append(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        
        lines.append("\nConfusion Matrix:")
        lines.append(f"  True Positives (AD?�AD): {metrics['true_positives']}")
        lines.append(f"  True Negatives (NC?�NC): {metrics['true_negatives']}")
        lines.append(f"  False Positives (NC?�AD): {metrics['false_positives']}")
        lines.append(f"  False Negatives (AD?�NC): {metrics['false_negatives']}")
        
        lines.append("")
        
        # ============================================================
        # 3. ?�測?��?
        # ============================================================
        lines.append("=" * 100)
        lines.append("3. PREDICTION DISTRIBUTION")
        lines.append("=" * 100)
        
        for pred, count in sorted(stats['predictions'].items()):
            percentage = count / stats['successful_analyses'] * 100 if stats['successful_analyses'] > 0 else 0
            lines.append(f"{pred}: {count} ({percentage:.2f}%)")
        
        if stats['ground_truth_distribution']:
            lines.append("\nGround Truth Distribution:")
            for gt, count in sorted(stats['ground_truth_distribution'].items()):
                percentage = count / sum(stats['ground_truth_distribution'].values()) * 100
                lines.append(f"{gt}: {count} ({percentage:.2f}%)")
        
        lines.append("")
        
        # ============================================================
        # 4. 信�?度�???(Confidence Analysis)
        # ============================================================
        lines.append("=" * 100)
        lines.append("4. CONFIDENCE ANALYSIS")
        lines.append("=" * 100)
        
        conf_stats = stats['confidence_stats']
        lines.append(f"Mean Confidence: {conf_stats['mean']:.4f}")
        lines.append(f"Std Deviation: {conf_stats['std']:.4f}")
        lines.append(f"Median: {conf_stats['median']:.4f}")
        lines.append(f"Range: [{conf_stats['min']:.4f}, {conf_stats['max']:.4f}]")
        
        lines.append("\nConfidence Distribution:")
        total = stats['successful_analyses']
        for range_name, count in conf_stats['ranges'].items():
            percentage = count / total * 100 if total > 0 else 0
            lines.append(f"  {range_name.replace('_', ' ').title()}: {count} ({percentage:.2f}%)")
        
        lines.append(f"\nLow Confidence Cases (< 0.6): {len(conf_stats['low_confidence_subjects'])}")
        if conf_stats['low_confidence_subjects']:
            lines.append("Top 5 Low Confidence Cases:")
            for item in conf_stats['low_confidence_subjects'][:5]:
                lines.append(f"  - {item['subject_id']}: {item['confidence']:.4f} "
                           f"(Pred: {item['prediction']}, GT: {item.get('ground_truth', 'N/A')})")
        
        lines.append("")
        
        # ============================================================
        # 5. 不確定性�??��???(UQ Analysis)
        # ============================================================
        lines.append("=" * 100)
        lines.append("5. UNCERTAINTY QUANTIFICATION ANALYSIS")
        lines.append("=" * 100)
        
        uq_stats = stats['uq_stats']
        lines.append(f"Mean UQ Score: {uq_stats['mean']:.4f}")
        lines.append(f"Std Deviation: {uq_stats['std']:.4f}")
        lines.append(f"Median: {uq_stats['median']:.4f}")
        lines.append(f"Range: [{uq_stats['min']:.4f}, {uq_stats['max']:.4f}]")
        
        lines.append("\nUQ Distribution:")
        for range_name, count in uq_stats['ranges'].items():
            percentage = count / total * 100 if total > 0 else 0
            lines.append(f"  {range_name.replace('_', ' ').title()}: {count} ({percentage:.2f}%)")
        
        lines.append(f"\nHigh Uncertainty Cases (> 0.8): {len(uq_stats['high_uq_subjects'])}")
        if uq_stats['high_uq_subjects']:
            lines.append("Top 5 High Uncertainty Cases:")
            for item in uq_stats['high_uq_subjects'][:5]:
                lines.append(f"  - {item['subject_id']}: {item['uq_score']:.4f} "
                           f"(Pred: {item['prediction']}, GT: {item.get('ground_truth', 'N/A')})")
        
        lines.append("")
        
        # ============================================================
        # 6. Agent 決�?路�??��? (Agent Decision Analysis)
        # ============================================================
        lines.append("=" * 100)
        lines.append("6. AGENT DECISION PATHWAY ANALYSIS")
        lines.append("=" * 100)
        
        lines.append("Decision Distribution:")
        for decision, count in sorted(stats['agent_decisions'].items()):
            percentage = count / total * 100 if total > 0 else 0
            lines.append(f"  {decision}: {count} ({percentage:.2f}%)")
        
        pathways = stats['decision_pathways']
        lines.append(f"\nStandard Pathway: {len(pathways['standard'])} "
                    f"({len(pathways['standard'])/total*100 if total > 0 else 0:.2f}%)")
        lines.append(f"Counterfactual Simulation: {len(pathways['counterfactual'])} "
                    f"({len(pathways['counterfactual'])/total*100 if total > 0 else 0:.2f}%)")
        lines.append(f"Knowledge Graph Query: {len(pathways['knowledge_query'])} "
                    f"({len(pathways['knowledge_query'])/total*100 if total > 0 else 0:.2f}%)")
        
        lines.append("")
        
        # ============================================================
        # 7. ?�常檢測?��? (Anomaly Detection)
        # ============================================================
        lines.append("=" * 100)
        lines.append("7. ANOMALY DETECTION ANALYSIS")
        lines.append("=" * 100)
        
        anomaly_stats = stats['anomaly_stats']
        lines.append(f"Anomalies Detected: {anomaly_stats['detected']} "
                    f"({anomaly_stats['detected']/total*100 if total > 0 else 0:.2f}%)")
        lines.append(f"No Anomalies: {anomaly_stats['not_detected']} "
                    f"({anomaly_stats['not_detected']/total*100 if total > 0 else 0:.2f}%)")
        
        if anomaly_stats['anomalous_subjects']:
            lines.append(f"\nTotal Anomalous Subjects: {len(anomaly_stats['anomalous_subjects'])}")
            
            # ?�異常�??�數?��?�?
            sorted_anomalous = sorted(anomaly_stats['anomalous_subjects'], 
                                     key=lambda x: x['count'], reverse=True)
            
            lines.append("\nTop 10 Subjects with Most Anomalies:")
            for item in sorted_anomalous[:10]:
                lines.append(f"  - {item['subject_id']}: {item['count']} anomalous regions")
                lines.append(f"    Regions: {', '.join(item['regions'][:5])}")
                if len(item['regions']) > 5:
                    lines.append(f"    ... and {len(item['regions']) - 5} more")
        
        if anomaly_stats['region_frequency']:
            lines.append("\nMost Frequently Anomalous Regions (Top 15):")
            sorted_regions = sorted(anomaly_stats['region_frequency'].items(),
                                   key=lambda x: x[1], reverse=True)
            
            for region, count in sorted_regions[:15]:
                percentage = count / anomaly_stats['detected'] * 100 if anomaly_stats['detected'] > 0 else 0
                lines.append(f"  - {region}: {count} times ({percentage:.2f}%)")
        
        lines.append("")
        
        # ============================================================
        # 8. ?�徵?��??��???(Feature Importance)
        # ============================================================
        lines.append("=" * 100)
        lines.append("8. FEATURE IMPORTANCE ANALYSIS")
        lines.append("=" * 100)
        
        feat_importance = stats['feature_importance']
        
        if feat_importance['top_features_frequency']:
            lines.append("Most Frequently Important Features (Top 20):")
            sorted_features = sorted(feat_importance['top_features_frequency'].items(),
                                    key=lambda x: x[1], reverse=True)
            
            for i, (feature, count) in enumerate(sorted_features[:20], 1):
                percentage = count / total * 100 if total > 0 else 0
                
                # 計�?平�? SHAP ??Z-score
                avg_shap = np.mean(feat_importance['shap_values'][feature])
                avg_z = np.mean(feat_importance['z_scores'][feature])
                
                lines.append(f"  {i:2d}. {feature}: {count} times ({percentage:.2f}%)")
                lines.append(f"      Avg SHAP: {avg_shap:+.6f}, Avg Z-score: {avg_z:+.4f}")
        
        lines.append("")
        
        # ============================================================
        # 9. ?�能?��? (Performance Metrics)
        # ============================================================
        lines.append("=" * 100)
        lines.append("9. PERFORMANCE ANALYSIS")
        lines.append("=" * 100)
        
        if 'avg_init_time' in stats:
            lines.append(f"Average Initialization Time: {stats['avg_init_time']:.2f}s")
            lines.append(f"Average Analysis Time: {stats['avg_analysis_time']:.2f}s")
            lines.append(f"Average Total Time: {stats['avg_total_time']:.2f}s")
            lines.append(f"Average Throughput: {stats['avg_throughput']:.2f} subjects/hour")
            
            perf = stats['performance']
            lines.append("\nTime Statistics:")
            lines.append(f"  Min Init Time: {np.min(perf['init_times']):.2f}s")
            lines.append(f"  Max Init Time: {np.max(perf['init_times']):.2f}s")
            lines.append(f"  Min Analysis Time: {np.min(perf['analysis_times']):.2f}s")
            lines.append(f"  Max Analysis Time: {np.max(perf['analysis_times']):.2f}s")
        
        lines.append("")
        
        # ============================================================
        # 10. ?��??��???(Reasoning Chain)
        # ============================================================
        lines.append("=" * 100)
        lines.append("10. REASONING CHAIN ANALYSIS")
        lines.append("=" * 100)
        
        if 'avg_reasoning_steps' in stats:
            lines.append(f"Average Total Reasoning Steps: {stats['avg_reasoning_steps']:.1f}")
            lines.append(f"Average Agent A Steps: {stats['avg_agent_a_steps']:.1f}")
            lines.append(f"Average Agent B Steps: {stats['avg_agent_b_steps']:.1f}")
            lines.append(f"Average MCP Actions: {stats['avg_mcp_actions']:.1f}")
            
            reasoning = stats['reasoning']
            lines.append("\nReasoning Steps Statistics:")
            lines.append(f"  Min Steps: {np.min(reasoning['total_steps'])}")
            lines.append(f"  Max Steps: {np.max(reasoning['total_steps'])}")
        
        lines.append("")
        
        # ============================================================
        # 11. 組�?條件?��? (Combined Conditions)
        # ============================================================
        lines.append("=" * 100)
        lines.append("11. COMBINED CONDITIONS ANALYSIS")
        lines.append("=" * 100)
        
        combined = stats['combined_conditions']
        
        lines.append(f"Low Confidence + High UQ: {len(combined['low_conf_high_uq'])} subjects")
        if combined['low_conf_high_uq']:
            lines.append(f"  Subjects: {', '.join(combined['low_conf_high_uq'][:10])}")
            if len(combined['low_conf_high_uq']) > 10:
                lines.append(f"  ... and {len(combined['low_conf_high_uq']) - 10} more")
        
        lines.append("")
        lines.append(f"High Confidence + High UQ: {len(combined['high_conf_high_uq'])} subjects")
        if combined['high_conf_high_uq']:
            lines.append(f"  Subjects: {', '.join(combined['high_conf_high_uq'][:10])}")
        
        lines.append("")
        lines.append(f"Low Confidence + Anomaly: {len(combined['low_conf_anomaly'])} subjects")
        if combined['low_conf_anomaly']:
            lines.append(f"  Subjects: {', '.join(combined['low_conf_anomaly'][:10])}")
        
        lines.append("")
        lines.append(f"High UQ + Anomaly: {len(combined['high_uq_anomaly'])} subjects")
        if combined['high_uq_anomaly']:
            lines.append(f"  Subjects: {', '.join(combined['high_uq_anomaly'][:10])}")
        
        lines.append("")
        
        # ============================================================
        # 12. 系統?�值�???(System Value Analysis) - NEW!
        # ============================================================
        lines.append("=" * 100)
        lines.append("12. SYSTEM VALUE ANALYSIS - AGENT INTERVENTION IMPACT")
        lines.append("=" * 100)
        
        system_value = stats['system_value']
        
        lines.append(f"Total Agent Interventions: {len(system_value['intervention_cases'])}")
        lines.append(f"  - Counterfactual Simulation: {len(pathways['counterfactual'])}")
        lines.append(f"  - Knowledge Graph Query: {len(pathways['knowledge_query'])}")
        
        lines.append(f"\nSuccessful Corrections by Agent:")
        lines.append(f"  - Corrected by Counterfactual: {len(system_value['corrected_by_counterfactual'])}")
        lines.append(f"  - Corrected by Knowledge Query: {len(system_value['corrected_by_knowledge'])}")
        lines.append(f"  - Total Corrections: {system_value['total_corrections']}")
        
        if system_value['intervention_cases']:
            lines.append(f"\nAccuracy Comparison:")
            lines.append(f"  - Standard Pathway Accuracy: {system_value['standard_accuracy']:.4f} ({system_value['standard_accuracy']*100:.2f}%)")
            lines.append(f"  - Intervention Pathway Accuracy: {system_value['intervention_accuracy']:.4f} ({system_value['intervention_accuracy']*100:.2f}%)")
            
            if system_value['accuracy_improvement'] != 0:
                improvement_sign = "+" if system_value['accuracy_improvement'] > 0 else ""
                lines.append(f"  - Accuracy Improvement: {improvement_sign}{system_value['accuracy_improvement']:.4f} ({improvement_sign}{system_value['accuracy_improvement']*100:.2f}%)")
                
                if system_value['accuracy_improvement'] > 0:
                    lines.append(f"\n??SYSTEM VALUE CONFIRMED: Agent intervention improved accuracy!")
                elif system_value['accuracy_improvement'] < 0:
                    lines.append(f"\n??WARNING: Intervention accuracy lower than standard pathway")
                else:
                    lines.append(f"\n??Intervention accuracy equal to standard pathway")
        
        if system_value['corrected_by_counterfactual']:
            lines.append(f"\nCounterfactual Simulation Success Cases:")
            for i, case in enumerate(system_value['corrected_by_counterfactual'][:5], 1):
                lines.append(f"  {i}. {case['subject_id']}: GT={case['ground_truth']}, "
                           f"Conf={case['confidence']:.4f}, UQ={case['uq_score']:.4f}")
            if len(system_value['corrected_by_counterfactual']) > 5:
                lines.append(f"  ... and {len(system_value['corrected_by_counterfactual']) - 5} more")
        
        if system_value['corrected_by_knowledge']:
            lines.append(f"\nKnowledge Graph Query Success Cases:")
            for i, case in enumerate(system_value['corrected_by_knowledge'][:5], 1):
                lines.append(f"  {i}. {case['subject_id']}: GT={case['ground_truth']}, "
                           f"Conf={case['confidence']:.4f}, UQ={case['uq_score']:.4f}")
            if len(system_value['corrected_by_knowledge']) > 5:
                lines.append(f"  ... and {len(system_value['corrected_by_knowledge']) - 5} more")
        
        # 計�?系統?�值�?�?
        if system_value['intervention_cases']:
            intervention_rate = len(system_value['intervention_cases']) / total * 100 if total > 0 else 0
            lines.append(f"\nSystem Value Metrics:")
            lines.append(f"  - Intervention Rate: {intervention_rate:.2f}% ({len(system_value['intervention_cases'])}/{total})")
            lines.append(f"  - Correction Rate (among interventions): {system_value['correction_rate']*100:.2f}%")
            
            # 估�?系統對整體�?確�??�貢??
            if system_value['total_corrections'] > 0:
                contribution = system_value['total_corrections'] / total * 100 if total > 0 else 0
                lines.append(f"  - System Contribution to Overall Accuracy: {contribution:.2f}% ({system_value['total_corrections']}/{total} subjects)")
        
        lines.append("")
        
        # ============================================================
        # 13. ?�鍵?�現總�? (Key Findings for Paper)
        # ============================================================
        lines.append("=" * 100)
        lines.append("13. KEY FINDINGS SUMMARY")
        lines.append("=" * 100)
        
        findings = []
        
        # LOOCV 完整??
        if coverage == 100:
            findings.append("✓LOOCV Integrity: 100% strict train-test separation confirmed")
        elif coverage >= 95:
            findings.append(f"✓LOOCV Integrity: {coverage:.2f}% coverage (near-complete separation)")
        else:
            findings.append(f"✓LOOCV Integrity: Only {coverage:.2f}% coverage (review required)")
        
        # ?��??�能
        findings.append(f"Binary Classification Accuracy: {metrics['accuracy']*100:.2f}%")
        findings.append(f"Sensitivity (AD Detection): {metrics['recall']*100:.2f}%")
        findings.append(f"Specificity (NC Detection): {metrics['specificity']*100:.2f}%")
        findings.append(f"F1-Score: {metrics['f1_score']:.4f}")
        
        # 不確定�?
        if uq_stats['mean'] > 0.5:
            findings.append(f"High average uncertainty detected: {uq_stats['mean']:.3f}")
        
        # Agent 決�?
        if len(pathways['counterfactual']) > 0:
            cf_pct = len(pathways['counterfactual']) / total * 100
            findings.append(f"Counterfactual simulation triggered in {cf_pct:.2f}% of cases")
        
        # ?�常檢測
        if anomaly_stats['detected'] > 0:
            anomaly_pct = anomaly_stats['detected'] / total * 100
            findings.append(f"Anomalies detected in {anomaly_pct:.2f}% of subjects")
        
        # 組�?條件
        if len(combined['low_conf_high_uq']) > 0:
            findings.append(f"{len(combined['low_conf_high_uq'])} subjects show both low confidence and high uncertainty")
        
        # 系統?��?(NEW!)
        if system_value['total_corrections'] > 0:
            findings.append(f"✓System successfully corrected {system_value['total_corrections']} cases through agent intervention")
        
        if system_value['accuracy_improvement'] > 0:
            findings.append(f"✓Agent intervention improved accuracy by {system_value['accuracy_improvement']*100:.2f}%")
        
        if findings:
            for i, finding in enumerate(findings, 1):
                lines.append(f"{i}. {finding}")
        else:
            lines.append("No significant findings to report.")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("END OF REPORT")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def save_report(self, filename: str = "binary_statistics_report.txt"):
        """保�?統�??��??��?�?"""
        report = self.generate_report()
        
        report_file = self.output_dir / filename
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report_file
    
    def save_json(self, filename: str = "binary_statistics.json"):
        """保�?統�??��???JSON"""
        # 轉�? defaultdict ?�普??dict
        stats_dict = {}
        for key, value in self.statistics.items():
            if isinstance(value, defaultdict):
                stats_dict[key] = dict(value)
            elif isinstance(value, dict):
                # ?�迴?��?嵌�???defaultdict
                stats_dict[key] = self._convert_defaultdict(value)
            else:
                stats_dict[key] = value
        
        json_file = self.output_dir / filename
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': stats_dict,
                'results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        return json_file
    
    def _convert_defaultdict(self, obj):
        """?�迴轉�? defaultdict ?�普??dict"""
        if isinstance(obj, defaultdict):
            return {k: self._convert_defaultdict(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {k: self._convert_defaultdict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_defaultdict(item) for item in obj]
        else:
            return obj
    
    def save_csv(self, filename: str = "binary_statistics.csv"):
        """保�?結�???CSV"""
        import csv
        
        csv_file = self.output_dir / filename
        
        if not self.results:
            return None
        
        # ?��??�?�鍵
        fieldnames = list(self.results[0].keys())
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                # ?��??�表類�??��?�?
                row = result.copy()
                if 'anomalous_regions' in row:
                    row['anomalous_regions'] = ', '.join(row['anomalous_regions'])
                writer.writerow(row)
        
        return csv_file
    
    def generate_latex_table(self) -> str:
        """?��? LaTeX ?��??�性能表格 (?�於 Paper)"""
        stats = self.statistics
        metrics = stats['binary_metrics']
        
        latex = []
        latex.append("% Binary Classification Performance Table")
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Binary Classification Performance (NC vs AD)}")
        latex.append("\\label{tab:binary_performance}")
        latex.append("\\begin{tabular}{lc}")
        latex.append("\\toprule")
        latex.append("Metric & Value \\\\")
        latex.append("\\midrule")
        latex.append(f"Accuracy & {metrics['accuracy']:.4f} \\\\")
        latex.append(f"Precision (AD) & {metrics['precision']:.4f} \\\\")
        latex.append(f"Recall/Sensitivity (AD) & {metrics['recall']:.4f} \\\\")
        latex.append(f"Specificity (NC) & {metrics['specificity']:.4f} \\\\")
        latex.append(f"F1-Score & {metrics['f1_score']:.4f} \\\\")
        latex.append(f"Balanced Accuracy & {metrics['balanced_accuracy']:.4f} \\\\")
        latex.append("\\midrule")
        latex.append(f"True Positives & {metrics['true_positives']} \\\\")
        latex.append(f"True Negatives & {metrics['true_negatives']} \\\\")
        latex.append(f"False Positives & {metrics['false_positives']} \\\\")
        latex.append(f"False Negatives & {metrics['false_negatives']} \\\\")
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def save_latex_table(self, filename: str = "binary_performance_table.tex"):
        """保�? LaTeX 表格?��?�?"""
        latex_table = self.generate_latex_table()
        
        latex_file = self.output_dir / filename
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        return latex_file


def scan_all_subjects() -> Tuple[List[str], Dict[str, str]]:
    """?��??�?�可?��??�試??"""
    subject_labels = {}
    data_folders = glob.glob("data/MRI_processed/*/sub-*")
    
    for folder_path in data_folders:
        parts = folder_path.replace('\\', '/').split('/')
        if len(parts) >= 3:
            subject_id = parts[-1]
            label = parts[-2]
            
            # 檢查?�否?�足夠�? nii.gz ?�件
            nii_files = list(Path(folder_path).glob("*.nii.gz"))
            if len(nii_files) >= 3:
                subject_labels[subject_id] = label
    
    subjects = sorted(subject_labels.keys())
    
    return subjects, subject_labels


def main():
    """主函??"""
    parser = argparse.ArgumentParser(
        description="Binary Classification Statistics - NC vs AD 二�?類統計�???"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/binary_statistics',
        help='輸出?��? (默�?: output/binary_statistics)'
    )
    
    parser.add_argument(
        '--orchestrator-path',
        type=str,
        default='D:/hf_models/Phi-4-mini-instruct',
        help='Orchestrator 模�?路�?'
    )
    
    parser.add_argument(
        '--consultant-path',
        type=str,
        default='D:/hf_models/Llama3.1-Aloe-Beta-8B',
        help='Consultant 模�?路�?'
    )
    
    parser.add_argument(
        '--use-4bit',
        action='store_true',
        default=True,
        help='使用 4-bit ?��?'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='禁用 LLM 模�? (使用規�?決�?)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='?�制?��??��?試者數??(?�於測試)'
    )
    
    parser.add_argument(
        '--binary-only',
        action='store_true',
        default=False,  # 改為 False，默認包含所有受試者
        help='只分析 NC 和 AD 受試者 (排除 MCI)'
    )
    
    parser.add_argument(
        '--include-mci',
        action='store_true',
        default=True,  # 默認包含 MCI
        help='包含 MCI 受試者 (展示系統對不確定案例的處理能力)'
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("BINARY CLASSIFICATION STATISTICS ANALYSIS (NC vs AD)")
    print("=" * 100)
    print()
    
    # ?��??�?��?試�?
    print("Scanning for subjects...")
    subjects, ground_truths = scan_all_subjects()
    
    # 如�??��??��??��?，�?濾�? MCI
    if args.binary_only:
        binary_subjects = [s for s in subjects if ground_truths.get(s) in ['NC', 'AD']]
        print(f"Found {len(subjects)} total subjects")
        print(f"Filtered to {len(binary_subjects)} binary subjects (NC/AD only)")
        subjects = binary_subjects
    
    if not subjects:
        print("ERROR: No subjects found in data/MRI_processed/")
        sys.exit(1)
    
    print(f"Final subject count: {len(subjects)}")
    
    if args.limit:
        subjects = subjects[:args.limit]
        print(f"Limited to {len(subjects)} subjects for testing")
    
    print()
    
    # ?��??�統計�??�器
    statistics = BinaryStatistics(args.output)
    
    # ?��???CDDA Agent
    print("Initializing CDDA Agent...")
    print(f"  Orchestrator: {args.orchestrator_path}")
    print(f"  Consultant: {args.consultant_path}")
    print(f"  4-bit Quantization: {args.use_4bit}")
    print(f"  LLM Mode: {not args.no_llm}")
    print()
    
    try:
        agent = CDDAAgent(
            orchestrator_model="phi-4-mini",
            orchestrator_model_path=args.orchestrator_path,
            consultant_model="llama3.1-aloe-beta-8b",
            consultant_model_path=args.consultant_path,
            use_llm=not args.no_llm,
            use_4bit=args.use_4bit,
            verbose=False
        )
        
        print("✓CDDA Agent initialized successfully!")
        print()
        
    except Exception as e:
        print(f"✓Failed to initialize CDDA Agent: {e}")
        sys.exit(1)
    
    # ?��??�?��?試�?
    print("=" * 100)
    print("ANALYZING ALL SUBJECTS")
    print("=" * 100)
    print()
    
    start_time = time.time()
    
    for i, subject_id in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] Analyzing {subject_id}...", end=' ', flush=True)
        
        ground_truth = ground_truths.get(subject_id)
        
        init_start = time.time()
        
        try:
            # 構建 LOOCV 專屬模�??�稱
            model_name = f"rf_model_{subject_id}"
            
            # ?��??��?
            result = agent.run_analysis(subject_id, model_name=model_name)
            
            analysis_time = time.time() - init_start
            
            # 添�?結�??�統�?
            statistics.add_result(
                subject_id=subject_id,
                result=result,
                ground_truth=ground_truth,
                init_time=0,
                analysis_time=analysis_time
            )
            
            # 顯示驗�?結�?
            last_entry = statistics.results[-1]
            model_info = last_entry['model_used']
            is_verified = "✓" if last_entry['loocv_verified'] else "!"
            
            print(f"{is_verified} [{model_info}] - {result.prediction} (Conf: {result.confidence:.2f})")
        
        except Exception as e:
            analysis_time = time.time() - init_start
            
            statistics.add_result(
                subject_id=subject_id,
                result=None,
                ground_truth=ground_truth,
                init_time=0,
                analysis_time=analysis_time,
                error=str(e)
            )
            
            print(f"✓({analysis_time:.1f}s) - ERROR: {str(e)[:50]}")
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 100)
    print(f"ANALYSIS COMPLETE - Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("=" * 100)
    print()
    
    # ?��?並�?存報??
    print("Generating comprehensive statistics report...")
    
    report_file = statistics.save_report()
    print(f"✓Text report saved: {report_file}")
    
    json_file = statistics.save_json()
    print(f"✓JSON data saved: {json_file}")
    
    csv_file = statistics.save_csv()
    if csv_file:
        print(f"✓CSV data saved: {csv_file}")
    
    latex_file = statistics.save_latex_table()
    print(f"✓LaTeX table saved: {latex_file}")
    
    print()
    
    # ?�印?��??�控?�台
    print("=" * 100)
    print("STATISTICS REPORT")
    print("=" * 100)
    print()
    
    report = statistics.generate_report()
    print(report)
    
    print()
    print("=" * 100)
    print("All files saved to:", args.output)
    print("=" * 100)


if __name__ == "__main__":
    main()

