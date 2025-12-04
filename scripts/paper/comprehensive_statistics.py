#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDDA Comprehensive Statistics Script

完整的統計分析腳本，自動掃描所有受試者並生成詳細統計報告。

使用方法:
    python scripts/comprehensive_statistics.py
    python scripts/comprehensive_statistics.py --output output/statistics
    python scripts/comprehensive_statistics.py --ground-truth-file ground_truth.json
"""

import sys
import re
import argparse
import json
import glob
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


class ComprehensiveStatistics:
    """綜合統計分析器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 統計數據容器
        self.results = []
        self.statistics = {
            # 基本統計
            'total_subjects': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,

            # [關鍵修復] 模型驗證統計
            'model_usage': {
                'loocv_verified': 0,      # 正確使用專屬模型
                'fallback_global': 0,     # 使用了通用模型
                'unknown': 0              # 無法判斷
            },
            'model_verification_details': [],
            
            # 預測統計
            'predictions': defaultdict(int),
            'ground_truth_distribution': defaultdict(int),
            
            # 信心度統計
            'confidence_ranges': {
                'very_high': 0,      # >= 0.9
                'high': 0,           # 0.8 - 0.9
                'medium': 0,         # 0.6 - 0.8
                'low': 0,            # 0.4 - 0.6
                'very_low': 0        # < 0.4
            },
            'low_confidence_subjects': [],  # < 0.6
            
            # 不確定性統計
            'uq_ranges': {
                'very_high': 0,      # >= 0.9
                'high': 0,           # 0.8 - 0.9
                'medium': 0,         # 0.5 - 0.8
                'low': 0,            # 0.3 - 0.5
                'very_low': 0        # < 0.3
            },
            'high_uq_subjects': [],  # > 0.8
            
            # Agent 決策統計
            'agent_decisions': defaultdict(int),
            'counterfactual_triggered': [],
            'knowledge_query_triggered': [],
            'standard_pathway': [],
            
            # 異常檢測統計
            'anomaly_detected': 0,
            'no_anomaly': 0,
            'anomalous_subjects': [],
            'anomalous_regions_frequency': defaultdict(int),
            
            # 準確性統計 (如果有真實標籤)
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'accuracy_by_class': defaultdict(lambda: {'correct': 0, 'total': 0}),
            
            # 性能統計
            'init_times': [],
            'analysis_times': [],
            'total_times': [],
            
            # 推理鏈統計
            'reasoning_steps': [],
            'agent_a_steps': [],
            'agent_b_steps': [],
            'mcp_actions_count': [],
            
            # 特徵統計
            'top_features_frequency': defaultdict(int),
            'shap_values': defaultdict(list),
            'z_scores': defaultdict(list),
            
            # 組合條件統計
            'low_confidence_high_uq': [],      # 信心度 < 0.6 且 UQ > 0.8
            'high_confidence_high_uq': [],     # 信心度 >= 0.8 且 UQ > 0.8
            'low_confidence_anomaly': [],      # 信心度 < 0.6 且有異常
            'high_uq_anomaly': [],             # UQ > 0.8 且有異常
            
            # 錯誤統計
            'errors': [],
            'fallback_used': defaultdict(int)
        }
    def _verify_model_usage(self, subject_id: str, reasoning_chain: List[str]) -> Tuple[str, str]:
        """
        [New] 解析 Log，驗證是否使用了正確的模型
        """
        # 你的 Agent Log 格式: "Inference complete for sub-001 using rf_model_sub-001.joblib"
        full_log = " ".join(reasoning_chain)
        
        # 寬鬆匹配，抓取 .joblib 結尾的檔案名稱
        match = re.search(r"using ([\w\-\.]+\.joblib)", full_log)
        
        if match:
            model_name = match.group(1)
            # 檢查檔名是否包含該 subject_id
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
        """添加單個受試者的分析結果"""
        
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
        
        # 提取基本信息
        prediction = result.prediction
        confidence = result.confidence
        uq_score = result.uq_score
        agent_decision = result.agent_decision

        # [New] 執行模型驗證
        verification_status, used_model_name = self._verify_model_usage(subject_id, result.reasoning_chain)
        self.statistics['model_usage'][verification_status] += 1
        
        # 記錄驗證細節 (這可以用來放在 Appendix)
        self.statistics['model_verification_details'].append({
            'subject_id': subject_id,
            'status': verification_status,
            'model_used': used_model_name
        })
        
        # 基本統計
        self.statistics['predictions'][prediction] += 1
        
        if ground_truth:
            self.statistics['ground_truth_distribution'][ground_truth] += 1
        
        # 信心度統計
        if confidence >= 0.9:
            self.statistics['confidence_ranges']['very_high'] += 1
        elif confidence >= 0.8:
            self.statistics['confidence_ranges']['high'] += 1
        elif confidence >= 0.6:
            self.statistics['confidence_ranges']['medium'] += 1
        elif confidence >= 0.4:
            self.statistics['confidence_ranges']['low'] += 1
        else:
            self.statistics['confidence_ranges']['very_low'] += 1
        
        if confidence < 0.6:
            self.statistics['low_confidence_subjects'].append({
                'subject_id': subject_id,
                'confidence': confidence,
                'prediction': prediction,
                'ground_truth': ground_truth
            })
        
        # 不確定性統計
        if uq_score >= 0.9:
            self.statistics['uq_ranges']['very_high'] += 1
        elif uq_score >= 0.8:
            self.statistics['uq_ranges']['high'] += 1
        elif uq_score >= 0.5:
            self.statistics['uq_ranges']['medium'] += 1
        elif uq_score >= 0.3:
            self.statistics['uq_ranges']['low'] += 1
        else:
            self.statistics['uq_ranges']['very_low'] += 1
        
        if uq_score > 0.8:
            self.statistics['high_uq_subjects'].append({
                'subject_id': subject_id,
                'uq_score': uq_score,
                'prediction': prediction,
                'ground_truth': ground_truth
            })
        
        # Agent 決策統計
        self.statistics['agent_decisions'][agent_decision] += 1
        
        if 'SIMULATION' in agent_decision:
            self.statistics['counterfactual_triggered'].append(subject_id)
        elif 'ANOMALY' in agent_decision or 'INVESTIGATION' in agent_decision:
            self.statistics['knowledge_query_triggered'].append(subject_id)
        else:
            self.statistics['standard_pathway'].append(subject_id)
        
        # 異常檢測統計
        context_object = result.context_object
        anomaly_status = context_object.diagnostic_report.anomaly_status
        
        if anomaly_status.has_anomaly:
            self.statistics['anomaly_detected'] += 1
            self.statistics['anomalous_subjects'].append({
                'subject_id': subject_id,
                'regions': anomaly_status.anomalous_regions,
                'count': len(anomaly_status.anomalous_regions)
            })
            
            # 統計異常區域頻率
            for region in anomaly_status.anomalous_regions:
                self.statistics['anomalous_regions_frequency'][region] += 1
        else:
            self.statistics['no_anomaly'] += 1
        
        # 準確性統計
        if ground_truth:
            correct = (prediction == ground_truth)
            
            if correct:
                self.statistics['correct_predictions'] += 1
            else:
                self.statistics['incorrect_predictions'] += 1
            
            # 按類別統計準確率
            self.statistics['accuracy_by_class'][ground_truth]['total'] += 1
            if correct:
                self.statistics['accuracy_by_class'][ground_truth]['correct'] += 1
        
        # 性能統計
        self.statistics['init_times'].append(init_time)
        self.statistics['analysis_times'].append(analysis_time)
        self.statistics['total_times'].append(init_time + analysis_time)
        
        # 推理鏈統計
        reasoning_chain = result.reasoning_chain
        self.statistics['reasoning_steps'].append(len(reasoning_chain))
        
        agent_a_steps = sum(1 for step in reasoning_chain if '[Agent A]' in step)
        agent_b_steps = sum(1 for step in reasoning_chain if '[Agent B]' in step)
        mcp_actions = len(context_object.mcp_actions)
        
        self.statistics['agent_a_steps'].append(agent_a_steps)
        self.statistics['agent_b_steps'].append(agent_b_steps)
        self.statistics['mcp_actions_count'].append(mcp_actions)
        
        # 特徵統計
        top_features = context_object.diagnostic_report.top_features[:10]
        
        for feat in top_features:
            roi_name = feat.roi_name if hasattr(feat, 'roi_name') else 'Unknown'
            shap_value = feat.shap_value if hasattr(feat, 'shap_value') else 0
            z_score = feat.z_score if hasattr(feat, 'z_score') else 0
            
            self.statistics['top_features_frequency'][roi_name] += 1
            self.statistics['shap_values'][roi_name].append(shap_value)
            self.statistics['z_scores'][roi_name].append(z_score)
        
        # 組合條件統計
        if confidence < 0.6 and uq_score > 0.8:
            self.statistics['low_confidence_high_uq'].append(subject_id)
        
        if confidence >= 0.8 and uq_score > 0.8:
            self.statistics['high_confidence_high_uq'].append(subject_id)
        
        if confidence < 0.6 and anomaly_status.has_anomaly:
            self.statistics['low_confidence_anomaly'].append(subject_id)
        
        if uq_score > 0.8 and anomaly_status.has_anomaly:
            self.statistics['high_uq_anomaly'].append(subject_id)
        
        # 檢查錯誤註釋
        if hasattr(context_object, 'errors') and context_object.errors:
            for error in context_object.errors:
                error_type = error.get('type', 'Unknown')
                self.statistics['fallback_used'][error_type] += 1
        
        # 保存完整結果
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
        """計算統計指標"""
        stats = self.statistics
        
        # 計算平均值
        if stats['init_times']:
            stats['avg_init_time'] = sum(stats['init_times']) / len(stats['init_times'])
            stats['avg_analysis_time'] = sum(stats['analysis_times']) / len(stats['analysis_times'])
            stats['avg_total_time'] = sum(stats['total_times']) / len(stats['total_times'])
            stats['avg_throughput'] = 3600 / stats['avg_analysis_time'] if stats['avg_analysis_time'] > 0 else 0
        
        if stats['reasoning_steps']:
            stats['avg_reasoning_steps'] = sum(stats['reasoning_steps']) / len(stats['reasoning_steps'])
            stats['avg_agent_a_steps'] = sum(stats['agent_a_steps']) / len(stats['agent_a_steps'])
            stats['avg_agent_b_steps'] = sum(stats['agent_b_steps']) / len(stats['agent_b_steps'])
            stats['avg_mcp_actions'] = sum(stats['mcp_actions_count']) / len(stats['mcp_actions_count'])
        
        # 計算準確率
        total_with_gt = stats['correct_predictions'] + stats['incorrect_predictions']
        if total_with_gt > 0:
            stats['overall_accuracy'] = stats['correct_predictions'] / total_with_gt * 100
        else:
            stats['overall_accuracy'] = None
        
        # 計算各類別準確率
        for class_name, class_stats in stats['accuracy_by_class'].items():
            if class_stats['total'] > 0:
                class_stats['accuracy'] = class_stats['correct'] / class_stats['total'] * 100
        
        # 計算百分比
        total = stats['successful_analyses']
        if total > 0:
            stats['low_confidence_percentage'] = len(stats['low_confidence_subjects']) / total * 100
            stats['high_uq_percentage'] = len(stats['high_uq_subjects']) / total * 100
            stats['anomaly_percentage'] = stats['anomaly_detected'] / total * 100
            stats['counterfactual_percentage'] = len(stats['counterfactual_triggered']) / total * 100
            stats['knowledge_query_percentage'] = len(stats['knowledge_query_triggered']) / total * 100
    
    def generate_report(self) -> str:
        """生成詳細統計報告"""
        self.calculate_statistics()
        
        stats = self.statistics
        lines = []
        
        # 標題
        lines.append("=" * 100)
        lines.append("CDDA COMPREHENSIVE STATISTICS REPORT".center(100))
        lines.append("=" * 100)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 0. [New] LOOCV 完整性檢查
        lines.append("=" * 100)
        lines.append("0. LOOCV INTEGRITY CHECK")
        lines.append("=" * 100)
        lines.append(f"LOOCV Verified Models: {stats['model_usage']['loocv_verified']}")
        lines.append(f"Global Fallback Models: {stats['model_usage']['fallback_global']}")
        lines.append(f"Unknown/Unverified: {stats['model_usage']['unknown']}")
        
        coverage = stats.get('loocv_coverage_percentage', 0)
        if coverage == 100:
            lines.append("STATUS: PASSED (100% Strict Separation)")
        else:
            lines.append(f"STATUS: WARNING ({coverage:.2f}% coverage)")
            
        lines.append("")

        # 1. 總體概況
        lines.append("=" * 100)
        lines.append("1. OVERALL SUMMARY")
        lines.append("=" * 100)
        lines.append(f"Total Subjects Scanned: {stats['total_subjects']}")
        lines.append(f"Successful Analyses: {stats['successful_analyses']}")
        lines.append(f"Failed Analyses: {stats['failed_analyses']}")
        
        if stats['failed_analyses'] > 0:
            success_rate = stats['successful_analyses'] / stats['total_subjects'] * 100
            lines.append(f"Success Rate: {success_rate:.2f}%")
        
        lines.append("")
        
        # 2. 預測分布
        lines.append("=" * 100)
        lines.append("2. PREDICTION DISTRIBUTION")
        lines.append("=" * 100)
        
        for pred, count in sorted(stats['predictions'].items()):
            percentage = count / stats['successful_analyses'] * 100 if stats['successful_analyses'] > 0 else 0
            lines.append(f"{pred}: {count} ({percentage:.2f}%)")
        
        lines.append("")
        
        # 3. 真實標籤分布 (如果有)
        if stats['ground_truth_distribution']:
            lines.append("=" * 100)
            lines.append("3. GROUND TRUTH DISTRIBUTION")
            lines.append("=" * 100)
            
            for gt, count in sorted(stats['ground_truth_distribution'].items()):
                percentage = count / sum(stats['ground_truth_distribution'].values()) * 100
                lines.append(f"{gt}: {count} ({percentage:.2f}%)")
            
            lines.append("")
        
        # 4. 準確性分析
        if stats['overall_accuracy'] is not None:
            lines.append("=" * 100)
            lines.append("4. ACCURACY ANALYSIS")
            lines.append("=" * 100)
            lines.append(f"Overall Accuracy: {stats['overall_accuracy']:.2f}%")
            lines.append(f"Correct Predictions: {stats['correct_predictions']}")
            lines.append(f"Incorrect Predictions: {stats['incorrect_predictions']}")
            lines.append("")
            
            lines.append("Accuracy by Class:")
            for class_name in sorted(stats['accuracy_by_class'].keys()):
                class_stats = stats['accuracy_by_class'][class_name]
                lines.append(f"  {class_name}: {class_stats['accuracy']:.2f}% "
                           f"({class_stats['correct']}/{class_stats['total']})")
            
            lines.append("")
        
        # 5. 信心度分析
        lines.append("=" * 100)
        lines.append("5. CONFIDENCE ANALYSIS")
        lines.append("=" * 100)
        
        total = stats['successful_analyses']
        lines.append("Confidence Distribution:")
        for range_name, count in stats['confidence_ranges'].items():
            percentage = count / total * 100 if total > 0 else 0
            lines.append(f"  {range_name.replace('_', ' ').title()}: {count} ({percentage:.2f}%)")
        
        lines.append("")
        lines.append(f"Low Confidence Subjects (< 0.6): {len(stats['low_confidence_subjects'])} "
                    f"({stats.get('low_confidence_percentage', 0):.2f}%)")
        
        if stats['low_confidence_subjects']:
            lines.append("\nLow Confidence Details:")
            for item in stats['low_confidence_subjects'][:10]:  # 只顯示前 10 個
                lines.append(f"  - {item['subject_id']}: Confidence={item['confidence']:.4f}, "
                           f"Prediction={item['prediction']}, GT={item.get('ground_truth', 'N/A')}")
            
            if len(stats['low_confidence_subjects']) > 10:
                lines.append(f"  ... and {len(stats['low_confidence_subjects']) - 10} more")
        
        lines.append("")
        
        # 6. 不確定性分析
        lines.append("=" * 100)
        lines.append("6. UNCERTAINTY ANALYSIS")
        lines.append("=" * 100)
        
        lines.append("Uncertainty (UQ) Distribution:")
        for range_name, count in stats['uq_ranges'].items():
            percentage = count / total * 100 if total > 0 else 0
            lines.append(f"  {range_name.replace('_', ' ').title()}: {count} ({percentage:.2f}%)")
        
        lines.append("")
        lines.append(f"High Uncertainty Subjects (> 0.8): {len(stats['high_uq_subjects'])} "
                    f"({stats.get('high_uq_percentage', 0):.2f}%)")
        
        if stats['high_uq_subjects']:
            lines.append("\nHigh Uncertainty Details:")
            for item in stats['high_uq_subjects'][:10]:
                lines.append(f"  - {item['subject_id']}: UQ={item['uq_score']:.4f}, "
                           f"Prediction={item['prediction']}, GT={item.get('ground_truth', 'N/A')}")
            
            if len(stats['high_uq_subjects']) > 10:
                lines.append(f"  ... and {len(stats['high_uq_subjects']) - 10} more")
        
        lines.append("")
        
        # 7. Agent 決策分析
        lines.append("=" * 100)
        lines.append("7. AGENT DECISION ANALYSIS")
        lines.append("=" * 100)
        
        lines.append("Decision Distribution:")
        for decision, count in sorted(stats['agent_decisions'].items()):
            percentage = count / total * 100 if total > 0 else 0
            lines.append(f"  {decision}: {count} ({percentage:.2f}%)")
        
        lines.append("")
        lines.append(f"Counterfactual Simulation Triggered: {len(stats['counterfactual_triggered'])} "
                    f"({stats.get('counterfactual_percentage', 0):.2f}%)")
        lines.append(f"Knowledge Graph Query Triggered: {len(stats['knowledge_query_triggered'])} "
                    f"({stats.get('knowledge_query_percentage', 0):.2f}%)")
        lines.append(f"Standard Pathway: {len(stats['standard_pathway'])} "
                    f"({len(stats['standard_pathway'])/total*100 if total > 0 else 0:.2f}%)")
        
        lines.append("")
        
        # 8. 異常檢測分析
        lines.append("=" * 100)
        lines.append("8. ANOMALY DETECTION ANALYSIS")
        lines.append("=" * 100)
        
        lines.append(f"Anomaly Detected: {stats['anomaly_detected']} "
                    f"({stats.get('anomaly_percentage', 0):.2f}%)")
        lines.append(f"No Anomaly: {stats['no_anomaly']} "
                    f"({stats['no_anomaly']/total*100 if total > 0 else 0:.2f}%)")
        
        if stats['anomalous_subjects']:
            lines.append(f"\nTotal Anomalous Subjects: {len(stats['anomalous_subjects'])}")
            lines.append("\nTop 10 Subjects with Most Anomalies:")
            
            sorted_anomalous = sorted(stats['anomalous_subjects'], 
                                     key=lambda x: x['count'], reverse=True)
            
            for item in sorted_anomalous[:10]:
                lines.append(f"  - {item['subject_id']}: {item['count']} anomalous regions")
                lines.append(f"    Regions: {', '.join(item['regions'][:5])}")
                if len(item['regions']) > 5:
                    lines.append(f"    ... and {len(item['regions']) - 5} more")
        
        if stats['anomalous_regions_frequency']:
            lines.append("\nMost Frequently Anomalous Regions:")
            sorted_regions = sorted(stats['anomalous_regions_frequency'].items(),
                                   key=lambda x: x[1], reverse=True)
            
            for region, count in sorted_regions[:15]:
                percentage = count / stats['anomaly_detected'] * 100 if stats['anomaly_detected'] > 0 else 0
                lines.append(f"  - {region}: {count} times ({percentage:.2f}%)")
        
        lines.append("")
        
        # 9. 組合條件分析
        lines.append("=" * 100)
        lines.append("9. COMBINED CONDITIONS ANALYSIS")
        lines.append("=" * 100)
        
        lines.append(f"Low Confidence + High UQ: {len(stats['low_confidence_high_uq'])} subjects")
        if stats['low_confidence_high_uq']:
            lines.append(f"  Subjects: {', '.join(stats['low_confidence_high_uq'][:10])}")
            if len(stats['low_confidence_high_uq']) > 10:
                lines.append(f"  ... and {len(stats['low_confidence_high_uq']) - 10} more")
        
        lines.append("")
        lines.append(f"High Confidence + High UQ: {len(stats['high_confidence_high_uq'])} subjects")
        if stats['high_confidence_high_uq']:
            lines.append(f"  Subjects: {', '.join(stats['high_confidence_high_uq'][:10])}")
        
        lines.append("")
        lines.append(f"Low Confidence + Anomaly: {len(stats['low_confidence_anomaly'])} subjects")
        if stats['low_confidence_anomaly']:
            lines.append(f"  Subjects: {', '.join(stats['low_confidence_anomaly'][:10])}")
        
        lines.append("")
        lines.append(f"High UQ + Anomaly: {len(stats['high_uq_anomaly'])} subjects")
        if stats['high_uq_anomaly']:
            lines.append(f"  Subjects: {', '.join(stats['high_uq_anomaly'][:10])}")
        
        lines.append("")
        
        # 10. 特徵重要性分析
        lines.append("=" * 100)
        lines.append("10. FEATURE IMPORTANCE ANALYSIS")
        lines.append("=" * 100)
        
        if stats['top_features_frequency']:
            lines.append("Most Frequently Important Features (Top 20):")
            sorted_features = sorted(stats['top_features_frequency'].items(),
                                    key=lambda x: x[1], reverse=True)
            
            for i, (feature, count) in enumerate(sorted_features[:20], 1):
                percentage = count / total * 100 if total > 0 else 0
                
                # 計算平均 SHAP 和 Z-score
                avg_shap = sum(stats['shap_values'][feature]) / len(stats['shap_values'][feature])
                avg_z = sum(stats['z_scores'][feature]) / len(stats['z_scores'][feature])
                
                lines.append(f"  {i:2d}. {feature}: {count} times ({percentage:.2f}%)")
                lines.append(f"      Avg SHAP: {avg_shap:+.6f}, Avg Z-score: {avg_z:+.4f}")
        
        lines.append("")
        
        # 11. 性能分析
        lines.append("=" * 100)
        lines.append("11. PERFORMANCE ANALYSIS")
        lines.append("=" * 100)
        
        if 'avg_init_time' in stats:
            lines.append(f"Average Initialization Time: {stats['avg_init_time']:.2f}s")
            lines.append(f"Average Analysis Time: {stats['avg_analysis_time']:.2f}s")
            lines.append(f"Average Total Time: {stats['avg_total_time']:.2f}s")
            lines.append(f"Average Throughput: {stats['avg_throughput']:.2f} subjects/hour")
            
            lines.append("")
            lines.append("Time Statistics:")
            lines.append(f"  Min Init Time: {min(stats['init_times']):.2f}s")
            lines.append(f"  Max Init Time: {max(stats['init_times']):.2f}s")
            lines.append(f"  Min Analysis Time: {min(stats['analysis_times']):.2f}s")
            lines.append(f"  Max Analysis Time: {max(stats['analysis_times']):.2f}s")
        
        lines.append("")
        
        # 12. 推理鏈分析
        lines.append("=" * 100)
        lines.append("12. REASONING CHAIN ANALYSIS")
        lines.append("=" * 100)
        
        if 'avg_reasoning_steps' in stats:
            lines.append(f"Average Total Reasoning Steps: {stats['avg_reasoning_steps']:.1f}")
            lines.append(f"Average Agent A Steps: {stats['avg_agent_a_steps']:.1f}")
            lines.append(f"Average Agent B Steps: {stats['avg_agent_b_steps']:.1f}")
            lines.append(f"Average MCP Actions: {stats['avg_mcp_actions']:.1f}")
            
            lines.append("")
            lines.append("Reasoning Steps Statistics:")
            lines.append(f"  Min Steps: {min(stats['reasoning_steps'])}")
            lines.append(f"  Max Steps: {max(stats['reasoning_steps'])}")
        
        lines.append("")
        
        # 13. 錯誤與回退分析
        if stats['errors'] or stats['fallback_used']:
            lines.append("=" * 100)
            lines.append("13. ERROR AND FALLBACK ANALYSIS")
            lines.append("=" * 100)
            
            if stats['errors']:
                lines.append(f"Total Errors: {len(stats['errors'])}")
                lines.append("\nError Details:")
                for error in stats['errors'][:10]:
                    lines.append(f"  - {error['subject_id']}: {error['error']}")
                
                if len(stats['errors']) > 10:
                    lines.append(f"  ... and {len(stats['errors']) - 10} more")
                
                lines.append("")
            
            if stats['fallback_used']:
                lines.append("Fallback Mechanisms Used:")
                for error_type, count in sorted(stats['fallback_used'].items()):
                    lines.append(f"  - {error_type}: {count} times")
                
                lines.append("")
        
        # 14. 關鍵發現總結
        lines.append("=" * 100)
        lines.append("14. KEY FINDINGS SUMMARY")
        lines.append("=" * 100)
        
        # 自動生成關鍵發現
        findings = []

        if coverage == 100:
            findings.append("LOOCV Integrity Confirmed: 100% of subjects used strictly separated models.")
        
        if stats['overall_accuracy'] is not None:
            findings.append(f"Overall system accuracy: {stats['overall_accuracy']:.2f}%")
        
        if stats.get('low_confidence_percentage', 0) > 20:
            findings.append(f"High proportion of low confidence predictions: "
                          f"{stats['low_confidence_percentage']:.2f}%")
        
        if stats.get('high_uq_percentage', 0) > 30:
            findings.append(f"Significant uncertainty detected in {stats['high_uq_percentage']:.2f}% of cases")
        
        if stats.get('counterfactual_percentage', 0) > 0:
            findings.append(f"Counterfactual simulation triggered in {stats['counterfactual_percentage']:.2f}% of cases")
        
        if stats.get('anomaly_percentage', 0) > 0:
            findings.append(f"Anomalies detected in {stats['anomaly_percentage']:.2f}% of subjects")
        
        if len(stats['low_confidence_high_uq']) > 0:
            findings.append(f"{len(stats['low_confidence_high_uq'])} subjects show both low confidence and high uncertainty")
        
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
    
    def save_report(self, filename: str = "comprehensive_statistics_report.txt"):
        """保存統計報告到文件"""
        report = self.generate_report()
        
        report_file = self.output_dir / filename
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report_file
    
    def save_json(self, filename: str = "comprehensive_statistics.json"):
        """保存統計數據為 JSON"""
        # 轉換 defaultdict 為普通 dict
        stats_dict = {}
        for key, value in self.statistics.items():
            if isinstance(value, defaultdict):
                stats_dict[key] = dict(value)
            else:
                stats_dict[key] = value
        
        json_file = self.output_dir / filename
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': stats_dict,
                'results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        return json_file
    
    def save_csv(self, filename: str = "comprehensive_statistics.csv"):
        """保存結果為 CSV"""
        import csv
        
        csv_file = self.output_dir / filename
        
        if not self.results:
            return None
        
        # 獲取所有鍵
        fieldnames = list(self.results[0].keys())
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                # 處理列表類型的字段
                row = result.copy()
                if 'anomalous_regions' in row:
                    row['anomalous_regions'] = ', '.join(row['anomalous_regions'])
                writer.writerow(row)
        
        return csv_file


def scan_all_subjects() -> Tuple[List[str], Dict[str, str]]:
    """掃描所有可用的受試者"""
    subject_labels = {}
    data_folders = glob.glob("data/MRI_processed/*/sub-*")
    
    for folder_path in data_folders:
        parts = folder_path.replace('\\', '/').split('/')
        if len(parts) >= 3:
            subject_id = parts[-1]
            label = parts[-2]
            
            # 檢查是否有足夠的 nii.gz 文件
            nii_files = list(Path(folder_path).glob("*.nii.gz"))
            if len(nii_files) >= 3:
                subject_labels[subject_id] = label
    
    subjects = sorted(subject_labels.keys())
    
    return subjects, subject_labels


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="CDDA Comprehensive Statistics - 完整統計分析所有受試者"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/comprehensive_statistics',
        help='輸出目錄 (默認: output/comprehensive_statistics)'
    )
    
    parser.add_argument(
        '--orchestrator-path',
        type=str,
        default='D:/hf_models/Phi-4-mini-instruct',
        help='Orchestrator 模型路徑'
    )
    
    parser.add_argument(
        '--consultant-path',
        type=str,
        default='D:/hf_models/Llama3.1-Aloe-Beta-8B',
        help='Consultant 模型路徑'
    )
    
    parser.add_argument(
        '--use-4bit',
        action='store_true',
        default=True,
        help='使用 4-bit 量化'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='禁用 LLM 模式 (使用規則決策)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='限制分析的受試者數量 (用於測試)'
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("CDDA COMPREHENSIVE STATISTICS ANALYSIS")
    print("=" * 100)
    print()
    
    # 掃描所有受試者
    print("Scanning for subjects...")
    subjects, ground_truths = scan_all_subjects()
    
    if not subjects:
        print("ERROR: No subjects found in data/MRI_processed/")
        sys.exit(1)
    
    print(f"Found {len(subjects)} subjects")
    
    if args.limit:
        subjects = subjects[:args.limit]
        print(f"Limited to {len(subjects)} subjects for testing")
    
    print()
    
    # 初始化統計分析器
    statistics = ComprehensiveStatistics(args.output)
    
    # 初始化 CDDA Agent
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
        
        print("✓ CDDA Agent initialized successfully!")
        print()
        
    except Exception as e:
        print(f"✗ Failed to initialize CDDA Agent: {e}")
        sys.exit(1)
    
    # 分析所有受試者
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
            # 構建 LOOCV 專屬模型名稱
            # 例如: sub-001 → rf_model_sub-001
            model_name = f"rf_model_{subject_id}"
            
            # 運行分析，傳入對應的模型名稱
            result = agent.run_analysis(subject_id, model_name=model_name)
            
            analysis_time = time.time() - init_start
            
            # 添加結果到統計
            statistics.add_result(
                subject_id=subject_id,
                result=result,
                ground_truth=ground_truth,
                init_time=0,  # 只有第一次需要初始化
                analysis_time=analysis_time
            )
            
            # 在 Console 顯示驗證結果
            # 我們需要從 statistics.results 最後一筆拿剛剛算出來的 model_used
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
            
            print(f"✗ ({analysis_time:.1f}s) - ERROR: {str(e)[:50]}\n")
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 100)
    print(f"ANALYSIS COMPLETE - Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("=" * 100)
    print()
    
    # 生成並保存報告
    print("Generating comprehensive statistics report...")
    
    report_file = statistics.save_report()
    print(f"✓ Text report saved: {report_file}")
    
    json_file = statistics.save_json()
    print(f"✓ JSON data saved: {json_file}")
    
    csv_file = statistics.save_csv()
    if csv_file:
        print(f"✓ CSV data saved: {csv_file}")
    
    print()
    
    # 打印報告到控制台
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
