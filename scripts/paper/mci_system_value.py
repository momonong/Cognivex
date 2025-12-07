#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCI System Value Analysis - 展示系統對 MCI 的處理能力

MCI (Mild Cognitive Impairment) 是系統價值的最佳展示場景：
1. MCI 沒有專屬的 LOOCV 模型（使用通用二分類模型）
2. MCI 介於 NC 和 AD 之間，具有高度不確定性
3. 系統的 Agent 介入機制應該在 MCI 案例中頻繁觸發
4. 這正是展示不確定性量化和自適應決策價值的最佳場景

使用方法:
    python scripts/paper/mci_system_value.py
    python scripts/paper/mci_system_value.py --limit 10
"""

import sys
import argparse
import json
import glob
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


class MCISystemValueAnalyzer:
    """MCI 系統價值分析器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.mci_results = []
        self.nc_ad_results = []
        
        self.statistics = {
            # MCI 統計
            'mci_stats': {
                'total': 0,
                'predictions': defaultdict(int),
                'mean_confidence': 0.0,
                'mean_uq': 0.0,
                'intervention_rate': 0.0,
                'high_uq_rate': 0.0,
                'agent_decisions': defaultdict(int),
                'cases': []
            },
            
            # NC/AD 統計（對比組）
            'nc_ad_stats': {
                'total': 0,
                'accuracy': 0.0,
                'mean_confidence': 0.0,
                'mean_uq': 0.0,
                'intervention_rate': 0.0,
                'high_uq_rate': 0.0
            },
            
            # 系統價值指標
            'system_value': {
                'mci_intervention_cases': [],
                'mci_high_confidence_ad': [],  # 高信心預測為 AD 的 MCI
                'mci_high_confidence_nc': [],  # 高信心預測為 NC 的 MCI
                'mci_uncertain_cases': [],     # 高不確定性的 MCI
                'mci_with_cf': [],             # 觸發反事實的 MCI
                'mci_with_kg': []              # 觸發知識圖譜的 MCI
            }
        }
    
    def add_mci_result(self, subject_id: str, result: object):
        """添加 MCI 受試者結果"""
        self.statistics['mci_stats']['total'] += 1
        
        prediction = result.prediction
        confidence = result.confidence
        uq_score = result.uq_score
        agent_decision = result.agent_decision
        
        # 記錄預測分布
        self.statistics['mci_stats']['predictions'][prediction] += 1
        self.statistics['mci_stats']['agent_decisions'][agent_decision] += 1
        
        # 判斷是否有介入
        has_intervention = (
            'SIMULATION' in agent_decision or 
            'ANOMALY' in agent_decision or 
            'INVESTIGATION' in agent_decision
        )
        
        case_info = {
            'subject_id': subject_id,
            'prediction': prediction,
            'confidence': confidence,
            'uq_score': uq_score,
            'agent_decision': agent_decision,
            'has_intervention': has_intervention
        }
        
        self.statistics['mci_stats']['cases'].append(case_info)
        self.mci_results.append(case_info)
        
        # 系統價值分析
        system_value = self.statistics['system_value']
        
        if has_intervention:
            system_value['mci_intervention_cases'].append(case_info)
            
            if 'SIMULATION' in agent_decision:
                system_value['mci_with_cf'].append(case_info)
            elif 'ANOMALY' in agent_decision or 'INVESTIGATION' in agent_decision:
                system_value['mci_with_kg'].append(case_info)
        
        if uq_score > 0.8:
            system_value['mci_uncertain_cases'].append(case_info)
        
        if confidence > 0.8:
            if prediction == 'AD':
                system_value['mci_high_confidence_ad'].append(case_info)
            elif prediction == 'NC':
                system_value['mci_high_confidence_nc'].append(case_info)
    
    def add_nc_ad_result(self, subject_id: str, result: object, ground_truth: str):
        """添加 NC/AD 受試者結果（對比組）"""
        self.statistics['nc_ad_stats']['total'] += 1
        
        prediction = result.prediction
        confidence = result.confidence
        uq_score = result.uq_score
        agent_decision = result.agent_decision
        
        has_intervention = (
            'SIMULATION' in agent_decision or 
            'ANOMALY' in agent_decision or 
            'INVESTIGATION' in agent_decision
        )
        
        self.nc_ad_results.append({
            'subject_id': subject_id,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'correct': (prediction == ground_truth),
            'confidence': confidence,
            'uq_score': uq_score,
            'has_intervention': has_intervention
        })
    
    def calculate_statistics(self):
        """計算統計指標"""
        # MCI 統計
        if self.mci_results:
            confidences = [r['confidence'] for r in self.mci_results]
            uq_scores = [r['uq_score'] for r in self.mci_results]
            interventions = sum(1 for r in self.mci_results if r['has_intervention'])
            high_uq = sum(1 for r in self.mci_results if r['uq_score'] > 0.8)
            
            mci_stats = self.statistics['mci_stats']
            mci_stats['mean_confidence'] = np.mean(confidences)
            mci_stats['mean_uq'] = np.mean(uq_scores)
            mci_stats['intervention_rate'] = interventions / len(self.mci_results)
            mci_stats['high_uq_rate'] = high_uq / len(self.mci_results)
        
        # NC/AD 統計
        if self.nc_ad_results:
            confidences = [r['confidence'] for r in self.nc_ad_results]
            uq_scores = [r['uq_score'] for r in self.nc_ad_results]
            correct = sum(1 for r in self.nc_ad_results if r['correct'])
            interventions = sum(1 for r in self.nc_ad_results if r['has_intervention'])
            high_uq = sum(1 for r in self.nc_ad_results if r['uq_score'] > 0.8)
            
            nc_ad_stats = self.statistics['nc_ad_stats']
            nc_ad_stats['accuracy'] = correct / len(self.nc_ad_results)
            nc_ad_stats['mean_confidence'] = np.mean(confidences)
            nc_ad_stats['mean_uq'] = np.mean(uq_scores)
            nc_ad_stats['intervention_rate'] = interventions / len(self.nc_ad_results)
            nc_ad_stats['high_uq_rate'] = high_uq / len(self.nc_ad_results)
    
    def generate_report(self) -> str:
        """生成 MCI 系統價值報告"""
        self.calculate_statistics()
        
        stats = self.statistics
        mci_stats = stats['mci_stats']
        nc_ad_stats = stats['nc_ad_stats']
        system_value = stats['system_value']
        
        lines = []
        
        lines.append("=" * 100)
        lines.append("MCI SYSTEM VALUE ANALYSIS - 展示系統對不確定案例的處理能力".center(100))
        lines.append("=" * 100)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 1. MCI 概述
        lines.append("=" * 100)
        lines.append("1. MCI (MILD COGNITIVE IMPAIRMENT) OVERVIEW")
        lines.append("=" * 100)
        lines.append(f"Total MCI Subjects: {mci_stats['total']}")
        lines.append("")
        lines.append("Why MCI is the Best Showcase for System Value:")
        lines.append("  1. MCI subjects have NO dedicated LOOCV models (use general binary model)")
        lines.append("  2. MCI is between NC and AD, inherently uncertain")
        lines.append("  3. System's agent intervention should trigger frequently for MCI")
        lines.append("  4. This demonstrates uncertainty quantification and adaptive decision-making")
        lines.append("")
        
        # 2. MCI 預測分布
        lines.append("=" * 100)
        lines.append("2. MCI PREDICTION DISTRIBUTION")
        lines.append("=" * 100)
        for pred, count in sorted(mci_stats['predictions'].items()):
            percentage = count / mci_stats['total'] * 100 if mci_stats['total'] > 0 else 0
            lines.append(f"{pred}: {count} ({percentage:.2f}%)")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  - MCI subjects are predicted as either AD or NC by the binary model")
        lines.append("  - The distribution reflects the model's uncertainty about MCI cases")
        lines.append("")
        
        # 3. 不確定性分析
        lines.append("=" * 100)
        lines.append("3. UNCERTAINTY ANALYSIS - MCI vs NC/AD")
        lines.append("=" * 100)
        lines.append(f"MCI Mean Confidence: {mci_stats['mean_confidence']:.4f}")
        lines.append(f"MCI Mean UQ Score: {mci_stats['mean_uq']:.4f}")
        lines.append(f"MCI High UQ Rate (> 0.8): {mci_stats['high_uq_rate']*100:.2f}%")
        lines.append("")
        lines.append(f"NC/AD Mean Confidence: {nc_ad_stats['mean_confidence']:.4f}")
        lines.append(f"NC/AD Mean UQ Score: {nc_ad_stats['mean_uq']:.4f}")
        lines.append(f"NC/AD High UQ Rate (> 0.8): {nc_ad_stats['high_uq_rate']*100:.2f}%")
        lines.append("")
        
        # 計算差異
        conf_diff = mci_stats['mean_confidence'] - nc_ad_stats['mean_confidence']
        uq_diff = mci_stats['mean_uq'] - nc_ad_stats['mean_uq']
        
        lines.append("Key Findings:")
        lines.append(f"  - MCI confidence is {abs(conf_diff):.4f} {'lower' if conf_diff < 0 else 'higher'} than NC/AD")
        lines.append(f"  - MCI uncertainty is {abs(uq_diff):.4f} {'higher' if uq_diff > 0 else 'lower'} than NC/AD")
        lines.append(f"  - MCI high UQ rate is {abs(mci_stats['high_uq_rate'] - nc_ad_stats['high_uq_rate'])*100:.2f}% {'higher' if uq_diff > 0 else 'lower'}")
        lines.append("")
        
        # 4. Agent 介入分析
        lines.append("=" * 100)
        lines.append("4. AGENT INTERVENTION ANALYSIS")
        lines.append("=" * 100)
        lines.append(f"MCI Intervention Rate: {mci_stats['intervention_rate']*100:.2f}%")
        lines.append(f"NC/AD Intervention Rate: {nc_ad_stats['intervention_rate']*100:.2f}%")
        lines.append("")
        
        intervention_diff = mci_stats['intervention_rate'] - nc_ad_stats['intervention_rate']
        lines.append(f"✓ MCI intervention rate is {abs(intervention_diff)*100:.2f}% {'HIGHER' if intervention_diff > 0 else 'LOWER'} than NC/AD")
        lines.append("")
        
        lines.append("MCI Agent Decision Distribution:")
        for decision, count in sorted(mci_stats['agent_decisions'].items()):
            percentage = count / mci_stats['total'] * 100 if mci_stats['total'] > 0 else 0
            lines.append(f"  {decision}: {count} ({percentage:.2f}%)")
        lines.append("")
        
        # 5. 系統價值展示
        lines.append("=" * 100)
        lines.append("5. SYSTEM VALUE DEMONSTRATION")
        lines.append("=" * 100)
        lines.append(f"MCI with Agent Intervention: {len(system_value['mci_intervention_cases'])}")
        lines.append(f"  - Counterfactual Simulation: {len(system_value['mci_with_cf'])}")
        lines.append(f"  - Knowledge Graph Query: {len(system_value['mci_with_kg'])}")
        lines.append("")
        lines.append(f"MCI with High Uncertainty (UQ > 0.8): {len(system_value['mci_uncertain_cases'])}")
        lines.append("")
        lines.append(f"MCI Predicted as AD (Confidence > 0.8): {len(system_value['mci_high_confidence_ad'])}")
        lines.append(f"MCI Predicted as NC (Confidence > 0.8): {len(system_value['mci_high_confidence_nc'])}")
        lines.append("")
        
        # 6. 代表性案例
        lines.append("=" * 100)
        lines.append("6. REPRESENTATIVE MCI CASES")
        lines.append("=" * 100)
        
        if system_value['mci_with_cf']:
            lines.append("\nMCI with Counterfactual Simulation:")
            for i, case in enumerate(system_value['mci_with_cf'][:5], 1):
                lines.append(f"  {i}. {case['subject_id']}: Pred={case['prediction']}, "
                           f"Conf={case['confidence']:.4f}, UQ={case['uq_score']:.4f}")
        
        if system_value['mci_with_kg']:
            lines.append("\nMCI with Knowledge Graph Query:")
            for i, case in enumerate(system_value['mci_with_kg'][:5], 1):
                lines.append(f"  {i}. {case['subject_id']}: Pred={case['prediction']}, "
                           f"Conf={case['confidence']:.4f}, UQ={case['uq_score']:.4f}")
        
        if system_value['mci_uncertain_cases']:
            lines.append("\nMCI with High Uncertainty:")
            for i, case in enumerate(system_value['mci_uncertain_cases'][:5], 1):
                lines.append(f"  {i}. {case['subject_id']}: Pred={case['prediction']}, "
                           f"Conf={case['confidence']:.4f}, UQ={case['uq_score']:.4f}")
        
        lines.append("")
        
        # 7. 關鍵發現
        lines.append("=" * 100)
        lines.append("7. KEY FINDINGS FOR PAPER")
        lines.append("=" * 100)
        
        findings = []
        
        if intervention_diff > 0:
            findings.append(f"✓ MCI cases trigger {intervention_diff*100:.2f}% MORE agent interventions than NC/AD")
        
        if uq_diff > 0:
            findings.append(f"✓ MCI cases show {uq_diff:.4f} HIGHER uncertainty than NC/AD")
        
        if mci_stats['high_uq_rate'] > nc_ad_stats['high_uq_rate']:
            findings.append(f"✓ {mci_stats['high_uq_rate']*100:.2f}% of MCI cases have high uncertainty (vs {nc_ad_stats['high_uq_rate']*100:.2f}% for NC/AD)")
        
        findings.append(f"✓ System demonstrates adaptive decision-making: {mci_stats['intervention_rate']*100:.2f}% intervention rate for MCI")
        
        if findings:
            for i, finding in enumerate(findings, 1):
                lines.append(f"{i}. {finding}")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("END OF REPORT")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def save_report(self, filename: str = "mci_system_value_report.txt"):
        """保存報告"""
        report = self.generate_report()
        report_file = self.output_dir / filename
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        return report_file
    
    def save_json(self, filename: str = "mci_system_value.json"):
        """保存 JSON 數據"""
        json_file = self.output_dir / filename
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'mci_results': self.mci_results,
                'nc_ad_results': self.nc_ad_results,
                'statistics': {
                    'mci_stats': dict(self.statistics['mci_stats']),
                    'nc_ad_stats': self.statistics['nc_ad_stats'],
                    'system_value': {
                        k: v for k, v in self.statistics['system_value'].items()
                    }
                }
            }, f, indent=2, ensure_ascii=False)
        return json_file


def scan_subjects_by_group() -> Dict[str, List[str]]:
    """按組別掃描受試者"""
    subjects_by_group = {'NC': [], 'MCI': [], 'AD': []}
    
    for group in ['NC', 'MCI', 'AD']:
        data_folders = glob.glob(f"data/MRI_processed/{group}/sub-*")
        for folder_path in data_folders:
            subject_id = Path(folder_path).name
            # 檢查是否有足夠的 nii.gz 文件
            nii_files = list(Path(folder_path).glob("*.nii.gz"))
            if len(nii_files) >= 3:
                subjects_by_group[group].append(subject_id)
    
    return subjects_by_group


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="MCI System Value Analysis - 展示系統對 MCI 的處理能力"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/mci_system_value',
        help='輸出目錄'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='限制每組的受試者數量（用於測試）'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='禁用 LLM 模式'
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("MCI SYSTEM VALUE ANALYSIS")
    print("=" * 100)
    print()
    
    # 掃描受試者
    print("Scanning for subjects...")
    subjects_by_group = scan_subjects_by_group()
    
    print(f"Found subjects:")
    print(f"  NC: {len(subjects_by_group['NC'])}")
    print(f"  MCI: {len(subjects_by_group['MCI'])}")
    print(f"  AD: {len(subjects_by_group['AD'])}")
    print()
    
    if args.limit:
        for group in subjects_by_group:
            subjects_by_group[group] = subjects_by_group[group][:args.limit]
        print(f"Limited to {args.limit} subjects per group")
        print()
    
    # 初始化分析器
    analyzer = MCISystemValueAnalyzer(args.output)
    
    # 初始化 CDDA Agent
    print("Initializing CDDA Agent...")
    try:
        agent = CDDAAgent(
            use_llm=not args.no_llm,
            verbose=False
        )
        print("✓ CDDA Agent initialized")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        sys.exit(1)
    
    # 分析 MCI 受試者
    print("=" * 100)
    print("ANALYZING MCI SUBJECTS")
    print("=" * 100)
    print()
    
    mci_subjects = subjects_by_group['MCI']
    for i, subject_id in enumerate(mci_subjects, 1):
        print(f"[{i}/{len(mci_subjects)}] {subject_id}...", end=' ', flush=True)
        
        try:
            result = agent.run_analysis(subject_id)
            analyzer.add_mci_result(subject_id, result)
            print(f"✓ {result.prediction} (Conf: {result.confidence:.2f}, UQ: {result.uq_score:.2f})")
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
    
    print()
    
    # 分析 NC/AD 受試者（對比組）
    print("=" * 100)
    print("ANALYZING NC/AD SUBJECTS (Control Group)")
    print("=" * 100)
    print()
    
    nc_ad_subjects = subjects_by_group['NC'] + subjects_by_group['AD']
    ground_truths = {s: 'NC' for s in subjects_by_group['NC']}
    ground_truths.update({s: 'AD' for s in subjects_by_group['AD']})
    
    for i, subject_id in enumerate(nc_ad_subjects, 1):
        print(f"[{i}/{len(nc_ad_subjects)}] {subject_id}...", end=' ', flush=True)
        
        try:
            result = agent.run_analysis(subject_id)
            analyzer.add_nc_ad_result(subject_id, result, ground_truths[subject_id])
            print(f"✓ {result.prediction} (GT: {ground_truths[subject_id]})")
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
    
    print()
    
    # 生成報告
    print("=" * 100)
    print("GENERATING REPORT")
    print("=" * 100)
    print()
    
    report_file = analyzer.save_report()
    print(f"✓ Report saved: {report_file}")
    
    json_file = analyzer.save_json()
    print(f"✓ JSON saved: {json_file}")
    
    print()
    
    # 打印報告
    report = analyzer.generate_report()
    print(report)


if __name__ == "__main__":
    main()
