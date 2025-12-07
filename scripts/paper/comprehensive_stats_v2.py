#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Statistics V2 - Three-Class Analysis with LOOCV

Complete statistical analysis including:
1. LOOCV Integrity Check (NC/AD with dedicated models)
2. Binary Classification Metrics (NC vs AD)
3. Three-Class Analysis (NC/MCI/AD with system value)
4. Uncertainty Quantification (MCI vs NC/AD comparison)
5. Agent Decision Analysis
6. System Value Quantification

Usage:
    python scripts/paper/comprehensive_stats_v2.py
    python scripts/paper/comprehensive_stats_v2.py --limit 10
"""

import sys
import json
import glob
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.agents.cdda_agent import CDDAAgent


class ComprehensiveStatistics:
    """Complete three-class statistical analyzer"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "nc": [],  # NC subjects
            "ad": [],  # AD subjects
            "mci": [],  # MCI subjects
        }

        self.statistics = {
            # LOOCV Integrity
            "loocv": {"total_nc_ad": 0, "verified": 0, "fallback": 0, "coverage": 0.0},
            # Binary Classification (NC vs AD)
            "binary": {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "fp_flagged_success": 0,  # Renamed from fp_corrected
                "confusion": {
                    "AD_as_AD": 0,
                    "AD_as_NC": 0,
                    "NC_as_AD": 0,
                    "NC_as_NC": 0,
                },
                "precision": 0.0,
                "recall": 0.0,
                "specificity": 0.0,
                "f1": 0.0,
            },
            # Three-Class Analysis
            "three_class": {
                "nc": {"total": 0, "predictions": defaultdict(int)},
                "ad": {"total": 0, "predictions": defaultdict(int)},
                "mci": {"total": 0, "predictions": defaultdict(int)},
            },
            # Uncertainty Analysis
            "uncertainty": {
                "nc_ad": {"mean_conf": 0.0, "mean_uq": 0.0, "high_uq_rate": 0.0},
                "mci": {"mean_conf": 0.0, "mean_uq": 0.0, "high_uq_rate": 0.0},
            },
            # Agent Decisions
            "agent_decisions": {"nc_ad": defaultdict(int), "mci": defaultdict(int)},
            # System Value (Risk Flagging & Error Interception)
            "system_value": {
                "nc_ad_interventions": 0,
                "mci_interventions": 0,
                "fp_flagged": 0,  # Total FP cases flagged
                "fn_flagged": 0,  # Total FN cases flagged
                "ambiguous_flagged": 0,  # Total ambiguous cases flagged
                "concurred": 0,  # Cases where agent agreed with model
            },
        }

    def _normalize_label(self, label) -> str:
        """
        Normalize labels to standard format (NC, AD, MCI)
        
        Binary model encoding: NC=0, AD=1
        Handles both numeric predictions and text labels
        
        Args:
            label: Raw label (int, str, or None)
        
        Returns:
            Standardized label: 'NC', 'AD', 'MCI', or 'UNKNOWN'
        """
        if label is None:
            return "UNKNOWN"

        s = str(label).upper().strip()

        # Handle numeric encoding from model predictions
        if s == "0":
            return "NC"  # Binary model: 0=NC
        if s == "1":
            return "AD"  # Binary model: 1=AD

        # Handle text labels from ground truth
        if "NORMAL" in s or s == "NC":
            return "NC"
        if "ALZHEIMER" in s or s == "AD":
            return "AD"
        if "MILD" in s or "MCI" in s:
            return "MCI"

        return s

    def _parse_agent_outcome(self, report: str) -> str:
        """
        Parse Agent B's clinical audit report to extract risk flag
        
        Maps Agent B's verdict to standardized risk flags:
        - FLAGGED_FP_NC: Agent corrects False Positive (Model AD -> Agent NC)
        - FLAGGED_FN_AD: Agent detects potential Missed Diagnosis (False Negative)
        - FLAGGED_AMBIGUOUS: Agent detects ambiguity/MCI
        - CONCURRED: Agent agrees with model (no intervention needed)
        
        Args:
            report: Agent B's clinical report text
        
        Returns:
            Risk flag string
        """
        if not report:
            return "UNKNOWN"

        r_low = report.lower()

        # 1. Parse Audit Verdict (Primary indicator)
        if "audit verdict:" in r_low:
            import re

            verdict_match = re.search(r"audit verdict:\s*\[?([^\]\n]+)\]?", r_low)
            if verdict_match:
                verdict = verdict_match.group(1).strip().lower()

                # Map verdict to risk flag
                if "concur" in verdict:
                    return "CONCURRED"
                elif "flagged for review" in verdict:
                    # Need to determine if FP or FN
                    if "false positive" in r_low or "probable normal" in r_low:
                        return "FLAGGED_FP_NC"
                    elif "false negative" in r_low or "missed diagnosis" in r_low:
                        return "FLAGGED_FN_AD"
                    else:
                        return "FLAGGED_AMBIGUOUS"
                elif "atypical ad" in verdict:
                    return "CONCURRED"  # Agent agrees it's AD (atypical variant)
                elif "probable normal" in verdict:
                    return "FLAGGED_FP_NC"

        # 2. Fallback: Parse by key phrases
        if "false positive" in r_low or "probable normal" in r_low:
            return "FLAGGED_FP_NC"
        if "false negative" in r_low or "missed diagnosis" in r_low:
            return "FLAGGED_FN_AD"
        if "suspected mci" in r_low or "ambiguous" in r_low:
            return "FLAGGED_AMBIGUOUS"

        # 3. Check Risk Assessment
        if "high risk of discrepancy" in r_low:
            return "FLAGGED_AMBIGUOUS"
        if "low risk" in r_low:
            return "CONCURRED"

        # Default: Assume concurrence if no flags detected
        return "CONCURRED"

    def add_result(self, subject_id: str, ground_truth: str, result: object):
        """
        Add analysis result to statistics with Clinical Safety Auditor philosophy
        
        This method:
        1. Normalizes labels (fixes 0/1 inversion bug)
        2. Parses Agent B's audit verdict
        3. Verifies LOOCV model usage
        4. Calculates risk flagging success metrics
        
        Args:
            subject_id: Subject identifier
            ground_truth: True diagnosis label
            result: Analysis result object from CDDA Agent
        """

        # 1. [CRITICAL] Normalize Labels (Fix 0/1 Inversion)
        gt = self._normalize_label(ground_truth)
        raw_pred = self._normalize_label(result.prediction)

        # 2. [NEW] Parse Agent B's Clinical Audit Verdict
        # Returns: FLAGGED_FP_NC, FLAGGED_FN_AD, FLAGGED_AMBIGUOUS, or CONCURRED
        agent_risk_flag = self._parse_agent_outcome(result.clinical_report)

        # 3. Verify Model Usage (LOOCV Integrity)
        model_used = "unknown"
        for step in result.reasoning_chain:
            if "using" in step and ".joblib" in step:
                import re

                match = re.search(r"(?:using\s+)?([\w\-]+\.joblib)", step)
                if match:
                    model_used = match.group(1)
                    break

        loocv_verified = (subject_id in model_used) if model_used != "unknown" else False

        # 4. Check for Agent Intervention
        has_intervention = any(
            x in result.agent_decision for x in ["SIMULATION", "ANOMALY", "INVESTIGATION"]
        )

        # 5. [CORE METRIC] Calculate False Positive Flagging Success
        # SUCCESS Condition: GT is NC AND Model predicted AD (FP) AND Agent flagged it as NC
        is_fp_flagged_success = (
            gt == "NC" and raw_pred == "AD" and agent_risk_flag == "FLAGGED_FP_NC"
        )

        # 6. Build Complete Data Record
        result_data = {
            "subject_id": subject_id,
            "ground_truth": gt,
            "prediction": raw_pred,
            "agent_risk_flag": agent_risk_flag,  # Renamed from agent_final
            "confidence": result.confidence,
            "uq_score": result.uq_score,
            "agent_decision": result.agent_decision,
            "model_used": model_used,
            "loocv_verified": loocv_verified,
            "has_intervention": has_intervention,
            "is_fp_flagged_success": is_fp_flagged_success,  # Renamed from is_fp_corrected
        }

        # 7. Store Result by Group
        if gt == "NC":
            self.results["nc"].append(result_data)
        elif gt == "AD":
            self.results["ad"].append(result_data)
        elif gt == "MCI":
            self.results["mci"].append(result_data)

        # 8. Update Running Statistics
        self._update_statistics(result_data)

    def _update_statistics(self, data):
        """
        Update running statistics with new result
        
        Tracks:
        - LOOCV integrity
        - Binary classification metrics
        - Risk flagging success (FP/FN detection)
        - Agent intervention patterns
        """

        gt = data["ground_truth"]
        pred = data["prediction"]
        agent_flag = data["agent_risk_flag"]
        intervention = data["has_intervention"]
        loocv_verified = data["loocv_verified"]

        # LOOCV stats
        if gt in ["NC", "AD"]:
            self.statistics["loocv"]["total_nc_ad"] += 1
            if loocv_verified:
                self.statistics["loocv"]["verified"] += 1
            else:
                self.statistics["loocv"]["fallback"] += 1

        # Binary classification (NC vs AD only)
        if gt in ["NC", "AD"]:
            self.statistics["binary"]["total"] += 1
            if pred == gt:
                self.statistics["binary"]["correct"] += 1

            # [CORE METRIC] Track FP flagging success
            if data.get("is_fp_flagged_success", False):
                self.statistics["binary"]["fp_flagged_success"] += 1

            # Confusion matrix
            key = f"{gt}_as_{pred}"
            if key in self.statistics["binary"]["confusion"]:
                self.statistics["binary"]["confusion"][key] += 1

        # Three-class predictions
        self.statistics["three_class"][gt.lower()]["total"] += 1
        self.statistics["three_class"][gt.lower()]["predictions"][pred] += 1

        # Agent decisions
        if gt in ["NC", "AD"]:
            self.statistics["agent_decisions"]["nc_ad"][data["agent_decision"]] += 1
            if intervention:
                self.statistics["system_value"]["nc_ad_interventions"] += 1
        elif gt == "MCI":
            self.statistics["agent_decisions"]["mci"][data["agent_decision"]] += 1
            if intervention:
                self.statistics["system_value"]["mci_interventions"] += 1

        # [NEW] Track risk flagging patterns
        if agent_flag == "FLAGGED_FP_NC":
            self.statistics["system_value"]["fp_flagged"] += 1
        elif agent_flag == "FLAGGED_FN_AD":
            self.statistics["system_value"]["fn_flagged"] += 1
        elif agent_flag == "FLAGGED_AMBIGUOUS":
            self.statistics["system_value"]["ambiguous_flagged"] += 1
        elif agent_flag == "CONCURRED":
            self.statistics["system_value"]["concurred"] += 1

    def calculate_statistics(self):
        """Calculate final statistics"""

        # LOOCV coverage
        loocv = self.statistics["loocv"]
        if loocv["total_nc_ad"] > 0:
            loocv["coverage"] = loocv["verified"] / loocv["total_nc_ad"] * 100

        # Binary metrics
        binary = self.statistics["binary"]
        if binary["total"] > 0:
            binary["accuracy"] = binary["correct"] / binary["total"]

        conf = binary["confusion"]
        tp = conf["AD_as_AD"]
        tn = conf["NC_as_NC"]
        fp = conf["NC_as_AD"]
        fn = conf["AD_as_NC"]

        if (tp + fp) > 0:
            binary["precision"] = tp / (tp + fp)
        if (tp + fn) > 0:
            binary["recall"] = tp / (tp + fn)
        if (tn + fp) > 0:
            binary["specificity"] = tn / (tn + fp)
        if binary["precision"] > 0 and binary["recall"] > 0:
            binary["f1"] = (
                2
                * (binary["precision"] * binary["recall"])
                / (binary["precision"] + binary["recall"])
            )

        # Uncertainty analysis
        nc_ad_results = self.results["nc"] + self.results["ad"]
        mci_results = self.results["mci"]

        if nc_ad_results:
            confs = [r["confidence"] for r in nc_ad_results]
            uqs = [r["uq_score"] for r in nc_ad_results]
            self.statistics["uncertainty"]["nc_ad"]["mean_conf"] = np.mean(confs)
            self.statistics["uncertainty"]["nc_ad"]["mean_uq"] = np.mean(uqs)
            self.statistics["uncertainty"]["nc_ad"]["high_uq_rate"] = sum(
                1 for u in uqs if u > 0.8
            ) / len(uqs)

        if mci_results:
            confs = [r["confidence"] for r in mci_results]
            uqs = [r["uq_score"] for r in mci_results]
            self.statistics["uncertainty"]["mci"]["mean_conf"] = np.mean(confs)
            self.statistics["uncertainty"]["mci"]["mean_uq"] = np.mean(uqs)
            self.statistics["uncertainty"]["mci"]["high_uq_rate"] = sum(
                1 for u in uqs if u > 0.8
            ) / len(uqs)

    def generate_report(self) -> str:
        """Generate comprehensive report"""
        self.calculate_statistics()

        stats = self.statistics
        lines = []

        lines.append("=" * 80)
        lines.append("COMPREHENSIVE STATISTICS REPORT (NC/MCI/AD)".center(80))
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 1. LOOCV Integrity
        lines.append("=" * 80)
        lines.append("1. LOOCV INTEGRITY CHECK")
        lines.append("=" * 80)
        loocv = stats["loocv"]
        lines.append(f"NC/AD Subjects: {loocv['total_nc_ad']}")
        lines.append(f"LOOCV Verified: {loocv['verified']}")
        lines.append(f"Fallback Models: {loocv['fallback']}")
        lines.append(f"Coverage: {loocv['coverage']:.2f}%")
        if loocv["coverage"] == 100:
            lines.append("STATUS: PASSED - 100% Strict Separation")
        lines.append("")

        # 2. Binary Classification
        lines.append("=" * 80)
        lines.append("2. BINARY CLASSIFICATION (NC vs AD)")
        lines.append("=" * 80)
        binary = stats["binary"]
        lines.append(f"Total: {binary['total']}")
        lines.append(f"Correct: {binary['correct']}")
        lines.append(
            f"Accuracy: {binary['accuracy']:.4f} ({binary['accuracy']*100:.2f}%)"
        )
        lines.append("")
        lines.append("Confusion Matrix:")
        conf = binary["confusion"]
        lines.append(f"  AD -> AD: {conf['AD_as_AD']}")
        lines.append(f"  AD -> NC: {conf['AD_as_NC']}")
        lines.append(f"  NC -> AD: {conf['NC_as_AD']}")
        lines.append(f"  NC -> NC: {conf['NC_as_NC']}")
        lines.append("")
        lines.append("Metrics:")
        lines.append(f"  Precision (AD): {binary['precision']:.4f}")
        lines.append(f"  Recall (AD): {binary['recall']:.4f}")
        lines.append(f"  Specificity (NC): {binary['specificity']:.4f}")
        lines.append(f"  F1-Score: {binary['f1']:.4f}")
        lines.append("")
        lines.append(
            f"  False Positives Flagged (Success): {self.statistics['binary']['fp_flagged_success']}"
        )
        lines.append("")


        # 3. Three-Class Analysis
        lines.append("=" * 80)
        lines.append("3. THREE-CLASS ANALYSIS (NC/MCI/AD)")
        lines.append("=" * 80)
        three = stats["three_class"]
        for group in ["nc", "ad", "mci"]:
            data = three[group]
            lines.append(f"\n{group.upper()} Subjects (n={data['total']}):")
            for pred, count in sorted(data["predictions"].items()):
                pct = count / data["total"] * 100 if data["total"] > 0 else 0
                lines.append(f"  Predicted as {pred}: {count} ({pct:.1f}%)")
        lines.append("")

        # 4. Uncertainty Analysis
        lines.append("=" * 80)
        lines.append("4. UNCERTAINTY ANALYSIS (MCI vs NC/AD)")
        lines.append("=" * 80)
        unc = stats["uncertainty"]
        lines.append("NC/AD:")
        lines.append(f"  Mean Confidence: {unc['nc_ad']['mean_conf']:.4f}")
        lines.append(f"  Mean UQ Score: {unc['nc_ad']['mean_uq']:.4f}")
        lines.append(f"  High UQ Rate: {unc['nc_ad']['high_uq_rate']*100:.2f}%")
        lines.append("\nMCI:")
        lines.append(f"  Mean Confidence: {unc['mci']['mean_conf']:.4f}")
        lines.append(f"  Mean UQ Score: {unc['mci']['mean_uq']:.4f}")
        lines.append(f"  High UQ Rate: {unc['mci']['high_uq_rate']*100:.2f}%")
        lines.append("\nComparison:")
        conf_diff = unc["mci"]["mean_conf"] - unc["nc_ad"]["mean_conf"]
        uq_diff = unc["mci"]["mean_uq"] - unc["nc_ad"]["mean_uq"]
        lines.append(f"  MCI Confidence Difference: {conf_diff:+.4f}")
        lines.append(f"  MCI UQ Difference: {uq_diff:+.4f}")
        lines.append("")

        # 5. Agent Decisions
        lines.append("=" * 80)
        lines.append("5. AGENT DECISION ANALYSIS")
        lines.append("=" * 80)
        lines.append("NC/AD Decisions:")
        for decision, count in sorted(stats["agent_decisions"]["nc_ad"].items()):
            lines.append(f"  {decision}: {count}")
        lines.append("\nMCI Decisions:")
        for decision, count in sorted(stats["agent_decisions"]["mci"].items()):
            lines.append(f"  {decision}: {count}")
        lines.append("")

        # 6. System Value & Agent Audit Summary
        lines.append("=" * 80)
        lines.append("6. SYSTEM VALUE & AGENT AUDIT SUMMARY")
        lines.append("=" * 80)
        sv = stats["system_value"]
        lines.append(f"NC/AD Interventions: {sv['nc_ad_interventions']}")
        lines.append(f"MCI Interventions: {sv['mci_interventions']}")

        nc_ad_total = len(self.results["nc"]) + len(self.results["ad"])
        mci_total = len(self.results["mci"])

        if nc_ad_total > 0:
            nc_ad_rate = sv["nc_ad_interventions"] / nc_ad_total * 100
            lines.append(f"NC/AD Intervention Rate: {nc_ad_rate:.2f}%")

        if mci_total > 0:
            mci_rate = sv["mci_interventions"] / mci_total * 100
            lines.append(f"MCI Intervention Rate: {mci_rate:.2f}%")
        
        # [NEW] Agent Risk Flagging Distribution
        lines.append("\nAgent Risk Flags (NC/AD):")
        flag_counts = defaultdict(int)
        for result in self.results["nc"] + self.results["ad"]:
            flag_counts[result["agent_risk_flag"]] += 1
        for flag, count in sorted(flag_counts.items()):
            pct = count / nc_ad_total * 100 if nc_ad_total > 0 else 0
            lines.append(f"  {flag}: {count} ({pct:.1f}%)")

        # [NEW] Clinical Safety Metrics (Risk Flagging & Error Interception)
        lines.append("\nClinical Safety Metrics:")
        lines.append(f"  False Positives Flagged: {sv['fp_flagged']}")
        lines.append(f"  False Negatives Flagged: {sv['fn_flagged']}")
        lines.append(f"  Ambiguous Cases Flagged: {sv['ambiguous_flagged']}")
        lines.append(f"  Cases Concurred: {sv['concurred']}")
        lines.append(
            f"  FP Flagging Success Rate: {binary['fp_flagged_success']}/{conf['NC_as_AD']} "
            f"({binary['fp_flagged_success']/conf['NC_as_AD']*100:.1f}%)"
            if conf["NC_as_AD"] > 0
            else "  FP Flagging Success Rate: N/A (no FP cases)"
        )

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_report(self, filename: str = "comprehensive_stats_report.txt"):
        """Save report"""
        report = self.generate_report()
        report_file = self.output_dir / filename
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        return report_file

    def save_json(self, filename: str = "comprehensive_stats.json"):
        """Save JSON data"""
        json_file = self.output_dir / filename
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "statistics": self._convert_defaultdict(self.statistics),
                    "results": self.results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return json_file
    
    def save_csv(self, filename: str = "final_results.csv"):
        """
        Save detailed results to CSV for paper analysis
        
        Columns:
        - subject_id, ground_truth, prediction, agent_final, confidence, uq_score
        - agent_decision, model_used, loocv_verified, has_intervention, is_fp_corrected
        """
        import csv
        
        csv_file = self.output_dir / filename
        
        # Collect all results
        all_results = self.results["nc"] + self.results["ad"] + self.results["mci"]
        
        if not all_results:
            print("[WARN] No results to save to CSV")
            return None
        
        # Define columns (updated for Clinical Safety Auditor)
        fieldnames = [
            "subject_id",
            "ground_truth",
            "prediction",
            "agent_risk_flag",
            "confidence",
            "uq_score",
            "agent_decision",
            "model_used",
            "loocv_verified",
            "has_intervention",
            "is_fp_flagged_success",
        ]

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in all_results:
                writer.writerow(
                    {
                        "subject_id": result["subject_id"],
                        "ground_truth": result["ground_truth"],
                        "prediction": result["prediction"],
                        "agent_risk_flag": result["agent_risk_flag"],
                        "confidence": f"{result['confidence']:.4f}",
                        "uq_score": f"{result['uq_score']:.4f}",
                        "agent_decision": result["agent_decision"],
                        "model_used": result["model_used"],
                        "loocv_verified": result["loocv_verified"],
                        "has_intervention": result["has_intervention"],
                        "is_fp_flagged_success": result["is_fp_flagged_success"],
                    }
                )
        
        return csv_file

    def _convert_defaultdict(self, obj):
        """Convert defaultdict to dict"""
        if isinstance(obj, defaultdict):
            return {k: self._convert_defaultdict(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {k: self._convert_defaultdict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_defaultdict(item) for item in obj]
        else:
            return obj


def scan_all_subjects():
    """Scan all subjects"""
    subjects = {}

    for group in ["NC", "MCI", "AD"]:
        data_folders = glob.glob(f"data/MRI_processed/{group}/sub-*")
        for folder_path in data_folders:
            subject_id = Path(folder_path).name
            nii_files = list(Path(folder_path).glob("*.nii.gz"))
            if len(nii_files) >= 3:
                subjects[subject_id] = group

    return subjects


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Statistics V2")
    parser.add_argument(
        "--output", default="output/comprehensive_stats_v2", help="Output directory"
    )
    parser.add_argument("--limit", type=int, help="Limit subjects per group")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM")
    parser.add_argument("--nc-ad-only", action="store_true", help="Only analyze NC/AD")

    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE STATISTICS V2")
    print("=" * 80)
    print()

    # Scan subjects
    print("Scanning subjects...")
    all_subjects = scan_all_subjects()

    nc_subjects = [s for s, g in all_subjects.items() if g == "NC"]
    ad_subjects = [s for s, g in all_subjects.items() if g == "AD"]
    mci_subjects = [s for s, g in all_subjects.items() if g == "MCI"]

    print(
        f"Found: NC={len(nc_subjects)}, AD={len(ad_subjects)}, MCI={len(mci_subjects)}"
    )

    if args.limit:
        nc_subjects = nc_subjects[: args.limit]
        ad_subjects = ad_subjects[: args.limit]
        mci_subjects = mci_subjects[: args.limit]
        print(f"Limited to {args.limit} per group")

    if args.nc_ad_only:
        mci_subjects = []
        print("MCI excluded (NC/AD only mode)")

    print()

    # Initialize
    analyzer = ComprehensiveStatistics(args.output)

    print("Initializing CDDA Agent...")
    try:
        agent = CDDAAgent(use_llm=not args.no_llm, verbose=False)
        print("Agent ready")
        print()
    except Exception as e:
        print(f"Failed: {e}")
        return

    # Analyze
    print("=" * 80)
    print("ANALYZING SUBJECTS")
    print("=" * 80)
    print()

    all_test_subjects = (
        [(s, "NC") for s in nc_subjects]
        + [(s, "AD") for s in ad_subjects]
        + [(s, "MCI") for s in mci_subjects]
    )

    for i, (subject_id, ground_truth) in enumerate(all_test_subjects, 1):
        print(
            f"[{i}/{len(all_test_subjects)}] {subject_id} ({ground_truth})...",
            end=" ",
            flush=True,
        )

        try:
            result = agent.run_analysis(subject_id)
            analyzer.add_result(subject_id, ground_truth, result)
            print(f"OK - {result.prediction} (Conf: {result.confidence:.2f})")
        except Exception as e:
            print(f"ERROR - {str(e)[:40]}")

    print()
    print("=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)
    print()

    report_file = analyzer.save_report()
    print(f"Report saved: {report_file}")

    json_file = analyzer.save_json()
    print(f"JSON saved: {json_file}")
    
    csv_file = analyzer.save_csv()
    if csv_file:
        print(f"CSV saved: {csv_file}")

    print()

    # Print report
    report = analyzer.generate_report()
    print(report)


if __name__ == "__main__":
    main()
