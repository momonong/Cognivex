"""
CNN-RF Model Evaluation Script

Comprehensive evaluation of CNN-RF model on all available data.
Provides detailed metrics including:
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- ROC curves and AUC scores
- Feature importance analysis
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.cnn_rf.config import MODELS

# Define paths
ROI_FEATURES_PATH = Path("data/roi_features.csv")
OUTPUT_DIR = Path("output/cnn_rf")


class CNNRFEvaluator:
    """Comprehensive evaluator for CNN-RF models"""
    
    def __init__(self, model_name="NC_vs_AD"):
        """
        Initialize evaluator
        
        Args:
            model_name: Model to evaluate ('NC_vs_AD' or 'NC_MCI_AD')
        """
        self.model_name = model_name
        self.model_config = MODELS[model_name]
        self.model_path = self.model_config['path']
        self.classes = self.model_config['classes']
        
        # Load model
        print(f"\n[INFO] Loading model: {self.model_name}")
        self.model = joblib.load(self.model_path)
        print(f"[OK] Model loaded from: {self.model_path}")
        
        # Load data
        print(f"\n[INFO] Loading data from: {ROI_FEATURES_PATH}")
        self.data = pd.read_csv(ROI_FEATURES_PATH)
        
        # Filter for relevant classes
        self.data = self.data[self.data['Group'].isin(self.classes)]
        print(f"[OK] Loaded {len(self.data)} samples")
        
        # Prepare features and labels
        self.X = self.data.drop(columns=['Subject_ID', 'Group'])
        self.y_true_text = self.data['Group']
        
        # Convert to numeric labels (alphabetically sorted)
        from pandas.api.types import CategoricalDtype
        cat_type = CategoricalDtype(categories=sorted(self.classes), ordered=True)
        self.y_true = self.y_true_text.astype(cat_type).cat.codes
        
        # Get predictions
        print(f"\n[INFO] Generating predictions...")
        self.y_pred = self.model.predict(self.X)
        self.y_proba = self.model.predict_proba(self.X)
        print(f"[OK] Predictions generated")
        
        # Create output directory
        self.output_dir = OUTPUT_DIR / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_basic_metrics(self):
        """Calculate basic classification metrics"""
        print("\n" + "="*80)
        print("Basic Classification Metrics")
        print("="*80)
        
        # Overall accuracy
        accuracy = accuracy_score(self.y_true, self.y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class metrics
        precision = precision_score(self.y_true, self.y_pred, average=None)
        recall = recall_score(self.y_true, self.y_pred, average=None)
        f1 = f1_score(self.y_true, self.y_pred, average=None)
        
        print(f"\nPer-Class Metrics:")
        print("-"*80)
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*80)
        
        for i, cls in enumerate(self.classes):
            print(f"{cls:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
        
        # Macro and weighted averages
        print("-"*80)
        macro_precision = precision_score(self.y_true, self.y_pred, average='macro')
        macro_recall = recall_score(self.y_true, self.y_pred, average='macro')
        macro_f1 = f1_score(self.y_true, self.y_pred, average='macro')
        
        print(f"{'Macro Avg':<10} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
        
        weighted_precision = precision_score(self.y_true, self.y_pred, average='weighted')
        weighted_recall = recall_score(self.y_true, self.y_pred, average='weighted')
        weighted_f1 = f1_score(self.y_true, self.y_pred, average='weighted')
        
        print(f"{'Weighted Avg':<10} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1
        }
    
    def print_classification_report(self):
        """Print detailed classification report"""
        print("\n" + "="*80)
        print("Detailed Classification Report")
        print("="*80)
        
        report = classification_report(
            self.y_true,
            self.y_pred,
            target_names=self.classes,
            digits=4
        )
        print(report)
    
    def plot_confusion_matrix(self, save=True):
        """Plot confusion matrix"""
        print("\n" + "="*80)
        print("Confusion Matrix")
        print("="*80)
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Print text version
        print("\nConfusion Matrix (counts):")
        print("-"*80)
        header = "True\\Pred  " + "  ".join([f"{cls:>8}" for cls in self.classes])
        print(header)
        print("-"*80)
        for i, true_cls in enumerate(self.classes):
            row = f"{true_cls:<10} " + "  ".join([f"{cm[i,j]:>8}" for j in range(len(self.classes))])
            print(row)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        print("\nConfusion Matrix (percentages):")
        print("-"*80)
        print(header)
        print("-"*80)
        for i, true_cls in enumerate(self.classes):
            row = f"{true_cls:<10} " + "  ".join([f"{cm_percent[i,j]:>7.1f}%" for j in range(len(self.classes))])
            print(row)
        
        # Plot
        if save:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot counts
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.classes, yticklabels=self.classes,
                       ax=ax1, cbar_kws={'label': 'Count'})
            ax1.set_title(f'Confusion Matrix - Counts\n{self.model_name}')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # Plot percentages
            sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                       xticklabels=self.classes, yticklabels=self.classes,
                       ax=ax2, cbar_kws={'label': 'Percentage (%)'})
            ax2.set_title(f'Confusion Matrix - Percentages\n{self.model_name}')
            ax2.set_ylabel('True Label')
            ax2.set_xlabel('Predicted Label')
            
            plt.tight_layout()
            
            save_path = self.output_dir / f"{self.model_name}_confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Confusion matrix saved: {save_path}")
            plt.close()
        
        return cm
    
    def calculate_roc_auc(self, save=True):
        """Calculate and plot ROC curves and AUC scores"""
        print("\n" + "="*80)
        print("ROC Curves and AUC Scores")
        print("="*80)
        
        n_classes = len(self.classes)
        
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(self.y_true, classes=range(n_classes))
        
        # Handle binary classification case
        if n_classes == 2 and y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], self.y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            print(f"{self.classes[i]:<10} AUC: {roc_auc[i]:.4f}")
        
        # Calculate macro-average ROC curve and AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        print(f"{'Macro Avg':<10} AUC: {roc_auc['macro']:.4f}")
        
        # Plot ROC curves
        if save:
            plt.figure(figsize=(10, 8))
            
            # Plot ROC curve for each class
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{self.classes[i]} (AUC = {roc_auc[i]:.4f})')
            
            # Plot macro-average ROC curve
            plt.plot(fpr["macro"], tpr["macro"], color='navy', lw=2, linestyle='--',
                    label=f'Macro Average (AUC = {roc_auc["macro"]:.4f})')
            
            # Plot diagonal
            plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curves - {self.model_name}', fontsize=14)
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(alpha=0.3)
            
            save_path = self.output_dir / f"{self.model_name}_roc_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] ROC curves saved: {save_path}")
            plt.close()
        
        return roc_auc
    
    def analyze_misclassifications(self):
        """Analyze misclassified samples"""
        print("\n" + "="*80)
        print("Misclassification Analysis")
        print("="*80)
        
        # Find misclassified samples
        misclassified_mask = self.y_true != self.y_pred
        n_misclassified = misclassified_mask.sum()
        n_total = len(self.y_true)
        
        print(f"\nTotal samples: {n_total}")
        print(f"Correctly classified: {n_total - n_misclassified} ({(n_total - n_misclassified)/n_total*100:.2f}%)")
        print(f"Misclassified: {n_misclassified} ({n_misclassified/n_total*100:.2f}%)")
        
        if n_misclassified > 0:
            # Get misclassified samples
            misclassified_data = self.data[misclassified_mask].copy()
            misclassified_data['Predicted'] = [self.classes[p] for p in self.y_pred[misclassified_mask]]
            misclassified_data['Confidence'] = self.y_proba[misclassified_mask].max(axis=1)
            
            print(f"\nMisclassification Patterns:")
            print("-"*80)
            
            # Count misclassification patterns
            for true_cls in self.classes:
                for pred_cls in self.classes:
                    if true_cls != pred_cls:
                        count = ((self.y_true_text == true_cls) & 
                                (misclassified_data['Predicted'] == pred_cls)).sum()
                        if count > 0:
                            print(f"  {true_cls} -> {pred_cls}: {count} samples")
            
            # Show some examples
            print(f"\nExample Misclassifications (showing first 10):")
            print("-"*80)
            
            examples = misclassified_data[['Subject_ID', 'Group', 'Predicted', 'Confidence']].head(10)
            print(examples.to_string(index=False))
            
            # Save full list
            save_path = self.output_dir / f"{self.model_name}_misclassified.csv"
            misclassified_data[['Subject_ID', 'Group', 'Predicted', 'Confidence']].to_csv(
                save_path, index=False
            )
            print(f"\n[OK] Full misclassification list saved: {save_path}")
    
    def analyze_confidence(self, save=True):
        """Analyze prediction confidence"""
        print("\n" + "="*80)
        print("Prediction Confidence Analysis")
        print("="*80)
        
        # Get confidence scores (max probability)
        confidence = self.y_proba.max(axis=1)
        
        # Overall statistics
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {confidence.mean():.4f}")
        print(f"  Median: {np.median(confidence):.4f}")
        print(f"  Std: {confidence.std():.4f}")
        print(f"  Min: {confidence.min():.4f}")
        print(f"  Max: {confidence.max():.4f}")
        
        # Confidence by correctness
        correct_mask = self.y_true == self.y_pred
        correct_confidence = confidence[correct_mask]
        incorrect_confidence = confidence[~correct_mask]
        
        print(f"\nConfidence by Correctness:")
        print(f"  Correct predictions: {correct_confidence.mean():.4f} ± {correct_confidence.std():.4f}")
        if len(incorrect_confidence) > 0:
            print(f"  Incorrect predictions: {incorrect_confidence.mean():.4f} ± {incorrect_confidence.std():.4f}")
        
        # Confidence by class
        print(f"\nConfidence by True Class:")
        for i, cls in enumerate(self.classes):
            class_mask = self.y_true == i
            class_confidence = confidence[class_mask]
            print(f"  {cls}: {class_confidence.mean():.4f} ± {class_confidence.std():.4f}")
        
        # Plot confidence distribution
        if save:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Overall confidence distribution
            axes[0, 0].hist(confidence, bins=20, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(confidence.mean(), color='red', linestyle='--', 
                              label=f'Mean: {confidence.mean():.3f}')
            axes[0, 0].set_xlabel('Confidence')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Overall Confidence Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # Confidence by correctness
            axes[0, 1].hist([correct_confidence, incorrect_confidence], 
                           bins=20, label=['Correct', 'Incorrect'],
                           edgecolor='black', alpha=0.7)
            axes[0, 1].set_xlabel('Confidence')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Confidence by Correctness')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
            
            # Confidence by class
            class_confidences = [confidence[self.y_true == i] for i in range(len(self.classes))]
            axes[1, 0].boxplot(class_confidences, labels=self.classes)
            axes[1, 0].set_ylabel('Confidence')
            axes[1, 0].set_title('Confidence by True Class')
            axes[1, 0].grid(alpha=0.3)
            
            # Confidence vs Accuracy
            confidence_bins = np.linspace(0, 1, 11)
            bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
            bin_accuracies = []
            
            for i in range(len(confidence_bins) - 1):
                mask = (confidence >= confidence_bins[i]) & (confidence < confidence_bins[i+1])
                if mask.sum() > 0:
                    bin_acc = (self.y_true[mask] == self.y_pred[mask]).mean()
                    bin_accuracies.append(bin_acc)
                else:
                    bin_accuracies.append(np.nan)
            
            axes[1, 1].plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
            axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Calibration Curve')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
            axes[1, 1].set_xlim([0, 1])
            axes[1, 1].set_ylim([0, 1])
            
            plt.tight_layout()
            
            save_path = self.output_dir / f"{self.model_name}_confidence_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Confidence analysis saved: {save_path}")
            plt.close()
    
    def generate_summary_report(self, metrics, roc_auc):
        """Generate summary report"""
        print("\n" + "="*80)
        print("Summary Report")
        print("="*80)
        
        report = f"""
CNN-RF Model Evaluation Report
{'='*80}

Model: {self.model_name}
Model Path: {self.model_path}
Classes: {', '.join(self.classes)}
Total Samples: {len(self.data)}

Overall Performance:
  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
  Macro F1-Score: {metrics['macro_f1']:.4f}
  Weighted F1-Score: {metrics['weighted_f1']:.4f}
  Macro AUC: {roc_auc['macro']:.4f}

Per-Class Performance:
"""
        
        for i, cls in enumerate(self.classes):
            report += f"""
  {cls}:
    Precision: {metrics['precision'][i]:.4f}
    Recall: {metrics['recall'][i]:.4f}
    F1-Score: {metrics['f1'][i]:.4f}
    AUC: {roc_auc[i]:.4f}
"""
        
        # Class distribution
        report += "\nClass Distribution:\n"
        for cls in self.classes:
            count = (self.y_true_text == cls).sum()
            percentage = count / len(self.data) * 100
            report += f"  {cls}: {count} samples ({percentage:.1f}%)\n"
        
        # Misclassifications
        n_misclassified = (self.y_true != self.y_pred).sum()
        report += f"\nMisclassifications: {n_misclassified} ({n_misclassified/len(self.data)*100:.2f}%)\n"
        
        print(report)
        
        # Save report
        save_path = self.output_dir / f"{self.model_name}_evaluation_report.txt"
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"[OK] Summary report saved: {save_path}")
        
        return report
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*80)
        print(f"CNN-RF Model Evaluation: {self.model_name}")
        print("="*80)
        
        # 1. Basic metrics
        metrics = self.calculate_basic_metrics()
        
        # 2. Classification report
        self.print_classification_report()
        
        # 3. Confusion matrix
        self.plot_confusion_matrix()
        
        # 4. ROC curves and AUC
        roc_auc = self.calculate_roc_auc()
        
        # 5. Misclassification analysis
        self.analyze_misclassifications()
        
        # 6. Confidence analysis
        self.analyze_confidence()
        
        # 7. Summary report
        self.generate_summary_report(metrics, roc_auc)
        
        print("\n" + "="*80)
        print("[SUCCESS] Evaluation completed!")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"\nGenerated files:")
        print(f"  - {self.model_name}_confusion_matrix.png")
        print(f"  - {self.model_name}_roc_curves.png")
        print(f"  - {self.model_name}_confidence_analysis.png")
        print(f"  - {self.model_name}_misclassified.csv")
        print(f"  - {self.model_name}_evaluation_report.txt")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CNN-RF Model")
    parser.add_argument(
        '--model',
        choices=['NC_vs_AD', 'NC_MCI_AD', 'all'],
        default='NC_vs_AD',
        help='Model to evaluate'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        # Evaluate all available models
        for model_name in MODELS.keys():
            if MODELS[model_name]['path'].exists():
                print(f"\n{'='*80}")
                print(f"Evaluating: {model_name}")
                print("="*80)
                
                evaluator = CNNRFEvaluator(model_name)
                evaluator.run_full_evaluation()
            else:
                print(f"\n[SKIP] Model not found: {model_name}")
    else:
        # Evaluate specific model
        evaluator = CNNRFEvaluator(args.model)
        evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
