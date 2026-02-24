import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
    
    def plot_confusion_matrix(self, save_path='models/confusion_matrix.png'):
        """Plot confusion matrix"""
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Unresolved', 'Resolved'],
                    yticklabels=['Unresolved', 'Resolved'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(self, save_path='models/roc_curve.png'):
        """Plot ROC-AUC curve"""
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC curve saved to {save_path}")
    
    def plot_prediction_distribution(self, save_path='models/prediction_distribution.png'):
        """Plot distribution of predicted probabilities"""
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        plt.figure(figsize=(10, 6))
        
        # Resolved cases
        plt.hist(y_pred_proba[self.y_test == 1], bins=30, alpha=0.6, label='Resolved', color='green')
        # Unresolved cases
        plt.hist(y_pred_proba[self.y_test == 0], bins=30, alpha=0.6, label='Unresolved', color='red')
        
        plt.xlabel('Predicted Probability of Resolution')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Prediction distribution plot saved to {save_path}")
    
    def generate_report(self, save_path='models/evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                     f1_score, roc_auc_score, classification_report)
        
        report_text = f"""
{'='*80}
COMPREHENSIVE MODEL EVALUATION REPORT
{'='*80}

TEST SET PERFORMANCE METRICS:
{'-'*80}
Accuracy:      {accuracy_score(self.y_test, y_pred):.4f}
Precision:     {precision_score(self.y_test, y_pred):.4f}
Recall:        {recall_score(self.y_test, y_pred):.4f}
F1-Score:      {f1_score(self.y_test, y_pred):.4f}
ROC-AUC Score: {roc_auc_score(self.y_test, y_pred_proba):.4f}

CLASSIFICATION REPORT:
{'-'*80}
{classification_report(self.y_test, y_pred, target_names=['Unresolved', 'Resolved'])}

CONFUSION MATRIX INTERPRETATION:
{'-'*80}
True Negatives:  {confusion_matrix(self.y_test, y_pred)[0, 0]}
False Positives: {confusion_matrix(self.y_test, y_pred)[0, 1]}
False Negatives: {confusion_matrix(self.y_test, y_pred)[1, 0]}
True Positives:  {confusion_matrix(self.y_test, y_pred)[1, 1]}

INTERPRETATION:
- The model correctly identifies {accuracy_score(self.y_test, y_pred)*100:.2f}% of cases.
- When the model predicts a case will be resolved, it's correct {precision_score(self.y_test, y_pred)*100:.2f}% of the time.
- The model identifies {recall_score(self.y_test, y_pred)*100:.2f}% of all actually resolved cases.

{'='*80}
"""
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Evaluation report saved to {save_path}")
        print(report_text)