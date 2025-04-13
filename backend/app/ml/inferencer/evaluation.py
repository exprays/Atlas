# Atlas Lynx
# Lynx Evaluation Standards by @exprays

import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(prediction, ground_truth):
    """
    Calculate evaluation metrics for change detection
    
    Args:
        prediction: Binary prediction mask
        ground_truth: Binary ground truth mask
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Flatten arrays
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    
    # Calculate overall accuracy
    accuracy = accuracy_score(gt_flat, pred_flat)
    
    # Calculate Kappa coefficient
    kappa = cohen_kappa_score(gt_flat, pred_flat)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()
    
    # Calculate FI error (False Information error)
    fi_error = fp / (fp + tp + 1e-10)
    
    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'fi_error': fi_error,
        'confusion_matrix': {
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn
        }
    }

def plot_metrics(metrics, output_path=None):
    """
    Create visualizations for evaluation metrics
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_path: Path to save the visualization
        
    Returns:
        Path to saved visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot confusion matrix
    cm = np.array([
        [metrics['confusion_matrix']['true_negative'], metrics['confusion_matrix']['false_positive']],
        [metrics['confusion_matrix']['false_negative'], metrics['confusion_matrix']['true_positive']]
    ])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Change', 'Change'],
                yticklabels=['No Change', 'Change'], ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Plot metrics as bar chart
    metrics_values = [metrics['accuracy'], metrics['kappa'], metrics['fi_error']]
    metrics_names = ['Accuracy', 'Kappa', 'FI Error']
    
    axes[1].bar(metrics_names, metrics_values, color=['green', 'blue', 'red'])
    axes[1].set_title('Evaluation Metrics')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    return fig