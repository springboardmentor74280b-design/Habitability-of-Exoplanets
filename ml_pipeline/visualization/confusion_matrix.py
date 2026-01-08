import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np

def plot_model_results(results_dict, save_path="Outputs/model_comparison.png"):
    """
    results_dict: dict where key is 'Model Name' and value is (y_test, y_pred)
    """
    n_models = len(results_dict)
    cols = 1 if n_models == 1 else 2
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)
    axes = axes.flatten()
    
    for i, (name, (y_test, y_pred)) in enumerate(results_dict.items()):
        ax = axes[i]
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        # Extract weighted avg or macro avg or accuracy to display
        accuracy = report['accuracy']
        weighted = report['weighted avg']
        metrics_text = (
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision (W): {weighted['precision']:.4f}\n"
            f"Recall (W): {weighted['recall']:.4f}\n"
            f"F1 Score (W): {weighted['f1-score']:.4f}"
        )
        
        # Add text box for metrics
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        # Shifted to the right of the plot to avoid obscuring cells
        ax.text(1.1, 0.5, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', horizontalalignment='left', bbox=props)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")
