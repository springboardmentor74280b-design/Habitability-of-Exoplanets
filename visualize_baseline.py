import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

def plot_baseline_confusion_matrices(y_test, preds_dict, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)

    for model_name, y_pred in preds_dict.items():
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        file_path = f"{save_dir}/{model_name.replace(' ', '_')}_cm.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        print(f"Saved: {file_path}")
