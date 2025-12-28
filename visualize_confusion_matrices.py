import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(
    y_test,
    preds_dict,
    experiment_name="baseline_before_smote",
    base_dir="outputs/plots"
):
    """
    experiment_name examples:
    - baseline_before_smote
    - baseline_after_smote
    - smote_models
    """

    save_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    for model_name, y_pred in preds_dict.items():
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(4,4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False
        )

        plt.title(f"{model_name} – {experiment_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        file_name = f"{model_name.replace(' ', '_')}_{experiment_name}_cm.png"
        file_path = os.path.join(save_dir, file_name)

        plt.savefig(file_path, dpi=300)
        plt.close()

        print(f"Saved: {file_path}")
