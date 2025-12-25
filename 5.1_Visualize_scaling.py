import matplotlib.pyplot as plt
import os

def save_scaling_plot(
    X_before,
    X_after,
    feature_name,
    save_dir="outputs/plots"
):
    # Create folder if not exists
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.hist(X_before[feature_name], bins=30)
    plt.title("Before Scaling")

    plt.subplot(1,2,2)
    plt.hist(X_after[feature_name], bins=30)
    plt.title("After Scaling")

    plt.tight_layout()

    # Save figure
    file_path = os.path.join(save_dir, f"{feature_name}_scaling.png")
    plt.savefig(r"C:\Users\Menaka\OneDrive\Desktop\INFOSYS_SPRINGBOARD_PROJECT\plot", dpi=300)
    plt.close()   # VERY important

    print(f"Plot saved at: {file_path}")
