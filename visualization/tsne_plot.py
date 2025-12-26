from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(X, y, save_path="Outputs/tsne_plot.png"):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap="viridis", s=5, alpha=0.6)
    plt.title("t-SNE Visualization")
    plt.colorbar(label='Target Class')
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")
