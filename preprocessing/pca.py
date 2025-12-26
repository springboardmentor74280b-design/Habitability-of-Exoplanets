from sklearn.decomposition import PCA

def apply_pca(train, test, variance=0.95):
    pca = PCA(n_components=variance, random_state=42)
    X_train_pca = pca.fit_transform(train)
    X_test_pca = pca.transform(test)
    return X_train_pca, X_test_pca, pca
