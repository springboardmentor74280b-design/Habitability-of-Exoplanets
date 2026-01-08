from sklearn.svm import SVC

def train_rbf_svm(X_train, y_train):
    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
