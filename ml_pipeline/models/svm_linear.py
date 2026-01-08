from sklearn.svm import LinearSVC

def train_linear_svm(X_train, y_train):
    model = LinearSVC(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model
