import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

from Data_Loader import load_and_split_data

# ---------------------------
# Load data
# ---------------------------
X_train, X_test, y_train, y_test = load_and_split_data(
    "train_smote_exoplanet.csv",
    target_col="P_HABITABLE"
)

# ---------------------------
# KEEP ONLY NUMERIC COLUMNS
# ---------------------------
X_train = X_train.select_dtypes(include=['int64', 'float64'])
X_test = X_test.select_dtypes(include=['int64', 'float64'])

# ---------------------------
# Logistic Regression
# ---------------------------
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])

lr_pipeline.fit(X_train, y_train)
y_lr_pred = lr_pipeline.predict(X_test)

# ---------------------------
# KNN
# ---------------------------
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'
    ))
])

knn_pipeline.fit(X_train, y_train)
y_knn_pred = knn_pipeline.predict(X_test)

# ---------------------------
# Naive Bayes
# ---------------------------
nb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nb', GaussianNB())
])

nb_pipeline.fit(X_train, y_train)
y_nb_pred = nb_pipeline.predict(X_test)

# ---------------------------
# Reports
# ---------------------------
print("Logistic Regression:\n", classification_report(y_test, y_lr_pred))
print("KNN:\n", classification_report(y_test, y_knn_pred))
print("Naive Bayes:\n", classification_report(y_test, y_nb_pred))

#from visualize_confusion_matrices import plot_baseline_confusion_matrices

from visualize_confusion_matrices import plot_confusion_matrices


smote_preds = {
    "Logistic Regression": y_lr_pred,
    "KNN": y_knn_pred,
    "Naive Bayes": y_nb_pred
}

plot_confusion_matrices(
    y_test,
    smote_preds,
    experiment_name="baseline_after_smote"
)
