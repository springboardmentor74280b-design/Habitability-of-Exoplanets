from Data_Loader import load_and_split_data

X_train, X_test, y_train, y_test = load_and_split_data(
    r"C:\Users\Menaka\OneDrive\Desktop\Habitability-of-Exoplanets\cleaned_exoplanet_dataset.csv",
    target_col="P_HABITABLE"
)

#Separate numerical & categorical columns
numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

#Create preprocessing pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

#Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

################# visualization ##############

import matplotlib.pyplot as plt
import numpy as np

# Pick a numerical feature
feature = numerical_cols[0]
feature_idx = list(numerical_cols).index(feature)

# Before scaling
plt.hist(X_train[feature], bins=30)
plt.title(f"Before Scaling: {feature}")
plt.show()

# After scaling (numerical part only)
num_scaled = X_train_processed[:, :len(numerical_cols)]
num_scaled_dense = num_scaled.toarray()

plt.hist(num_scaled_dense[:, feature_idx], bins=30)
plt.title(f"After Scaling: {feature}")
plt.show()

######## verification (Mean ≈ 0, Std ≈ 1) #########
import numpy as np


num_data = X_train_processed[:, :len(numerical_cols)]

# Convert sparse → dense (numerical part only)
num_data_dense = num_data.toarray()

print("Mean after scaling:", np.mean(num_data_dense, axis=0))
print("Std after scaling:", np.std(num_data_dense, axis=0))

#Get feature names
num_features = numerical_cols.tolist()
cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)

all_features = num_features + cat_features.tolist()

#Convert to DataFrame
import pandas as pd

X_train_df = pd.DataFrame(
    X_train_processed.toarray(),
    columns=all_features
)

X_train_df["P_HABITABLE"] = y_train.values

X_train_df.to_csv("scaled_exoplanet_dataset.csv", index=False)
print("Scaled dataset saved successfully!")
