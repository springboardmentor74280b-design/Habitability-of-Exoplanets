from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Menaka\OneDrive\Desktop\INFOSYS_SPRINGBOARD_PROJECT\cleaned_exoplanet_dataset.csv")
X = df.drop(columns=["P_HABITABLE"])
y = df["P_HABITABLE"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,       # 80:20 split
    random_state=42,
    stratify=y            # preserves class ratio
)

print("Original distribution:")
print(y.value_counts(normalize=True))

print("\nTraining distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting distribution:")
print(y_test.value_counts(normalize=True))

 
########### Visualization ###########

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.countplot(x=y, ax=axes[0])
axes[0].set_title("Original Dataset")

sns.countplot(x=y_train, ax=axes[1])
axes[1].set_title("Training Set")

sns.countplot(x=y_test, ax=axes[2])
axes[2].set_title("Testing Set")

plt.tight_layout()
plt.show()

