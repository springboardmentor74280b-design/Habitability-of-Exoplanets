# ==============================
# EXOPLANET HABITABILITY PROJECT
# Complete Python Script
# ==============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 2. Load Dataset
# ------------------------------
file_path = "phl_exoplanet_catalog_2019.csv"

df = pd.read_csv(file_path)
print("Dataset Loaded Successfully\n")
print(df.head())

# ------------------------------
# 3. Dataset Information
# ------------------------------
print("\nDataset Info:")
df.info()

# ------------------------------
# 4. Descriptive Statistics
# ------------------------------
print("\nDescriptive Statistics:")
print(df.describe())

# ------------------------------
# 5. CREATE HABITABILITY COLUMN
# (Because dataset has NO P.Habitable column)
# ------------------------------

df['P.Habitable'] = np.where(
    (df['P_RADIUS_EST'] >= 0.5) & (df['P_RADIUS_EST'] <= 2.0) &
    (df['P_SEMI_MAJOR_AXIS_EST'] >= 0.5) & (df['P_SEMI_MAJOR_AXIS_EST'] <= 1.5),
    1, 0
)

print("\nHabitability column created successfully")

# ------------------------------
# 6. Habitability Percentage
# ------------------------------
print("\nHabitability Percentage:")
habitability_percentage = df['P.Habitable'].value_counts(normalize=True) * 100
print(habitability_percentage)

# ------------------------------
# 7. Missing Value Check
# ------------------------------
print("\nMissing Values:")
print(df.isna().sum())

# ------------------------------
# 8. Select Important Features
# ------------------------------
features = [
    'P_RADIUS_EST',
    'P_MASS_EST',
    'P_SEMI_MAJOR_AXIS_EST',
    'P.Habitable'
]

df = df[features].copy()

# ------------------------------
# 9. Handle Missing Values
# ------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print("\nMissing Values After Cleaning:")
print(df.isna().sum())

# ------------------------------
# 10. Univariate Analysis
# ------------------------------
plt.figure()
plt.hist(df['P_RADIUS_EST'], bins=30)
plt.xlabel("Planet Radius (Earth)")
plt.ylabel("Frequency")
plt.title("Planet Radius Distribution")
plt.show()

plt.figure()
plt.boxplot(df['P_SEMI_MAJOR_AXIS_EST'])
plt.ylabel("Semi-Major Axis (AU)")
plt.title("Orbital Distance Distribution")
plt.show()

# ------------------------------
# 11. Bivariate Analysis
# ------------------------------
plt.figure()
plt.scatter(df['P_RADIUS_EST'], df['P_SEMI_MAJOR_AXIS_EST'])
plt.xlabel("Planet Radius (Earth)")
plt.ylabel("Orbital Distance (AU)")
plt.title("Radius vs Orbital Distance")
plt.show()

plt.figure()
sns.boxplot(x='P.Habitable', y='P_RADIUS_EST', data=df)
plt.title("Planet Radius vs Habitability")
plt.show()

# ------------------------------
# 12. Correlation Heatmap
# ------------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------
# 13. Feature Engineering
# ------------------------------
df['Habitability_Score'] = (
    1 / df['P_RADIUS_EST'].replace(0, np.nan) +
    1 / df['P_SEMI_MAJOR_AXIS_EST'].replace(0, np.nan)
)

print("\nFeature Engineering Completed")
print(df.head())

# ------------------------------
# 14. Save Cleaned Dataset
# ------------------------------
output_path = "exoplanet_cleaned_final.csv"
df.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved as {output_path}")
print("\nPROJECT EXECUTION COMPLETED SUCCESSFULLY 🎉")
