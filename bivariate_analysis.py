import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import combinations
from sklearn.linear_model import LinearRegression

# ---------------------------
# Step 1: Load CSV
# ---------------------------
file_name = "exoplanet dataset.csv"  # Replace with your CSV file name
df = pd.read_csv("exoplanet dataset.csv")

# Create output directories
os.makedirs("scatter_plots", exist_ok=True)
os.makedirs("regression_plots", exist_ok=True)

# ---------------------------
# Step 2: Identify numeric columns
# ---------------------------
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
print("Numeric columns identified for analysis:\n", numeric_cols)

# ---------------------------
# Step 3: Correlation matrix
# ---------------------------
corr_matrix = df[numeric_cols].corr()
corr_matrix.to_csv("correlation_matrix.csv")
print("Correlation matrix saved as 'correlation_matrix.csv'")

# ---------------------------
# Step 4: Find strongly correlated pairs
# ---------------------------
threshold = 0.5  # correlation threshold
strong_pairs = []

for col_x, col_y in combinations(numeric_cols, 2):
    corr = corr_matrix.loc[col_x, col_y]
    if abs(corr) >= threshold:
        strong_pairs.append((col_x, col_y, corr))

print(f"Found {len(strong_pairs)} strongly correlated pairs (|corr| >= {threshold})")

# ---------------------------
# Step 5: Generate plots and regression info
# ---------------------------
summary_data = []

for col_x, col_y, corr in strong_pairs:
    # Scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(df[col_x], df[col_y], alpha=0.5)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f"Scatter: {col_y} vs {col_x} (corr={corr:.2f})")
    plt.tight_layout()
    plt.savefig(f"scatter_plots/{col_y}_vs_{col_x}.png")
    plt.close()

    # Linear regression
    X = df[[col_x]].dropna()
    y = df[col_y].dropna()
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    slope = intercept = None
    if len(X) > 1:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        slope = model.coef_[0]
        intercept = model.intercept_

        # Regression plot
        plt.figure(figsize=(6, 4))
        plt.scatter(X, y, alpha=0.5)
        plt.plot(X, y_pred, color="red")
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.title(f"Regression: {col_y} vs {col_x} (corr={corr:.2f})")
        plt.tight_layout()
        plt.savefig(f"regression_plots/{col_y}_vs_{col_x}_regression.png")
        plt.close()

    # Add to summary
    summary_data.append({
        "Variable 1": col_x,
        "Variable 2": col_y,
        "Correlation": corr,
        "Regression Slope": slope,
        "Regression Intercept": intercept
    })

# ---------------------------
# Step 6: Save summary CSV
# ---------------------------
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("bivariate_summary.csv", index=False)
print("Summary CSV saved as 'bivariate_summary.csv'")
print("All strongly correlated plots saved in 'scatter_plots' and 'regression_plots' folders")
