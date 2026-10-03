import pandas as pd
import numpy as np
from scipy.stats import zscore

# Retail sales data
data = {
    "Day": range(1, 11),
    "Revenue": [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53]
}

df = pd.DataFrame(data)

# 1. Check missing values
print("Missing values in Revenue:", df["Revenue"].isnull().sum())

# 2. Fill missing values using median
median_revenue = df["Revenue"].median()
df["Revenue"] = df["Revenue"].fillna(median_revenue)

print("\nAfter median fill:")
print(df)

# 3. Min-Max normalization
df["Revenue_MinMax"] = (
    (df["Revenue"] - df["Revenue"].min())
    / (df["Revenue"].max() - df["Revenue"].min())
)

# 4. Z-score standardization
df["Revenue_Zscore"] = zscore(df["Revenue"])

# 5. IQR-based outlier detection
Q1 = df["Revenue"].quantile(0.25)
Q3 = df["Revenue"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df["Outlier"] = (
    (df["Revenue"] < lower_bound)
    | (df["Revenue"] > upper_bound)
)

# Print outlier days
outlier_days = df.loc[df["Outlier"], "Day"].tolist()

print("\nIQR bounds:", lower_bound, "-", upper_bound)
print("IQR outliers detected on day(s):", outlier_days)

# 6. Final table
print("\nFinal table:")
print(df[["Day", "Revenue", "Revenue_MinMax", "Revenue_Zscore", "Outlier"]])