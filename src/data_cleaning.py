# src/data_cleaning.py

import pandas as pd

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data/houses.csv")

print("Initial Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# -----------------------------
# 2. CHECK MISSING VALUES
# -----------------------------
print("\nMissing Values:\n", df.isnull().sum())

# -----------------------------
# 3. REMOVE DUPLICATES
# -----------------------------
df = df.drop_duplicates()
print("\nShape after removing duplicates:", df.shape)

# -----------------------------
# 4. ENCODE CATEGORICAL DATA
# -----------------------------
df = pd.get_dummies(df, columns=["furnishing", "location"], drop_first=True)

print("\nColumns after encoding:\n", df.columns)

# -----------------------------
# 5. REMOVE OUTLIERS
# -----------------------------
df = df[df["area"] < df["area"].quantile(0.99)]
df = df[df["price"] < df["price"].quantile(0.99)]

print("\nShape after outlier removal:", df.shape)

# -----------------------------
# 6. SAVE CLEAN DATA
# -----------------------------
df.to_csv("data/houses_clean.csv", index=False)

print("\n Clean dataset saved at: data/houses_clean.csv")