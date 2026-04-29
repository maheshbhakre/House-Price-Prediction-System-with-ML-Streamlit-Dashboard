# notebooks/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. LOAD CLEAN DATA
# -----------------------------
df = pd.read_csv("data/houses_clean.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# -----------------------------
# 2. BASIC INFO
# -----------------------------
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# -----------------------------
# 3. CORRELATION HEATMAP
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")

# SAVE IMAGE HERE 👇
plt.savefig("images/heatmap.png")
plt.close()

# -----------------------------
# 4. PRICE DISTRIBUTION
# -----------------------------
plt.figure()
sns.histplot(df["price"], bins=30, kde=True)

plt.title("Price Distribution")

# SAVE IMAGE HERE 👇
plt.savefig("images/price_distribution.png")
plt.close()

# -----------------------------
# 5. AREA VS PRICE
# -----------------------------
plt.figure()
sns.scatterplot(x=df["area"], y=df["price"])

plt.title("Area vs Price")

# SAVE IMAGE HERE 👇
plt.savefig("images/area_vs_price.png")
plt.close()

# -----------------------------
# 6. BEDROOMS VS PRICE
# -----------------------------
plt.figure()
sns.boxplot(x=df["bedrooms"], y=df["price"])

plt.title("Bedrooms vs Price")

# SAVE IMAGE HERE 👇
plt.savefig("images/bedrooms_vs_price.png")
plt.close()

# -----------------------------
# 7. PARKING VS PRICE
# -----------------------------
plt.figure()
sns.boxplot(x=df["parking"], y=df["price"])

plt.title("Parking vs Price")

# SAVE IMAGE HERE 👇
plt.savefig("images/parking_vs_price.png")
plt.close()

print("\nEDA completed. Check images/ folder.")