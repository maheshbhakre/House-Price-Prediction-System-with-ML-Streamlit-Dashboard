# src/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data/houses_featured.csv")

# Remove leakage feature
if "price_per_sqft" in df.columns:
    df = df.drop("price_per_sqft", axis=1)

# -----------------------------
# 2. LOAD MODEL
# -----------------------------
model = joblib.load("models/house_price_model.pkl")

X = df.drop("price", axis=1)

# -----------------------------
# 3. FEATURE IMPORTANCE (RF)
# -----------------------------
importances = model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# -----------------------------
# 4. PLOT FEATURE IMPORTANCE
# -----------------------------
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance_df)

plt.title("Feature Importance")
plt.savefig("images/feature_importance.png")
plt.close()

# -----------------------------
# 5. TOP FEATURES PRINT
# -----------------------------
print("\nTop 10 Important Features:")
print(importance_df.head(10))

# -----------------------------
# 6. PRICE VS TOP FEATURES
# -----------------------------
top_features = importance_df.head(3)["Feature"].tolist()

for feature in top_features:
    plt.figure()
    sns.scatterplot(x=df[feature], y=df["price"])

    plt.title(f"{feature} vs Price")
    plt.savefig(f"images/{feature}_vs_price.png")
    plt.close()

# -----------------------------
# 7. SAVE IMPORTANCE DATA
# -----------------------------
importance_df.to_csv("outputs/feature_importance.csv", index=False)

print("\nFeature importance saved at: outputs/feature_importance.csv")
print("Graphs saved in images/")