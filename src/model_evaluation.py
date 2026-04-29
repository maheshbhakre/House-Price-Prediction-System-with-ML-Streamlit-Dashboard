# src/model_evaluation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data/houses_featured.csv")

# Remove leakage feature
if "price_per_sqft" in df.columns:
    df = df.drop("price_per_sqft", axis=1)

X = df.drop("price", axis=1)
y = df["price"]

# Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 2. LOAD MODEL
# -----------------------------
model = joblib.load("models/house_price_model.pkl")

# -----------------------------
# 3. PREDICTIONS
# -----------------------------
pred = model.predict(X_test)

# -----------------------------
# 4. METRICS
# -----------------------------
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# -----------------------------
# 5. ACTUAL vs PREDICTED PLOT
# -----------------------------
plt.figure()
sns.scatterplot(x=y_test, y=pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")

plt.savefig("images/actual_vs_predicted.png")
plt.close()

# -----------------------------
# 6. RESIDUAL PLOT
# -----------------------------
residuals = y_test - pred

plt.figure()
sns.histplot(residuals, bins=30, kde=True)

plt.title("Residual Distribution")

plt.savefig("images/residuals.png")
plt.close()

# -----------------------------
# 7. ERROR ANALYSIS
# -----------------------------
error_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred,
    "Error": residuals
})

error_df.to_csv("outputs/error_analysis.csv", index=False)

print("\nError analysis saved at: outputs/error_analysis.csv")
print("Plots saved in images/")