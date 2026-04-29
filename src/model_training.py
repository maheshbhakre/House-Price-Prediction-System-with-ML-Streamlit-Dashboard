# src/model_training.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. LOAD FEATURED DATA
# -----------------------------
df = pd.read_csv("data/houses_featured.csv")

print("Dataset Shape:", df.shape)

# -----------------------------
# 2. REMOVE DATA LEAKAGE
# -----------------------------
# Drop target leakage feature
if "price_per_sqft" in df.columns:
    df = df.drop("price_per_sqft", axis=1)

# -----------------------------
# 3. SPLIT FEATURES & TARGET
# -----------------------------
X = df.drop("price", axis=1)
y = df["price"]

print("Feature Columns:\n", X.columns)

# -----------------------------
# 4. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain Size:", X_train.shape)
print("Test Size:", X_test.shape)

# -----------------------------
# 5. TRAIN MODELS
# -----------------------------
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -----------------------------
# 6. EVALUATION FUNCTION
# -----------------------------
def evaluate(model, name):
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"\n{name} Performance:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

# -----------------------------
# 7. EVALUATE MODELS
# -----------------------------
evaluate(lr, "Linear Regression")
evaluate(rf, "Random Forest")

# -----------------------------
# 8. SAVE BEST MODEL
# -----------------------------
import joblib

joblib.dump(rf, "models/house_price_model.pkl")

print("\nModel saved at: models/house_price_model.pkl")