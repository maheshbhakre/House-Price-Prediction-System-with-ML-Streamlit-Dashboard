# src/predict.py

import pandas as pd
import joblib

# -----------------------------
# 1. LOAD MODEL
# -----------------------------
model = joblib.load("models/house_price_model.pkl")

# -----------------------------
# 2. USER INPUT (MANUAL)
# -----------------------------
# You can modify these values

input_data = {
    "area": 2000,
    "bedrooms": 3,
    "bathrooms": 2,
    "floors": 1,
    "age": 5,
    "parking": 1,
    
    # encoded columns (IMPORTANT)
    "furnishing_semi-furnished": 1,
    "furnishing_unfurnished": 0,
    "location_suburban": 1,
    "location_urban": 0,

    # engineered features
    "total_rooms": 5,
    "house_age_score": 25,
    "room_density": 5 / 2000,
    "parking_per_room": 1 / 6
}

# -----------------------------
# 3. CONVERT TO DATAFRAME
# -----------------------------
input_df = pd.DataFrame([input_data])

# -----------------------------
# 4. PREDICT
# -----------------------------
prediction = model.predict(input_df)

print("\n Predicted House Price:", int(prediction[0]))