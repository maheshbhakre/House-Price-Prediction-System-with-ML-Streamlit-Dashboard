# serving/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/house_price_model.pkl")

app = FastAPI()

# -----------------------------
# INPUT SCHEMA
# -----------------------------
class HouseInput(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    floors: int
    age: int
    parking: int

    furnishing: str
    location: str

# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
def predict(data: HouseInput):

    # Convert input to dict
    input_data = data.dict()

    # -----------------------------
    # ENCODE INPUT
    # -----------------------------
    encoded = {
        "area": input_data["area"],
        "bedrooms": input_data["bedrooms"],
        "bathrooms": input_data["bathrooms"],
        "floors": input_data["floors"],
        "age": input_data["age"],
        "parking": input_data["parking"],

        "furnishing_semi-furnished": 1 if input_data["furnishing"] == "semi-furnished" else 0,
        "furnishing_unfurnished": 1 if input_data["furnishing"] == "unfurnished" else 0,

        "location_suburban": 1 if input_data["location"] == "suburban" else 0,
        "location_urban": 1 if input_data["location"] == "urban" else 0,
    }

    df = pd.DataFrame([encoded])

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["house_age_score"] = 30 - df["age"]
    df["room_density"] = df["total_rooms"] / df["area"]
    df["parking_per_room"] = df["parking"] / (df["total_rooms"] + 1)

    # -----------------------------
    # PREDICT
    # -----------------------------
    prediction = model.predict(df)[0]

    return {
        "predicted_price": int(prediction)
    }