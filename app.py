# app.py

import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/house_price_model.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.title("🏠 House Price Prediction System")
st.markdown("Estimate property value using machine learning")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sqft)", 500, 5000, 2000)
    bedrooms = st.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.slider("Bathrooms", 1, 3, 2)

with col2:
    floors = st.slider("Floors", 1, 2, 1)
    age = st.slider("Age", 0, 30, 5)
    parking = st.slider("Parking", 0, 2, 1)

furnishing = st.selectbox("Furnishing", ["furnished", "semi-furnished", "unfurnished"])
location = st.selectbox("Location", ["urban", "suburban", "rural"])

# -----------------------------
# PREPARE DATA
# -----------------------------
input_data = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "floors": floors,
    "age": age,
    "parking": parking,
    "furnishing_semi-furnished": 1 if furnishing == "semi-furnished" else 0,
    "furnishing_unfurnished": 1 if furnishing == "unfurnished" else 0,
    "location_suburban": 1 if location == "suburban" else 0,
    "location_urban": 1 if location == "urban" else 0,
}

df = pd.DataFrame([input_data])

# Feature Engineering
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
df["house_age_score"] = 30 - df["age"]
df["room_density"] = df["total_rooms"] / df["area"]
df["parking_per_room"] = df["parking"] / (df["total_rooms"] + 1)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Price"):
    price = model.predict(df)[0]
    st.success(f"Estimated Price: ₹{int(price):,}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built using Machine Learning • Streamlit Dashboard")