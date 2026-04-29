# src/feature_engineering.py

import pandas as pd

# -----------------------------
# 1. LOAD CLEAN DATA
# -----------------------------
df = pd.read_csv("data/houses_clean.csv")

print("Original Columns:\n", df.columns)

# -----------------------------
# 2. CREATE NEW FEATURES
# -----------------------------

# Total rooms proxy
df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

# Price per sqft proxy
df["price_per_sqft"] = df["price"] / df["area"]

# Age impact feature (invert age → newer house more valuable)
df["house_age_score"] = 30 - df["age"]

# Bedroom density (rooms per area)
df["room_density"] = df["total_rooms"] / df["area"]

# Parking value interaction
df["parking_per_room"] = df["parking"] / (df["total_rooms"] + 1)

# -----------------------------
# 3. DROP IRRELEVANT / REDUNDANT (OPTIONAL)
# -----------------------------
# (We keep everything for now)

# -----------------------------
# 4. SAVE FEATURED DATA
# -----------------------------
df.to_csv("data/houses_featured.csv", index=False)

print("\nNew Columns Added:")
print(df.columns)

print("\nFeature engineered dataset saved at: data/houses_featured.csv")