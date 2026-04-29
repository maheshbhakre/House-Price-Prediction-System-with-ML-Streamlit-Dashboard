import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "area": np.random.randint(500, 5000, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 4, n),
    "floors": np.random.randint(1, 3, n),
    "age": np.random.randint(0, 30, n),
    "parking": np.random.randint(0, 3, n),
    "furnishing": np.random.choice(["unfurnished", "semi-furnished", "furnished"], n),
    "location": np.random.choice(["urban", "suburban", "rural"], n)
})

# Price logic (hidden pattern)
data["price"] = (
    data["area"] * 3000 +
    data["bedrooms"] * 50000 +
    data["bathrooms"] * 30000 +
    data["floors"] * 40000 -
    data["age"] * 10000 +
    data["parking"] * 20000 +
    np.where(data["location"] == "urban", 200000, 0) +
    np.where(data["location"] == "suburban", 100000, 0) +
    np.where(data["furnishing"] == "furnished", 150000, 0) +
    np.random.randint(-50000, 50000, n)
)

# Save dataset
data.to_csv("data/houses.csv", index=False)

print("Dataset created successfully!")
print(data.head())