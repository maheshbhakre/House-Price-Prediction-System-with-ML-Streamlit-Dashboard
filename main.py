import pandas as pd

# Load dataset
df = pd.read_csv("data/houses.csv")

# Print shape (rows, columns)
print("Shape of dataset:", df.shape)

# Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())