
import pandas as pd
import numpy as np

# Load Spotify dataset
df = pd.read_csv("data/spotify.csv", encoding="latin1", low_memory=False)

print("\n" + "=" * 50)
print("       SPOTIFY MUSIC TRENDS - EDA PROJECT")
print("=" * 50)

# STEP 1: First 5 rows
print("\n1. FIRST 5 SONGS")
print(df.head().to_string())

# STEP 2: Dataset dimensions
print("\n2. DATASET SHAPE")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# STEP 3: Column names
print("\n3. COLUMN NAMES")
for column in df.columns:
    print("-", column)

# STEP 4: Data types
print("\n4. DATA TYPES")
print(df.dtypes.to_string())

# STEP 5: Statistical summary
print("\n5. STATISTICAL SUMMARY")
print(df.describe(include="all").to_string())

# STEP 6: Dataset information
print("\n6. DATASET INFORMATION")
df.info()

print("\n" + "=" * 50)
print("       STEP 1 COMPLETED SUCCESSFULLY!")
print("=" * 50)
