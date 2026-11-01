
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/spotify_cleaned.csv")

print("\n========== MISSING VALUES TREATMENT ==========")

# 1. Check missing values before treatment
print("\n1. Missing values before treatment:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 2. Remove rows with missing Artist
before_rows = len(df)

df = df.dropna(subset=["Artist"])

print("\n2. Missing Artist values removed:")
print("Rows removed:", before_rows - len(df))

# 3. Fill missing numeric values with median
numeric_columns = df.select_dtypes(include="number").columns

for col in numeric_columns:
    median_value = df[col].median()

    if pd.notna(median_value):
        df[col] = df[col].fillna(median_value)

# 4. Fill missing categorical values with "Unknown"
categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

# 5. Check remaining missing values
print("\n3. Missing values after treatment:")
remaining = df.isnull().sum()
print(remaining[remaining > 0])

# 6. Save final dataset
df.to_csv("data/spotify_final.csv", index=False)

print("\n4. FINAL DATASET")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Remaining missing values:", df.isnull().sum().sum())

print("\nFinal dataset saved successfully!")
print("File: data/spotify_final.csv")
