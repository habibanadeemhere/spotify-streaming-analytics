
import pandas as pd
import matplotlib.pyplot as plt

# Load original dataset
df = pd.read_csv("data/spotify.csv", encoding="latin1", low_memory=False)

print("\n========== DATA CLEANING ==========")

# 1. Missing values BEFORE cleaning
print("\n1. MISSING VALUES")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# 2. Missing values graph
plt.figure(figsize=(12, 6))
missing[missing > 0].sort_values().plot(
    kind="barh", color="skyblue"
)
plt.title("Missing Values Before Data Cleaning")
plt.xlabel("Number of Missing Values")
plt.ylabel("Columns")
plt.tight_layout()
plt.savefig("missing_values.png", dpi=200)
plt.show()

# 3. Duplicate rows
print("\n2. DUPLICATE ROWS")
duplicates = df.duplicated().sum()
print("Duplicate rows found:", duplicates)

df = df.drop_duplicates()
print("Rows after removing duplicates:", len(df))

# 4. Fix numeric columns stored as text
numeric_columns = [
    "All Time Rank",
    "Spotify Streams",
    "Spotify Playlist Count",
    "Spotify Playlist Reach",
    "YouTube Views",
    "YouTube Likes",
    "TikTok Posts",
    "TikTok Likes",
    "TikTok Views",
    "YouTube Playlist Reach",
    "AirPlay Spins",
    "SiriusXM Spins",
    "Deezer Playlist Reach",
    "Pandora Streams",
    "Pandora Track Stations",
    "Soundcloud Streams",
    "Shazam Counts"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

print("\n3. NUMERIC DATA TYPES FIXED")

# 5. Convert release date
df["Release Date"] = pd.to_datetime(
    df["Release Date"], errors="coerce"
)

print("\n4. RELEASE DATE CONVERTED")
print(df["Release Date"].dtype)

# 6. Remove completely empty columns
empty_columns = df.columns[df.isnull().all()].tolist()

print("\n5. COMPLETELY EMPTY COLUMNS")
print(empty_columns)

df = df.drop(columns=empty_columns)

# 7. Save cleaned dataset
df.to_csv("data/spotify_cleaned.csv", index=False)

print("\n6. FINAL DATASET")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nCleaned data saved successfully!")
print("File: data/spotify_cleaned.csv")
