import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/spotify_cleaned.csv")

# Select two numeric columns
columns = ["Track Score", "Spotify Streams"]

# Convert Spotify Streams to numeric
df["Spotify Streams"] = pd.to_numeric(
    df["Spotify Streams"], errors="coerce"
)

# Create boxplots
plt.figure(figsize=(12, 5))

for i, col in enumerate(columns, 1):
    plt.subplot(1, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f"Outliers in {col}")

plt.tight_layout()
plt.savefig("outliers_boxplots.png", dpi=200)
plt.show()

# Calculate outliers using IQR
for col in columns:
    data = df[col].dropna()

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = data[(data < lower) | (data > upper)]

    print(f"\n{col}")
    print("Number of outliers:", len(outliers))
    print("Lower limit:", lower)
    print("Upper limit:", upper)

print("\nOutlier analysis completed!")