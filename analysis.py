
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load final dataset
df = pd.read_csv("data/spotify_final.csv")

sns.set_theme(style="darkgrid")

print("\n========== SPOTIFY EDA ANALYSIS ==========")

# QUESTION 1: Top 10 songs by Spotify streams
print("\n1. TOP 10 SONGS BY SPOTIFY STREAMS")

top_songs = df.nlargest(10, "Spotify Streams")

print(top_songs[["Track", "Artist", "Spotify Streams"]].to_string(index=False))

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_songs,
    x="Spotify Streams",
    y="Track"
)
plt.title("Top 10 Songs by Spotify Streams")
plt.tight_layout()
plt.savefig("question1_top_songs.png", dpi=200)
plt.show()


# QUESTION 2: Top 10 artists by number of songs
print("\n2. TOP 10 ARTISTS BY NUMBER OF SONGS")

top_artists = df["Artist"].value_counts().head(10)
print(top_artists)

plt.figure(figsize=(12, 6))
top_artists.plot(kind="bar")
plt.title("Top 10 Artists by Number of Songs")
plt.xlabel("Artist")
plt.ylabel("Number of Songs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("question2_top_artists.png", dpi=200)
plt.show()


# QUESTION 3: Spotify popularity distribution
print("\n3. SPOTIFY POPULARITY DISTRIBUTION")

print(df["Spotify Popularity"].describe())

plt.figure(figsize=(10, 5))
sns.histplot(df["Spotify Popularity"], bins=20, kde=True)
plt.title("Distribution of Spotify Popularity")
plt.tight_layout()
plt.savefig("question3_popularity.png", dpi=200)
plt.show()


# QUESTION 4: Track Score vs Spotify Streams
print("\n4. TRACK SCORE VS SPOTIFY STREAMS")

correlation = df["Track Score"].corr(df["Spotify Streams"])
print("Correlation:", round(correlation, 3))

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Track Score",
    y="Spotify Streams"
)
plt.title("Track Score vs Spotify Streams")
plt.tight_layout()
plt.savefig("question4_score_streams.png", dpi=200)
plt.show()


# QUESTION 5: YouTube Views vs Spotify Streams
print("\n5. YOUTUBE VIEWS VS SPOTIFY STREAMS")

youtube_corr = df["YouTube Views"].corr(df["Spotify Streams"])
print("Correlation:", round(youtube_corr, 3))

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="YouTube Views",
    y="Spotify Streams"
)
plt.title("YouTube Views vs Spotify Streams")
plt.tight_layout()
plt.savefig("question5_youtube_spotify.png", dpi=200)
plt.show()


# QUESTION 6: Top 10 songs by TikTok views
print("\n6. TOP 10 SONGS BY TIKTOK VIEWS")

top_tiktok = df.nlargest(10, "TikTok Views")

print(top_tiktok[["Track", "Artist", "TikTok Views"]].to_string(index=False))

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_tiktok,
    x="TikTok Views",
    y="Track"
)
plt.title("Top 10 Songs by TikTok Views")
plt.tight_layout()
plt.savefig("question6_tiktok_views.png", dpi=200)
plt.show()


# QUESTION 7: Explicit vs Non-Explicit Songs
print("\n7. EXPLICIT VS NON-EXPLICIT SONGS")

explicit_comparison = df.groupby("Explicit Track")[
    "Spotify Popularity"
].mean()

explicit_comparison.index = ["Non-Explicit", "Explicit"]

print(explicit_comparison)

plt.figure(figsize=(8, 5))
explicit_comparison.plot(kind="bar")
plt.title("Average Spotify Popularity: Explicit vs Non-Explicit")
plt.ylabel("Average Popularity")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("question7_explicit.png", dpi=200)
plt.show()


# QUESTION 8: Playlist presence across platforms
print("\n8. PLAYLIST PRESENCE ACROSS PLATFORMS")

platforms = [
    "Spotify Playlist Count",
    "Apple Music Playlist Count",
    "Deezer Playlist Count",
    "Amazon Playlist Count"
]

platform_means = df[platforms].mean()
print(platform_means)

plt.figure(figsize=(10, 6))
platform_means.plot(kind="bar")
plt.title("Average Playlist Count Across Platforms")
plt.ylabel("Average Playlist Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("question8_platforms.png", dpi=200)
plt.show()


print("\n====================================")
print("ALL 8 EDA QUESTIONS COMPLETED!")
print("All graphs have been saved.")
print("====================================")
