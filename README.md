# 🎵 Spotify Streaming Analytics Dashboard

An interactive Exploratory Data Analysis (EDA) project built using Python and Streamlit to explore the most streamed Spotify songs of 2024.

## 🌐 Live Dashboard

[**Open Spotify Streaming Analytics Dashboard**](https://spotify-streaming-analytics.streamlit.app/)

## 📊 Project Overview

This project analyzes Spotify streaming data alongside music performance across platforms such as YouTube, TikTok, Apple Music, and Deezer.

The goal is to discover patterns in music popularity, streaming performance, social media engagement, and playlist presence.

## ✨ Features

- Interactive Streamlit dashboard
- Dynamic KPI cards
- Sidebar filters
- Spotify streaming analytics
- Artist and track analysis
- YouTube and TikTok engagement analysis
- Music platform comparisons
- Interactive Plotly visualizations
- Data explorer with CSV download
- Key findings and insights

## 📁 Dataset

**Dataset:** Most Streamed Spotify Songs 2024

**Original source:** [Kaggle Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024)

The dataset contains song information, streaming statistics, playlist counts, and social media engagement metrics.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Jupyter Notebook
- GitHub

## 🧹 Data Cleaning

- Removed duplicate records.
- Handled missing values.
- Converted numeric columns to appropriate data types.
- Converted release dates into datetime format.
- Detected outliers using the IQR method.
- Retained outliers where they could represent genuine popular songs.

## 📈 Key Findings

1. YouTube Views and Spotify Streams show a moderate positive correlation.
2. Track Score and Spotify Streams show a weak positive correlation.
3. Taylor Swift and Drake have the highest number of song entries in this dataset.
4. Explicit tracks have a slightly higher average Spotify Popularity score than non-explicit tracks.

## 🚀 Run Locally

Clone the repository:

```bash
git clone https://github.com/habibanadeemhere/spotify-streaming-analytics.git
```

Open the project folder:

```bash
cd spotify-streaming-analytics
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## 👩‍💻 Author

**Habiba Nadeem**

GitHub: [@habibanadeemhere](https://github.com/habibanadeemhere)

---

⭐ If you find this project interesting, feel free to explore the dashboard!