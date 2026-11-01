"""
Spotify Music Trends - Exploratory Data Analysis Dashboard
==========================================================
Run with:  streamlit run app.py

How this file is organised (useful for your viva):
  1. Page setup and constants
  2. Custom CSS (the premium dark look)
  3. Data loading           -> uses @st.cache_data
  4. Helper functions       -> number formatting, correlations, chart styling
  5. Sidebar filters        -> search, artist, explicit, year, popularity, reset
  6. Dashboard sections     -> Overview, Streaming, Social, Platforms, Explorer, Insights

Important idea about missing values
-----------------------------------
In handle_missing.py, missing numbers were filled with the column MEDIAN.
That keeps all rows, but it also creates many identical "estimated" values
(for example, about 72% of Soundcloud Streams). Comparing or correlating those
estimates would be misleading, so the charts below leave out rows whose value
equals the fill value. The helper `observed()` does this in one place.
"""

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# 1. PAGE SETUP AND CONSTANTS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Music Trends",
    page_icon=":material/equalizer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "spotify_final.csv"

GREEN = "#1ED760"
GREEN_DEEP = "#1DB954"
VIOLET = "#8B7CFF"
AMBER = "#F5B93A"
ROSE = "#FF6B81"
TEXT = "#E8ECF1"
MUTED = "#8B95A3"
GRID = "rgba(255,255,255,0.07)"

TYPE_COLORS = {"Clean": GREEN, "Explicit": AMBER}

# Playlist columns used on the Platform Comparison page
PLATFORMS = {
    "Spotify": ("Spotify Playlist Count", GREEN),
    "Apple Music": ("Apple Music Playlist Count", ROSE),
    "Deezer": ("Deezer Playlist Count", VIOLET),
    "Amazon": ("Amazon Playlist Count", AMBER),
}

# Columns that had missing values in the raw file (see cleaning.py).
# After handle_missing.py these contain many copies of the median.
IMPUTED_COLUMNS = [
    "Spotify Streams", "Spotify Playlist Count", "Spotify Playlist Reach",
    "Spotify Popularity", "YouTube Views", "YouTube Likes", "TikTok Posts",
    "TikTok Likes", "TikTok Views", "YouTube Playlist Reach",
    "Apple Music Playlist Count", "AirPlay Spins", "SiriusXM Spins",
    "Deezer Playlist Count", "Deezer Playlist Reach", "Amazon Playlist Count",
    "Pandora Streams", "Pandora Track Stations", "Soundcloud Streams",
    "Shazam Counts",
]

# ---------------------------------------------------------------------------
# 2. CUSTOM CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Bebas+Neue&display=swap');

:root {
  --bg: #0B0E12;
  --surface: rgba(255,255,255,0.04);
  --line: rgba(255,255,255,0.09);
  --text: #E8ECF1;
  --muted: #8B95A3;
  --green: #1ED760;
  --violet: #8B7CFF;
  --amber: #F5B93A;
}

.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp [data-baseweb="select"], .stApp [data-baseweb="tab"] {
  font-family: 'Sora', system-ui, sans-serif;
}
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

.stApp {
  background:
    radial-gradient(900px 520px at 6% -8%, rgba(30,215,96,0.17), transparent 60%),
    radial-gradient(760px 480px at 100% 0%, rgba(139,124,255,0.14), transparent 55%),
    var(--bg);
  background-attachment: fixed;
}

/* Remove default Streamlit chrome but keep the sidebar toggle */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1280px; padding: 2.2rem 2rem 4rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(16,21,27,0.96), rgba(10,13,17,0.98));
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
.brand-name { font-weight: 600; font-size: 1.05rem; color: var(--text); line-height: 1.2; }
.brand-sub { font-size: 0.74rem; color: var(--muted); }
.side-title { font-weight: 600; font-size: 0.85rem; color: var(--text); margin: 6px 0 10px; }
.match {
  margin: 14px 0 6px; padding: 12px 14px; border-radius: 14px;
  background: rgba(30,215,96,0.08); border: 1px solid rgba(30,215,96,0.25);
  font-size: 0.82rem; color: var(--muted);
}
.match b { color: var(--green); font-size: 1.05rem; }

/* Inputs */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
  border-radius: 12px !important;
  background: rgba(255,255,255,0.04) !important;
}
.stButton button, .stDownloadButton button {
  border-radius: 12px; border: 1px solid var(--line); background: var(--surface);
  transition: border-color .2s, color .2s, box-shadow .2s;
}
.stButton button:hover, .stDownloadButton button:hover {
  border-color: rgba(30,215,96,0.6); color: var(--green);
  box-shadow: 0 0 22px -8px rgba(30,215,96,0.6);
}
button[data-testid="stBaseButton-primary"] { color: #06210F; font-weight: 600; background: var(--green); }
button[data-testid="stBaseButton-primary"]:hover { color: #06210F; background: #3BE477; }

/* Tabs as a pill navigation bar (selectors for Streamlit 1.66) */
[data-testid="stTabs"] [role="tablist"] {
  gap: 4px; padding: 5px; width: fit-content; max-width: 100%; overflow-x: auto;
  background: var(--surface); border: 1px solid var(--line); border-radius: 999px; border-bottom: 1px solid var(--line);
}
[data-testid="stTab"] {
  height: 38px; padding: 0 18px; border-radius: 999px; background: transparent;
  color: var(--muted); font-weight: 500; font-size: 0.86rem; border: none; box-shadow: none;
  transition: background .2s ease, color .2s ease;
}
[data-testid="stTab"]:hover { color: var(--text); background: rgba(255,255,255,0.05); }
[data-testid="stTab"][aria-selected="true"] { background: rgba(30,215,96,0.15); color: var(--green); }
[data-testid="stTab"]::after, [data-testid="stTab"]::before { display: none !important; }
[data-testid="stTab"] > div + div { display: none !important; }
[data-testid="stTabs"] [role="tablist"]::after { display: none !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
[data-testid="stTabPanel"] { padding-top: 1.4rem; }

/* Hero */
.hero h1 {
  font-family: 'Bebas Neue', sans-serif; font-weight: 400;
  font-size: 5.4rem; line-height: 0.92; letter-spacing: 0.01em;
  margin: 0 0 14px; padding: 0; color: #fff;
}
.hero p { color: var(--muted); font-size: 1rem; line-height: 1.65; max-width: 46ch; margin: 0 0 18px; }

/* Section headers */
.sec { margin: 4px 0 18px; }
.sec h2 { font-size: 1.45rem; font-weight: 600; margin: 0; padding: 0; color: var(--text); }
.sec p { color: var(--muted); margin: 6px 0 0; font-size: 0.9rem; max-width: 72ch; line-height: 1.6; }
.sub { font-weight: 600; font-size: 1rem; margin: 26px 0 10px; color: var(--text); }
.sub::before {
  content: ""; display: inline-block; width: 4px; height: 16px; border-radius: 2px;
  background: var(--green); margin-right: 10px; vertical-align: -2px;
}
.note { color: var(--muted); font-size: 0.8rem; line-height: 1.6; margin: 6px 2px 14px; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 4px 0 22px; }
@media (min-width: 1100px) { .kpi-grid { grid-template-columns: repeat(6, 1fr); } }
.kpi {
  position: relative; overflow: hidden; padding: 16px 18px; border-radius: 22px;
  background: linear-gradient(160deg, rgba(255,255,255,0.085), rgba(255,255,255,0.025));
  border: 1px solid var(--line); backdrop-filter: blur(14px);
  transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.kpi::after {
  content: ""; position: absolute; right: -40px; bottom: -70px; width: 160px; height: 160px;
  background: radial-gradient(circle, rgba(30,215,96,0.28), transparent 70%);
  opacity: 0; transition: opacity .3s ease;
}
.kpi:hover { transform: translateY(-3px); border-color: rgba(30,215,96,0.45); box-shadow: 0 18px 46px -26px rgba(30,215,96,0.55); }
.kpi:hover::after { opacity: 1; }
.kpi-label { color: var(--muted); font-size: 0.78rem; font-weight: 500; }
.kpi-value { font-family: 'Bebas Neue', sans-serif; font-size: 2.6rem; line-height: 1.05; color: #fff; margin-top: 6px; letter-spacing: 0.02em; }
.kpi-note { color: var(--muted); font-size: 0.74rem; margin-top: 2px; }

/* Insight and finding cards */
.insight-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; margin: 4px 0 18px; }
.insight {
  padding: 18px 20px; border-radius: 18px; background: var(--surface);
  border: 1px solid var(--line); border-left: 3px solid var(--green);
  transition: border-color .25s ease, background .25s ease;
}
.insight:hover { background: rgba(255,255,255,0.065); border-color: rgba(30,215,96,0.4); border-left-color: var(--green); }
.insight.violet { border-left-color: var(--violet); }
.insight.amber { border-left-color: var(--amber); }
.insight h4 { margin: 0 0 8px; padding: 0; font-size: 0.95rem; font-weight: 600; color: var(--text); }
.insight p { margin: 0; color: var(--muted); font-size: 0.86rem; line-height: 1.65; }
.insight p b { color: var(--text); font-weight: 600; }

/* Chips */
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 12px; }
.chip {
  display: inline-flex; gap: 6px; align-items: baseline; padding: 6px 11px; border-radius: 999px;
  background: var(--surface); border: 1px solid var(--line); font-size: 0.74rem; color: var(--muted);
}
.chip b { color: var(--text); font-weight: 600; }
.chip.green { border-color: rgba(30,215,96,0.35); background: rgba(30,215,96,0.08); }

/* Chart and table cards */
[data-testid="stPlotlyChart"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.02));
  border: 1px solid var(--line); border-radius: 22px; padding: 0; overflow: hidden; backdrop-filter: blur(14px);
  transition: border-color .25s ease, box-shadow .25s ease;
}
[data-testid="stPlotlyChart"]:hover { border-color: rgba(30,215,96,0.35); box-shadow: 0 18px 50px -28px rgba(30,215,96,0.5); }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 16px; background: var(--surface); }

@media (max-width: 768px) {
  .hero h1 { font-size: 3.6rem; }
  .block-container { padding: 1.2rem 1rem 3rem; }
  .kpi-value { font-size: 2.4rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. DATA LOADING  (cached so the CSV is only read once)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str):
    """Read the cleaned CSV and add a few helper columns. The file itself is never changed."""
    df = pd.read_csv(path)
    for col in ["Track", "Artist", "Album Name"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
    df["Release Year"] = df["Release Date"].dt.year
    df["Track Type"] = np.where(df["Explicit Track"] == 1, "Explicit", "Clean")

    # The value used to fill gaps equals the column median (see handle_missing.py)
    fill_values = {c: float(df[c].median()) for c in IMPUTED_COLUMNS if c in df.columns}
    return df, fill_values


@st.cache_data
def to_csv_bytes(data: pd.DataFrame) -> bytes:
    """Convert a table to CSV bytes (utf-8-sig so Excel on Windows shows accents correctly)."""
    return data.to_csv(index=False).encode("utf-8-sig")


if not DATA_PATH.exists():
    st.error(
        f"Data file not found: {DATA_PATH}. "
        "Create a folder named data next to app.py and place spotify_final.csv inside it."
    )
    st.stop()

df, FILL = load_data(str(DATA_PATH))
EXTRA_COLUMNS = ["Release Year", "Track Type"]
ORIGINAL_COLUMNS = [c for c in df.columns if c not in EXTRA_COLUMNS]
MIN_YEAR, MAX_YEAR = int(df["Release Year"].min()), int(df["Release Year"].max())


# ---------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def fmt(n, decimals: int = 1) -> str:
    """Format large numbers: 1234567 -> 1.2M."""
    if n is None or pd.isna(n):
        return "n/a"
    n = float(n)
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= limit:
            text = f"{n / limit:.{decimals}f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return text + suffix
    return f"{n:,.0f}"


def esc(text) -> str:
    return html.escape(str(text))


def shorten(text: str, limit: int = 34) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def top_songs(data, col, n):
    """Top n songs by `col`. A track listed under several releases is shown once, with its highest value."""
    ranked = observed(data, col).dropna(subset=[col]).sort_values(col, ascending=False)
    return ranked.drop_duplicates(["Track", "Artist"]).head(n)


def song_labels(frame):
    """Bar labels like 'Track - Artist' (track names repeat across artists in this dataset)."""
    return unique_labels([shorten(f"{t} - {a}", 40) for t, a in zip(frame["Track"], frame["Artist"])])


def unique_labels(labels):
    """Plotly merges bars with identical labels, so add (2), (3) to repeats."""
    seen, result = {}, []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        result.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return result


def observed(data: pd.DataFrame, *cols) -> pd.DataFrame:
    """Keep only rows where every given column holds a real value (not the median fill)."""
    keep = pd.Series(True, index=data.index)
    for col in cols:
        if col in FILL:
            keep &= data[col] != FILL[col]
    return data[keep]


def correlations(data, x, y):
    """Pearson (raw numbers) and Spearman (ranks) correlation on real values only."""
    pair = observed(data, x, y)[[x, y]].dropna()
    if len(pair) < 10 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
        return None
    return {
        "pearson": pair[x].corr(pair[y]),
        "spearman": pair[x].corr(pair[y], method="spearman"),
        "n": len(pair),
    }


def strength(r: float) -> str:
    a = abs(r)
    if a < 0.2:
        return "very weak"
    if a < 0.4:
        return "weak"
    if a < 0.6:
        return "moderate"
    if a < 0.8:
        return "strong"
    return "very strong"


def log_axis(values) -> dict:
    """Log axis with readable tick labels (1K, 1M, 1B) instead of 1G."""
    values = values[values > 0]
    if values.empty:
        return {"type": "log"}
    lo = int(np.floor(np.log10(values.min())))
    hi = int(np.ceil(np.log10(values.max())))
    ticks = [10.0 ** k for k in range(lo, hi + 1)]
    return {"type": "log", "tickvals": ticks, "ticktext": [fmt(t) for t in ticks]}


# ----- HTML building blocks ------------------------------------------------
def section_header(title, description):
    st.markdown(
        f'<div class="sec"><h2>{esc(title)}</h2><p>{esc(description)}</p></div>',
        unsafe_allow_html=True,
    )


def sub(text):
    st.markdown(f'<div class="sub">{esc(text)}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{esc(text)}</div>', unsafe_allow_html=True)


def kpi_row(items):
    cards = "".join(
        f'<div class="kpi"><div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div><div class="kpi-note">{esc(hint)}</div></div>'
        for label, value, hint in items
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def insight_cards(items):
    """items: list of (title, body_html, colour_class). Body may contain <b> tags."""
    cards = "".join(
        f'<div class="insight {cls}"><h4>{esc(title)}</h4><p>{body}</p></div>'
        for title, body, cls in items
    )
    st.markdown(f'<div class="insight-grid">{cards}</div>', unsafe_allow_html=True)


def corr_chips(c, caption=""):
    if c is None:
        st.markdown('<div class="note">Not enough songs with recorded values to calculate a correlation.</div>', unsafe_allow_html=True)
        return
    direction = "positive" if c["spearman"] >= 0 else "negative"
    st.markdown(
        '<div class="chips">'
        f'<span class="chip">Pearson r <b>{c["pearson"]:.2f}</b></span>'
        f'<span class="chip">Spearman ρ <b>{c["spearman"]:.2f}</b></span>'
        f'<span class="chip">n <b>{c["n"]:,}</b></span>'
        f'<span class="chip green"><b>{strength(c["spearman"]).capitalize()} {direction}</b></span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ----- Plotly helpers --------------------------------------------------------
def style(fig, title, subtitle=None, height=420, legend=False):
    """One place that gives every chart the same dark, glass-friendly look."""
    heading = f"<b>{title}</b>"
    if subtitle:
        heading += f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=heading, x=0.01, xanchor="left", y=0.97, yanchor="top", pad=dict(t=12, l=8), font=dict(size=16, color=TEXT)),
        height=height,
        margin=dict(l=24, r=28, t=84 if subtitle else 60, b=56),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Sora, sans-serif", color=TEXT, size=12),
        showlegend=legend, legend_title_text="",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom", bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="#161B22", bordercolor="rgba(255,255,255,0.18)", font=dict(family="Sora, sans-serif", color=TEXT, size=12)),
        colorway=[GREEN, VIOLET, AMBER, ROSE],
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), title_font=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED), title_font=dict(color=MUTED), automargin=True)
    return fig


def show(fig, empty_message="Not enough data for this chart with the current filters."):
    if fig is None:
        st.info(empty_message)
    else:
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def hbar_fig(labels, values, hover, title, subtitle=None, color=GREEN, text=None, height=None):
    """Horizontal bar chart, biggest value on top, every bar labelled, axis starts at zero."""
    labels, values, hover = list(labels)[::-1], list(values)[::-1], list(hover)[::-1]
    text = list(text)[::-1] if text is not None else [fmt(v) for v in values]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            text=text, textposition="outside", cliponaxis=False,
            textfont=dict(color=TEXT, size=11),
            customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, max(values) * 1.2], showticklabels=False)
    fig.update_yaxes(showgrid=False)
    return style(fig, title, subtitle, height=height or 130 + 34 * len(labels))


def scatter_fig(data, x, y, title, subtitle, x_log=False, height=470):
    """Scatter of two measures on real values only, coloured by explicit/clean."""
    d = observed(data, x, y).dropna(subset=[x, y])
    d = d[(d[y] > 0) & (d[x] > 0 if x_log else True)]
    if len(d) < 5:
        return None
    x_format = ":.1f" if x == "Track Score" else ":,.0f"
    fig = px.scatter(
        d, x=x, y=y, color="Track Type",
        color_discrete_map=TYPE_COLORS, category_orders={"Track Type": ["Clean", "Explicit"]},
        hover_name="Track", hover_data={"Artist": True, "Track Type": False, x: x_format, y: ":,.0f"},
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    fig.update_yaxes(**log_axis(d[y]))
    if x == "Track Score":
        fig.update_xaxes(type="log")
    elif x_log:
        fig.update_xaxes(**log_axis(d[x]))
    return style(fig, title, subtitle, height=height, legend=True)


# ---------------------------------------------------------------------------
# 5. SIDEBAR FILTERS
# ---------------------------------------------------------------------------
DEFAULT_FILTERS = {
    "f_search": "",
    "f_artists": [],
    "f_type": "All songs",
    "f_years": (MIN_YEAR, MAX_YEAR),
    "f_pop": (0, 100),
}
for key, value in DEFAULT_FILTERS.items():
    st.session_state.setdefault(key, value)


def reset_filters():
    """Runs before the page reruns, so the widgets pick up the default values."""
    for key, value in DEFAULT_FILTERS.items():
        st.session_state[key] = value


ARTISTS_BY_SONGS = df["Artist"].value_counts().index.tolist()

with st.sidebar:
    st.markdown(
        '<div class="brand"><svg width="32" height="32" viewBox="0 0 30 30" aria-hidden="true">'
        '<rect x="3" y="14" width="4" height="12" rx="2" fill="#1ED760"/>'
        '<rect x="10" y="6" width="4" height="20" rx="2" fill="#1ED760"/>'
        '<rect x="17" y="10" width="4" height="16" rx="2" fill="#1ED760" opacity=".7"/>'
        '<rect x="24" y="3" width="4" height="23" rx="2" fill="#1ED760" opacity=".45"/></svg>'
        '<div><div class="brand-name">Music Trends</div><div class="brand-sub">Spotify EDA dashboard</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-title">Filters</div>', unsafe_allow_html=True)
    st.text_input("Search song, artist or album", key="f_search", placeholder="Type a name")
    st.multiselect("Artists", options=ARTISTS_BY_SONGS, key="f_artists", placeholder="All artists")
    st.radio("Track type", ["All songs", "Explicit only", "Clean only"], key="f_type")
    st.slider("Release year", MIN_YEAR, MAX_YEAR, key="f_years")
    st.slider("Spotify popularity", 0, 100, key="f_pop")
    st.button("Reset filters", on_click=reset_filters, width="stretch")


def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    """Apply every sidebar control. Each filter only acts when it differs from its default."""
    out = data
    query = st.session_state.f_search.strip()
    if query:
        out = out[
            out["Track"].str.contains(query, case=False, regex=False)
            | out["Artist"].str.contains(query, case=False, regex=False)
            | out["Album Name"].str.contains(query, case=False, regex=False)
        ]
    if st.session_state.f_artists:
        out = out[out["Artist"].isin(st.session_state.f_artists)]
    if st.session_state.f_type == "Explicit only":
        out = out[out["Explicit Track"] == 1]
    elif st.session_state.f_type == "Clean only":
        out = out[out["Explicit Track"] == 0]
    if tuple(st.session_state.f_years) != (MIN_YEAR, MAX_YEAR):
        low, high = st.session_state.f_years
        out = out[out["Release Year"].between(low, high)]
    if tuple(st.session_state.f_pop) != (0, 100):
        low, high = st.session_state.f_pop
        out = out[out["Spotify Popularity"].between(low, high)]
    return out


filtered = apply_filters(df)

with st.sidebar:
    st.markdown(
        f'<div class="match"><b>{len(filtered):,}</b> of {len(df):,} songs match your filters</div>',
        unsafe_allow_html=True,
    )

if filtered.empty:
    st.markdown(
        '<div class="sec"><h2>No songs match these filters</h2>'
        "<p>Widen the release year or popularity range, clear the search box, or select Reset filters in the sidebar.</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# 6. DASHBOARD
# ---------------------------------------------------------------------------
tab_overview, tab_stream, tab_social, tab_platform, tab_explorer, tab_insights = st.tabs(
    ["Overview", "Streaming Analytics", "Social Media", "Platform Comparison", "Data Explorer", "Insights"]
)

# ============================= A. OVERVIEW =================================
with tab_overview:
    left, right = st.columns([1.1, 1], gap="large", vertical_alignment="center")

    with left:
        st.markdown(
            '<div class="hero"><h1>Spotify Music<br>Trends</h1>'
            f"<p>An exploratory look at {len(df):,} popular tracks: how they stream, how far they travel "
            "on social media, and how often they land in playlists across four platforms.</p></div>"
            '<div class="chips">'
            f'<span class="chip green">In view <b>{len(filtered):,} songs</b></span>'
            f'<span class="chip">Source <b>Most Streamed Spotify Songs 2024 (Kaggle)</b></span>'
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        years = filtered["Release Year"].dropna().astype(int).value_counts().sort_index()
        years = years.reindex(range(years.index.min(), years.index.max() + 1), fill_value=0)
        hero = go.Figure(
            go.Bar(
                x=years.index, y=years.values,
                marker=dict(
                    color=years.values,
                    colorscale=[[0, "rgba(30,215,96,0.28)"], [1, "rgba(30,215,96,1)"]],
                    line=dict(width=0),
                ),
                hovertemplate="%{x}: %{y} songs<extra></extra>",
            )
        )
        hero.update_layout(bargap=0.22)
        hero.update_yaxes(visible=False)
        hero.update_xaxes(showgrid=False, tickmode="auto", nticks=7)
        show(style(hero, "Songs by release year", "Each bar is one year of releases in the current view", height=300))

    # KPI cards: they recalculate from `filtered`, so they react to every sidebar control
    total_streams = filtered["Spotify Streams"].sum()
    kpi_row([
        ("Total songs", f"{len(filtered):,}", f"of {len(df):,} in the dataset"),
        ("Total artists", f"{filtered['Artist'].nunique():,}", "unique names in view"),
        ("Avg Spotify popularity", f"{filtered['Spotify Popularity'].mean():.1f}", "score from 0 to 100"),
        ("Total Spotify streams", fmt(total_streams, 2), "summed across songs in view"),
        ("Avg track score", f"{filtered['Track Score'].mean():.1f}", "cross-platform score"),
        ("Explicit songs", f"{filtered['Explicit Track'].mean() * 100:.0f}%", "share of songs in view"),
    ])

    # Key insights computed from the filtered data (nothing is typed in by hand)
    sub("Key insights")
    streams_ok = observed(filtered, "Spotify Streams")
    cards = []
    if not streams_ok.empty:
        top_row = streams_ok.loc[streams_ok["Spotify Streams"].idxmax()]
        cards.append((
            "Most streamed song",
            f"<b>{esc(top_row['Track'])}</b> by {esc(top_row['Artist'])} leads with <b>{fmt(top_row['Spotify Streams'])}</b> Spotify streams.",
            "",
        ))
        by_artist = streams_ok.groupby("Artist")["Spotify Streams"].sum().sort_values(ascending=False)
        share = by_artist.iloc[0] / by_artist.sum() * 100
        cards.append((
            "Artist with most streams",
            f"<b>{esc(by_artist.index[0])}</b> has <b>{fmt(by_artist.iloc[0])}</b> streams across their songs here, {share:.1f}% of all streams in view.",
            "violet",
        ))
    artist_counts = filtered["Artist"].value_counts()
    cards.append((
        "Artist with most songs",
        f"<b>{esc(artist_counts.index[0])}</b> appears <b>{artist_counts.iloc[0]}</b> times in the current view.",
        "amber",
    ))
    peak_year = int(filtered["Release Year"].value_counts().idxmax())
    peak_count = int(filtered["Release Year"].value_counts().max())
    cards.append((
        "Busiest release year",
        f"<b>{peak_year}</b> has the most songs in view (<b>{peak_count:,}</b>).",
        "",
    ))
    insight_cards(cards)

    with st.expander("About this data and how missing values were handled"):
        st.markdown(
            "The dataset has one row per song. Before this dashboard, `cleaning.py` fixed data types and removed duplicates, "
            "and `handle_missing.py` filled missing numbers with the **median** of each column. "
            "The chart shows how much of each column now holds that fill value. Charts that compare platforms or calculate "
            "correlations leave those rows out, because identical estimated values would distort the result."
        )
        share_filled = pd.Series({c: (df[c] == FILL[c]).mean() * 100 for c in FILL}).sort_values(ascending=False)
        show(hbar_fig(
            share_filled.index, share_filled.values,
            [f"{c}<br>{v:.1f}% of values equal the fill value" for c, v in share_filled.items()],
            "Share of each column that equals the median fill value",
            "Approximate: a few real values can match the median by chance",
            color=VIOLET, text=[f"{v:.0f}%" for v in share_filled.values],
        ))
        note("Spotify Popularity is the one exception: about 17% of its values were filled with 67, but 67 is also a common real score, so it is not removed from the popularity charts.")

# ========================= B. STREAMING ANALYTICS ==========================
with tab_stream:
    section_header(
        "Streaming Analytics",
        "Univariate charts look at one variable at a time. Bivariate charts compare two variables. "
        "Categorical charts compare groups.",
    )
    top_n = st.slider("Number of songs and artists to show", 5, 20, 10, key="top_n")

    sub("Univariate analysis")
    c1, c2 = st.columns(2)
    with c1:
        top = top_songs(filtered, "Spotify Streams", top_n)
        labels = song_labels(top)
        hover = [f"<b>{esc(t)}</b><br>{esc(a)}<br>{s:,.0f} streams" for t, a, s in zip(top["Track"], top["Artist"], top["Spotify Streams"])]
        show(hbar_fig(labels, top["Spotify Streams"], hover, f"Top {len(top)} most-streamed songs", "Spotify streams, hover for the artist") if not top.empty else None)
    with c2:
        by_artist = (
            filtered.groupby("Artist")
            .agg(songs=("Track", "size"), streams=("Spotify Streams", "sum"))
            .sort_values(["songs", "streams"], ascending=False)
            .head(top_n)
        )
        hover = [f"<b>{esc(a)}</b><br>{r.songs} songs<br>{fmt(r.streams)} total streams" for a, r in by_artist.iterrows()]
        show(hbar_fig(unique_labels([shorten(a) for a in by_artist.index]), by_artist["songs"], hover,
                      f"Top {len(by_artist)} artists by number of songs", "Songs per artist in the current view",
                      color=VIOLET, text=[str(v) for v in by_artist["songs"]]))

    note("Track names repeat across artists and releases in this dataset, so bars are labelled Track - Artist. "
         "If one artist has the same track listed twice, it is shown once with its highest value. Artist song counts follow the rows in the dataset.")

    c3, c4 = st.columns(2)
    with c3:
        pop = filtered["Spotify Popularity"].dropna()
        fill_score = int(FILL["Spotify Popularity"])
        counts = pop.round().astype(int).value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker=dict(color=[AMBER if v == fill_score else GREEN_DEEP for v in counts.index], line=dict(width=0)),
            hovertemplate="Popularity %{x}<br>%{y} songs<extra></extra>",
        ))
        fig.update_xaxes(title="Spotify popularity (0 to 100)", range=[0, 101])
        fig.update_yaxes(title="Number of songs")
        show(style(fig, "Distribution of Spotify popularity", f"Average {pop.mean():.1f}, median {pop.median():.0f}. The amber bar ({fill_score}) includes median-filled values", height=420))
        note(f"Each bar is one popularity score. The amber bar is taller than its neighbours because missing popularity values were filled with the median ({fill_score}) during cleaning.")
    with c4:
        s = observed(filtered, "Spotify Streams")["Spotify Streams"]
        s = s[s > 0]
        if len(s) >= 5:
            logs = np.log10(s)
            lo, hi = int(np.floor(logs.min())), int(np.ceil(logs.max()))
            fig = go.Figure(go.Histogram(
                x=logs, nbinsx=30, marker=dict(color=VIOLET, line=dict(width=0)),
                hovertemplate="%{y} songs in this range<extra></extra>",
            ))
            fig.update_xaxes(tickvals=list(range(lo, hi + 1)), ticktext=[fmt(10 ** k) for k in range(lo, hi + 1)], title="Spotify streams (log scale)")
            fig.update_yaxes(title="Number of songs")
            show(style(fig, "Distribution of Spotify streams", "Log scale: each step to the right is 10 times more streams", height=420))
            note(f"Streams are heavily skewed: the median song has {fmt(s.median())} streams while the average is {fmt(s.mean())}.")
        else:
            show(None)

    sub("Bivariate analysis")
    c5, c6 = st.columns(2)
    with c5:
        corr_chips(correlations(filtered, "Track Score", "Spotify Streams"))
        show(scatter_fig(filtered, "Track Score", "Spotify Streams", "Track Score vs Spotify Streams",
                         "Both axes use a log scale"))
    with c6:
        corr_chips(correlations(filtered, "YouTube Views", "Spotify Streams"))
        show(scatter_fig(filtered, "YouTube Views", "Spotify Streams", "YouTube Views vs Spotify Streams",
                         "Both axes use a log scale", x_log=True))
    note("Pearson r measures a straight-line relationship using the raw numbers, so a few giant songs can dominate it. "
         "Spearman ρ compares ranks and is safer for skewed data like streams. Songs with estimated (median-filled) values are excluded. "
         "A correlation shows that two measures move together, not that one causes the other.")

    sub("Categorical comparison: explicit and clean songs")
    groups = filtered.groupby("Track Type").agg(songs=("Track", "size"), avg_pop=("Spotify Popularity", "mean"))
    med_streams = observed(filtered, "Spotify Streams").groupby("Track Type")["Spotify Streams"].median()
    c7, c8, c9 = st.columns(3)
    with c7:
        fig = go.Figure(go.Pie(
            labels=groups.index, values=groups["songs"], hole=0.62,
            marker=dict(colors=[TYPE_COLORS[g] for g in groups.index], line=dict(color="#0B0E12", width=3)),
            textinfo="percent", textfont=dict(color="#0B0E12", size=13), hovertemplate="%{label}: %{value:,} songs<extra></extra>",
        ))
        show(style(fig, "Share of songs", "Explicit and clean tracks", height=360, legend=True))
    with c8:
        fig = go.Figure(go.Bar(
            x=groups.index, y=groups["avg_pop"], marker=dict(color=[TYPE_COLORS[g] for g in groups.index]),
            text=[f"{v:.1f}" for v in groups["avg_pop"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}: %{y:.1f}<extra></extra>",
        ))
        fig.update_yaxes(range=[0, 100], title="Average popularity")
        show(style(fig, "Average popularity", "Scale starts at zero", height=360))
    with c9:
        if not med_streams.empty:
            fig = go.Figure(go.Bar(
                x=med_streams.index, y=med_streams.values, marker=dict(color=[TYPE_COLORS[g] for g in med_streams.index]),
                text=[fmt(v) for v in med_streams.values], textposition="outside", cliponaxis=False,
                hovertemplate="%{x}: %{y:,.0f} streams<extra></extra>",
            ))
            fig.update_yaxes(range=[0, med_streams.max() * 1.2], title="Median streams")
            show(style(fig, "Median Spotify streams", "Median is less affected by hit songs", height=360))
        else:
            show(None)

# ========================= C. SOCIAL MEDIA ANALYTICS =======================
with tab_social:
    section_header(
        "Social Media Analytics",
        "TikTok and YouTube attention compared with streaming. Songs with estimated values are left out of every chart on this page.",
    )

    sub("TikTok views")
    c1, c2 = st.columns(2)
    with c1:
        tt_top = top_songs(filtered, "TikTok Views", top_n)
        hover = [f"<b>{esc(t)}</b><br>{esc(a)}<br>{v:,.0f} TikTok views" for t, a, v in zip(tt_top["Track"], tt_top["Artist"], tt_top["TikTok Views"])]
        show(hbar_fig(song_labels(tt_top), tt_top["TikTok Views"], hover,
                      f"Top {len(tt_top)} songs by TikTok views", "Total views of TikTok videos using the song", color=ROSE) if not tt_top.empty else None)
    with c2:
        corr_chips(correlations(filtered, "TikTok Views", "Spotify Streams"))
        show(scatter_fig(filtered, "TikTok Views", "Spotify Streams", "TikTok Views vs Spotify Streams",
                         "Both axes use a log scale", x_log=True, height=420))

    sub("YouTube engagement")
    yt = observed(filtered, "YouTube Views", "YouTube Likes")
    yt = yt[yt["YouTube Views"] >= 1_000_000].copy()
    yt["Like rate"] = yt["YouTube Likes"] / yt["YouTube Views"] * 100
    c3, c4 = st.columns(2)
    with c3:
        if len(yt) >= 5:
            fig = go.Figure(go.Histogram(x=yt["Like rate"], nbinsx=30, marker=dict(color=GREEN_DEEP, line=dict(width=0)),
                                         hovertemplate="Like rate %{x:.1f}%<br>%{y} songs<extra></extra>"))
            fig.add_vline(x=yt["Like rate"].median(), line_dash="dot", line_color=AMBER,
                          annotation_text=f"Median {yt['Like rate'].median():.2f}%", annotation_font_color=AMBER)
            fig.update_xaxes(title="Likes per 100 views (%)")
            fig.update_yaxes(title="Number of songs")
            show(style(fig, "How often YouTube viewers press like", "Songs with at least 1 million views", height=440))
        else:
            show(None)
    with c4:
        best = yt.sort_values("Like rate", ascending=False).drop_duplicates(["Track", "Artist"]).head(top_n)
        hover = [f"<b>{esc(t)}</b><br>{esc(a)}<br>{r:.2f} likes per 100 views" for t, a, r in zip(best["Track"], best["Artist"], best["Like rate"])]
        show(hbar_fig(song_labels(best), best["Like rate"], hover,
                      f"Top {len(best)} songs by YouTube like rate", "Likes per 100 views, minimum 1 million views",
                      color=AMBER, text=[f"{v:.1f}%" for v in best["Like rate"]]) if not best.empty else None)

    sub("Comparing social media engagement")
    c5, c6 = st.columns(2)
    with c5:
        reach_cols = {
            "Spotify streams": "Spotify Streams", "YouTube views": "YouTube Views", "TikTok views": "TikTok Views",
            "Pandora streams": "Pandora Streams", "Soundcloud streams": "Soundcloud Streams",
        }
        rows = []
        for name, col in reach_cols.items():
            values = observed(filtered, col)[col].dropna()
            if len(values) >= 5:
                rows.append((name, values.median(), len(values)))
        rows.sort(key=lambda r: r[1], reverse=True)
        if rows:
            show(hbar_fig(
                [r[0] for r in rows], [r[1] for r in rows],
                [f"<b>{r[0]}</b><br>Median {r[1]:,.0f}<br>Based on {r[2]:,} songs with a recorded value" for r in rows],
                "Typical song reach by platform", "Median per song. Streams and views are counted differently",
                color=VIOLET, height=420,
            ))
        else:
            show(None)
    with c6:
        tk = observed(filtered, "TikTok Views", "TikTok Likes")
        tk = tk[tk["TikTok Views"] >= 1_000_000]
        fig = go.Figure()
        for name, data, color in (("YouTube", yt["Like rate"] if not yt.empty else pd.Series(dtype=float), GREEN),
                                  ("TikTok", tk["TikTok Likes"] / tk["TikTok Views"] * 100 if not tk.empty else pd.Series(dtype=float), ROSE)):
            if len(data) >= 5:
                fig.add_trace(go.Box(y=data, name=name, marker_color=color, line=dict(color=color), boxpoints=False,
                                     hovertemplate="%{y:.2f} likes per 100 views<extra></extra>"))
        if fig.data:
            fig.update_yaxes(type="log", title="Likes per 100 views (log scale)")
            show(style(fig, "Like rate: YouTube vs TikTok", "Songs with at least 1 million views. Log scale: each gridline is 10 times larger", height=420))
        else:
            show(None)
    n_over = int((tk["TikTok Likes"] > tk["TikTok Views"]).sum())
    note("TikTok views count plays of videos that use the song, while YouTube views count plays of the music video, "
         "so compare the shape and spread of engagement rather than treating the two platforms as identical measures. "
         + (f" {n_over} TikTok row(s) show more likes than views; they come from the source data and are kept as they are." if n_over else ""))

# ======================= D. PLATFORM COMPARISON ============================
with tab_platform:
    section_header(
        "Platform Comparison",
        "How many playlists each streaming service holds for the songs in view. "
        "Services differ in size, so compare patterns and rankings rather than raw counts.",
    )
    ctrl1, ctrl2, _ = st.columns([1, 1, 2])
    with ctrl1:
        stat_choice = st.radio("Statistic", ["Median", "Average"], horizontal=True)
    with ctrl2:
        use_log = st.checkbox("Log scale", value=True, help="Spotify has far more playlists than the others, so a log scale keeps every bar visible.")

    stats_rows, box_data = [], {}
    for name, (col, color) in PLATFORMS.items():
        values = observed(filtered, col)[col].dropna()
        if len(values) >= 5:
            box_data[name] = values
            stats_rows.append((name, values.median() if stat_choice == "Median" else values.mean(), len(values), color))

    sub("Playlist count by platform")
    c1, c2 = st.columns(2)
    with c1:
        if stats_rows:
            fig = go.Figure(go.Bar(
                x=[r[0] for r in stats_rows], y=[r[1] for r in stats_rows],
                marker=dict(color=[r[3] for r in stats_rows]),
                text=[fmt(r[1]) for r in stats_rows], textposition="outside", cliponaxis=False,
                customdata=[[r[2]] for r in stats_rows],
                hovertemplate="<b>%{x}</b><br>" + stat_choice + " %{y:,.0f} playlists<br>Based on %{customdata[0]:,} songs<extra></extra>",
            ))
            if use_log:
                fig.update_yaxes(**log_axis(pd.Series([r[1] for r in stats_rows])), title=f"{stat_choice} playlists per song (log scale)")
            else:
                fig.update_yaxes(title=f"{stat_choice} playlists per song", rangemode="tozero")
            show(style(fig, f"{stat_choice} playlist count per song", "Log scale: each gridline is 10 times larger" if use_log else "Linear scale", height=430))
        else:
            show(None)
    with c2:
        if box_data:
            fig = go.Figure()
            for name, values in box_data.items():
                color = PLATFORMS[name][1]
                fig.add_trace(go.Box(y=values, name=name, marker_color=color, line=dict(color=color), boxpoints=False, boxmean=True))
            fig.update_yaxes(**log_axis(pd.concat(box_data.values())), title="Playlists per song (log scale)")
            show(style(fig, "Spread of playlist counts", "Box shows the middle 50% of songs, dashed line is the average", height=430))
        else:
            show(None)

    sub("Compare two platforms")
    names = list(PLATFORMS.keys())
    sel1, sel2, _ = st.columns([1, 1, 2])
    with sel1:
        x_name = st.selectbox("Horizontal axis", names, index=0)
    with sel2:
        y_name = st.selectbox("Vertical axis", names, index=1)
    c3, c4 = st.columns(2)
    with c3:
        if x_name == y_name:
            st.info("Choose two different platforms to compare.")
        else:
            xc, yc = PLATFORMS[x_name][0], PLATFORMS[y_name][0]
            corr_chips(correlations(filtered, xc, yc))
            fig = scatter_fig(filtered, xc, yc, f"{x_name} vs {y_name} playlists", "Both axes use a log scale", x_log=True, height=440)
            if fig is not None:
                fig.update_xaxes(title=f"{x_name} playlists")
                fig.update_yaxes(title=f"{y_name} playlists")
            show(fig)
    with c4:
        heat_cols = {n: c for n, (c, _) in PLATFORMS.items()}
        heat_cols["Spotify streams"] = "Spotify Streams"
        mat = filtered[list(heat_cols.values())].copy()
        for col in mat.columns:
            if col in FILL:
                mat.loc[mat[col] == FILL[col], col] = np.nan
        if len(filtered) >= 10:
            corr = mat.corr(method="spearman")
            corr.index = corr.columns = list(heat_cols.keys())
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                colorscale=[[0, VIOLET], [0.5, "#161C24"], [1, "#17A34A"]], showscale=False,
                texttemplate="%{z:.2f}", textfont=dict(color="#FFFFFF"), hovertemplate="%{y} and %{x}<br>Spearman ρ %{z:.2f}<extra></extra>"
            ))
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(showgrid=False)
            show(style(fig, "How closely platforms move together", "Spearman rank correlation, real values only", height=470))
        else:
            show(None)

    sub("Top songs on a platform")
    pick = st.selectbox("Platform", names, key="top_platform")
    pcol = PLATFORMS[pick][0]
    ptop = top_songs(filtered, pcol, top_n)
    hover = [f"<b>{esc(t)}</b><br>{esc(a)}<br>{v:,.0f} {pick} playlists" for t, a, v in zip(ptop["Track"], ptop["Artist"], ptop[pcol])]
    show(hbar_fig(song_labels(ptop), ptop[pcol], hover,
                  f"Top {len(ptop)} songs by {pick} playlist count", "Number of playlists featuring the song",
                  color=PLATFORMS[pick][1]) if not ptop.empty else None)

# ========================== E. DATA EXPLORER ===============================
with tab_explorer:
    section_header(
        "Data Explorer",
        "Browse and sort the songs that match your sidebar filters. Use the search box in the sidebar to find a song, artist or album.",
    )
    sort_options = ["Spotify Streams", "Spotify Popularity", "Track Score", "TikTok Views", "YouTube Views",
                    "Spotify Playlist Count", "Release Date"]
    e1, e2, e3 = st.columns([1.3, 1, 1.2])
    with e1:
        sort_col = st.selectbox("Sort by", sort_options)
    with e2:
        order = st.radio("Order", ["Highest first", "Lowest first"], horizontal=True)
    with e3:
        show_raw = st.checkbox("Show raw data (all 28 columns, unformatted)")

    table = filtered.sort_values(sort_col, ascending=(order == "Lowest first"))

    if show_raw:
        st.dataframe(table[ORIGINAL_COLUMNS], hide_index=True, height=520)
    else:
        view_cols = ["Track", "Artist", "Album Name", "Release Date", "Track Type", "Spotify Streams",
                     "Spotify Popularity", "Track Score", "YouTube Views", "TikTok Views", "Spotify Playlist Count"]
        st.dataframe(
            table[view_cols], hide_index=True, height=520,
            column_config={
                "Release Date": st.column_config.DateColumn("Release date", format="YYYY-MM-DD"),
                "Track Type": st.column_config.TextColumn("Type"),
                "Spotify Streams": st.column_config.NumberColumn("Spotify streams", format="compact"),
                "Spotify Popularity": st.column_config.ProgressColumn("Popularity", min_value=0, max_value=100, format="%d"),
                "Track Score": st.column_config.NumberColumn("Track score", format="%.1f"),
                "YouTube Views": st.column_config.NumberColumn("YouTube views", format="compact"),
                "TikTok Views": st.column_config.NumberColumn("TikTok views", format="compact"),
                "Spotify Playlist Count": st.column_config.NumberColumn("Spotify playlists", format="compact"),
            },
        )

    d1, d2 = st.columns([1, 2.4], vertical_alignment="center")
    with d1:
        st.download_button(
            "Download filtered data (CSV)", data=to_csv_bytes(filtered[ORIGINAL_COLUMNS]),
            file_name="spotify_filtered.csv", mime="text/csv", type="primary", width="stretch",
        )
    with d2:
        note(f"{len(filtered):,} songs in the file. The download keeps all original columns and follows your sidebar filters.")

# ===================== F. INSIGHTS AND RECOMMENDATIONS =====================
with tab_insights:
    section_header(
        "Insights and Recommendations",
        "Every number on this page is calculated from the songs currently in view, so the text updates when you change the filters.",
    )

    # Rank correlation of many measures with Spotify streams
    candidates = {
        "Spotify playlist count": "Spotify Playlist Count", "Spotify playlist reach": "Spotify Playlist Reach",
        "YouTube views": "YouTube Views", "YouTube likes": "YouTube Likes",
        "TikTok views": "TikTok Views", "TikTok likes": "TikTok Likes", "TikTok posts": "TikTok Posts",
        "Apple Music playlists": "Apple Music Playlist Count", "Deezer playlists": "Deezer Playlist Count",
        "Amazon playlists": "Amazon Playlist Count", "Pandora streams": "Pandora Streams",
        "Shazam counts": "Shazam Counts", "AirPlay spins": "AirPlay Spins", "Track score": "Track Score",
    }
    rows = []
    for name, col in candidates.items():
        c = correlations(filtered, col, "Spotify Streams")
        if c is not None:
            rows.append({"Measure": name, "Pearson r": c["pearson"], "Spearman ρ": c["spearman"], "Songs": c["n"]})
    corr_table = pd.DataFrame(rows).sort_values("Spearman ρ", ascending=False) if rows else pd.DataFrame()

    sub("Three important findings")
    findings = []
    if not corr_table.empty:
        best = corr_table.iloc[0]
        findings.append((
            "Playlist presence tracks streams most closely" if "playlist" in best["Measure"].lower() else f"{best['Measure'].capitalize()} tracks streams most closely",
            f"Of {len(corr_table)} measures tested, <b>{esc(best['Measure'])}</b> has the strongest rank correlation with Spotify streams "
            f"(<b>ρ = {best['Spearman ρ']:.2f}</b>, {strength(best['Spearman ρ'])}, {int(best['Songs']):,} songs).",
            "",
        ))
    tiktok = correlations(filtered, "TikTok Views", "Spotify Streams")
    if tiktok is not None:
        gap = tiktok["spearman"] - tiktok["pearson"]
        if gap > 0.1:
            reason = ("The rank-based value is higher, which most likely means a small group of viral songs with extreme TikTok "
                      "views pulls the Pearson value down while the typical song shows a modest link.")
        else:
            reason = "Both measures agree, so extreme values are not driving this result."
        findings.append((
            "TikTok attention is only loosely tied to streams",
            f"TikTok views against Spotify streams give <b>Pearson r = {tiktok['pearson']:.2f}</b> and "
            f"<b>Spearman ρ = {tiktok['spearman']:.2f}</b> ({strength(tiktok['spearman'])}). {reason}",
            "violet",
        ))
    s_sorted = observed(filtered, "Spotify Streams")["Spotify Streams"].sort_values(ascending=False)
    if len(s_sorted) >= 10:
        k = max(1, int(round(len(s_sorted) * 0.1)))
        share = s_sorted.head(k).sum() / s_sorted.sum() * 100
        findings.append((
            "Streams are concentrated in a few hits",
            f"The top 10% of songs (<b>{k:,}</b> tracks) hold <b>{share:.0f}%</b> of all streams in view. "
            f"The median song has <b>{fmt(s_sorted.median())}</b> streams, while the average is <b>{fmt(s_sorted.mean())}</b>.",
            "amber",
        ))
    if findings:
        insight_cards(findings)
    else:
        st.info("Not enough songs in the current view to calculate findings. Widen your filters.")

    sub("What moves with Spotify streams?")
    if not corr_table.empty:
        ranked = corr_table.sort_values("Spearman ρ", ascending=False)
        colors = [GREEN if v >= 0.4 else (VIOLET if v >= 0.2 else "#4A5563") for v in ranked["Spearman ρ"]]
        fig = go.Figure(go.Bar(
            x=ranked["Spearman ρ"][::-1], y=ranked["Measure"][::-1], orientation="h",
            marker=dict(color=colors[::-1]), text=[f"{v:.2f}" for v in ranked["Spearman ρ"][::-1]],
            textposition="outside", cliponaxis=False,
            customdata=np.stack([ranked["Pearson r"][::-1], ranked["Songs"][::-1]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Spearman ρ %{x:.2f}<br>Pearson r %{customdata[0]:.2f}<br>%{customdata[1]:,} songs<extra></extra>",
        ))
        fig.update_xaxes(range=[min(0, ranked["Spearman ρ"].min() * 1.2), max(1, ranked["Spearman ρ"].max() * 1.2)], title="Spearman rank correlation with Spotify streams")
        fig.update_yaxes(showgrid=False)
        show(style(fig, "Which measures are linked to Spotify streams", "Green is 0.4 or higher, violet is 0.2 to 0.4", height=140 + 30 * len(ranked)))

        with st.expander("See the full correlation table"):
            table_out = corr_table.copy()
            table_out["Strength (by ρ)"] = table_out["Spearman ρ"].map(strength)
            st.dataframe(
                table_out, hide_index=True,
                column_config={
                    "Pearson r": st.column_config.NumberColumn(format="%.2f"),
                    "Spearman ρ": st.column_config.NumberColumn(format="%.2f"),
                    "Songs": st.column_config.NumberColumn(format="localized"),
                },
            )

    sub("How to read the correlation numbers")
    insight_cards([
        ("Pearson r and Spearman ρ",
         "Both range from -1 to +1. Pearson uses the raw numbers and is sensitive to extreme values. "
         "Spearman uses ranks, so it describes whether bigger values tend to go with bigger values even when the data is skewed.", ""),
        ("Size of the relationship",
         "This dashboard calls 0 to 0.2 very weak, 0.2 to 0.4 weak, 0.4 to 0.6 moderate, 0.6 to 0.8 strong and above 0.8 very strong. "
         "These labels are rules of thumb, not strict limits.", "violet"),
        ("Correlation is not causation",
         "A strong link between playlists and streams can run both ways: playlists may drive streams, and popular songs may also be added to more playlists.", "amber"),
    ])

    sub("Recommendations")
    lead_measure = corr_table.iloc[0]["Measure"] if not corr_table.empty else "playlist presence"
    insight_cards([
        ("Track playlist reach as an early signal",
         f"In this view, {esc(lead_measure)} is the measure most closely tied to streams. Artists and labels can monitor it as a leading indicator "
         "and test playlist pitching, while checking results over time because this data cannot prove cause and effect.", ""),
        ("Use TikTok for awareness, then plan the conversion",
         "TikTok views show only a loose link with streams, so viral attention does not guarantee listening. "
         "Songs with big TikTok numbers but modest streams (listed below) are candidates for extra promotion on streaming platforms.", "violet"),
        ("Report ranks and log scales in future analysis",
         "Streams and views are highly skewed. Using Spearman correlation, medians and log axes gives a fairer picture than averages and raw scatter plots.", "amber"),
    ])

    tt_pair = observed(filtered, "TikTok Views", "Spotify Streams").copy()
    if len(tt_pair) >= 20:
        tt_pair["tiktok_rank"] = tt_pair["TikTok Views"].rank(pct=True)
        tt_pair["stream_rank"] = tt_pair["Spotify Streams"].rank(pct=True)
        opp = tt_pair[(tt_pair["tiktok_rank"] >= 0.9) & (tt_pair["stream_rank"] <= 0.5)].nlargest(8, "TikTok Views")
        if not opp.empty:
            sub("High TikTok attention, below-median streams")
            st.dataframe(
                opp[["Track", "Artist", "TikTok Views", "Spotify Streams"]], hide_index=True,
                column_config={
                    "TikTok Views": st.column_config.NumberColumn("TikTok views", format="compact"),
                    "Spotify Streams": st.column_config.NumberColumn("Spotify streams", format="compact"),
                },
            )
            note("Songs in the top 10% for TikTok views whose Spotify streams fall at or below the median of the songs in view.")

    sub("Limits of this analysis")
    insight_cards([
        ("Only popular songs",
         "The dataset lists songs that already succeeded. Because it leaves out unpopular songs, relationships found here may be weaker or different across all music.", ""),
        ("Estimated values",
         "Missing numbers were filled with medians during cleaning. This dashboard excludes those values from comparisons and correlations, "
         "but popularity charts still include them.", "violet"),
        ("A single snapshot",
         "The data shows one moment in time, so it cannot show how a song's streams, playlists and social attention change after release.", "amber"),
    ])

st.markdown(
    '<div class="note" style="text-align:center;margin-top:28px">Data: Most Streamed Spotify Songs 2024 (Kaggle). '
    "Built with Streamlit, Pandas and Plotly.</div>",
    unsafe_allow_html=True,
)