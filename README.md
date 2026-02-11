"""
Data Science Project: Social Media Shorts/Reels and Tourism Growth

This script is beginner-friendly and runs end-to-end:
1) Creates sample data (if no dataset file exists)
2) Loads and cleans the data
3) Builds a few useful features
4) Creates visualizations (line, bar, scatter, map)

Run:
    python datascience.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# 1) Project paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = DATA_DIR / "social_media_tourism_sample.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# 2) Create sample dataset
# -------------------------
def create_sample_data(file_path: Path) -> None:
    """Create a realistic sample dataset for 2010-2025 tourism + short video influence."""
    np.random.seed(42)

    # City-level tourism destinations often discussed in short-form travel content
    destinations = [
        ("Bali", "Indonesia", "IDN"),
        ("Dubai", "United Arab Emirates", "ARE"),
        ("Santorini", "Greece", "GRC"),
        ("Kyoto", "Japan", "JPN"),
        ("Cappadocia", "Turkey", "TUR"),
        ("Banff", "Canada", "CAN"),
        ("Marrakech", "Morocco", "MAR"),
        ("Reykjavik", "Iceland", "ISL"),
    ]

    years = list(range(2010, 2026))
    rows = []

    for city, country, iso3 in destinations:
        base_tourists = np.random.uniform(1.0, 9.0)  # in millions
        base_social = np.random.uniform(5, 18)

        for i, year in enumerate(years):
            # Social media trend: starts low, rises strongly after 2016
            if year < 2016:
                social_boost = i * np.random.uniform(1.2, 2.4)
            else:
                social_boost = i * np.random.uniform(2.4, 4.0)

            reels_index = min(100, base_social + social_boost + np.random.normal(0, 2.5))
            shorts_index = min(100, base_social + social_boost * 0.95 + np.random.normal(0, 2.8))

            # Tourism is affected by base growth + social media index
            tourists = (
                base_tourists
                + i * np.random.uniform(0.05, 0.25)
                + (reels_index + shorts_index) / 250
                + np.random.normal(0, 0.25)
            )

            # Pandemic dip example in 2020-2021
            if year == 2020:
                tourists *= np.random.uniform(0.35, 0.55)
            if year == 2021:
                tourists *= np.random.uniform(0.55, 0.80)

            tourists = max(0.2, tourists)

            rows.append(
                {
                    "year": str(year),
                    "city": city,
                    "country": country,
                    "iso3": iso3,
                    "tourists_millions": f"{tourists:,.2f}",
                    "reels_index": f"{reels_index:.1f}%",  # intentionally string+%
                    "shorts_index": f"{shorts_index:.1f}%",  # intentionally string+%
                }
            )

    df = pd.DataFrame(rows)

    # Add a few missing values for cleaning practice
    missing_idx = np.random.choice(df.index, size=6, replace=False)
    df.loc[missing_idx[:2], "reels_index"] = np.nan
    df.loc[missing_idx[2:4], "shorts_index"] = np.nan
    df.loc[missing_idx[4:], "tourists_millions"] = np.nan

    df.to_csv(file_path, index=False)


# -------------------------
# 3) Load and clean data
# -------------------------
def load_data(file_path: Path) -> pd.DataFrame:
    """Read CSV data into a DataFrame."""
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text/numeric columns and fill missing values."""
    clean_df = df.copy()

    # Remove spaces and normalize text
    for col in ["city", "country", "iso3"]:
        clean_df[col] = clean_df[col].astype(str).str.strip()

    # Convert year to integer
    clean_df["year"] = pd.to_numeric(clean_df["year"], errors="coerce").astype("Int64")

    # Convert strings like "12.3%" or "1,234.5" to numeric
    for col in ["tourists_millions", "reels_index", "shorts_index"]:
        clean_df[col] = (
            clean_df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .replace("nan", np.nan)
        )
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # Fill missing values using city-level median, then overall median
    for col in ["tourists_millions", "reels_index", "shorts_index"]:
        clean_df[col] = clean_df.groupby("city")[col].transform(lambda s: s.fillna(s.median()))
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # Drop rows where key fields are still missing
    clean_df = clean_df.dropna(subset=["year", "city", "tourists_millions"])

    # Convert year back to plain int for plotting
    clean_df["year"] = clean_df["year"].astype(int)

    return clean_df


# -------------------------
# 4) Feature engineering
# -------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a combined social media metric and yearly growth metric."""
    feat_df = df.copy()

    feat_df["social_media_index"] = (feat_df["reels_index"] + feat_df["shorts_index"]) / 2

    feat_df = feat_df.sort_values(["city", "year"])  # needed for growth calculation
    feat_df["tourist_growth_pct"] = (
        feat_df.groupby("city")["tourists_millions"].pct_change() * 100
    )

    return feat_df


# -------------------------
# 5) Visualizations
# -------------------------
def plot_line_chart(df: pd.DataFrame) -> None:
    """Line chart: tourism trend across years for top social-media destinations."""
    top_cities = (
        df.groupby("city")["social_media_index"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    subset = df[df["city"].isin(top_cities)]

    plt.figure(figsize=(11, 6))
    sns.lineplot(data=subset, x="year", y="tourists_millions", hue="city", marker="o")
    plt.title("Tourist Arrivals Over Time (Top 5 Social-Media Destinations)")
    plt.xlabel("Year")
    plt.ylabel("Tourists (Millions)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "line_tourism_trend.png", dpi=150)
    plt.close()


def plot_bar_chart(df: pd.DataFrame) -> None:
    """Bar chart: average social media index by city."""
    city_social = (
        df.groupby("city", as_index=False)["social_media_index"].mean()
        .sort_values("social_media_index", ascending=False)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=city_social, x="social_media_index", y="city", palette="viridis")
    plt.title("Average Social Media Interest Index by Destination")
    plt.xlabel("Average Social Media Index")
    plt.ylabel("City")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bar_social_index_by_city.png", dpi=150)
    plt.close()


def plot_scatter(df: pd.DataFrame) -> None:
    """Scatter plot: relationship between social media index and tourism."""
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df,
        x="social_media_index",
        y="tourists_millions",
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red"},
    )
    plt.title("Relationship: Social Media Index vs Tourist Arrivals")
    plt.xlabel("Social Media Index (Reels + Shorts)")
    plt.ylabel("Tourists (Millions)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_social_vs_tourists.png", dpi=150)
    plt.close()


def plot_map(df: pd.DataFrame) -> None:
    """Choropleth map for latest year using Plotly, with fallback to country bar chart."""
    latest_year = df["year"].max()
    latest = (
        df[df["year"] == latest_year]
        .groupby(["country", "iso3"], as_index=False)["tourists_millions"]
        .sum()
    )

    try:
        import plotly.express as px

        fig = px.choropleth(
            latest,
            locations="iso3",
            color="tourists_millions",
            hover_name="country",
            color_continuous_scale="Blues",
            title=f"Tourists by Country ({latest_year})",
        )
        fig.write_html(str(OUTPUT_DIR / "map_tourists_latest_year.html"))
    except Exception:
        # If Plotly is unavailable, we still provide a map-alternative chart.
        plt.figure(figsize=(10, 6))
        latest_sorted = latest.sort_values("tourists_millions", ascending=False)
        sns.barplot(data=latest_sorted, x="tourists_millions", y="country", palette="crest")
        plt.title(f"Tourists by Country ({latest_year})")
        plt.xlabel("Tourists (Millions)")
        plt.ylabel("Country")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "country_tourists_latest_year.png", dpi=150)
        plt.close()


# -------------------------
# 6) Main workflow
# -------------------------
def main() -> None:
    if not DATA_FILE.exists():
        create_sample_data(DATA_FILE)
        print(f"Created sample dataset: {DATA_FILE}")

    raw_df = load_data(DATA_FILE)
    clean_df = clean_data(raw_df)
    final_df = build_features(clean_df)

    # Save cleaned data so you can inspect it later.
    clean_output = OUTPUT_DIR / "cleaned_social_media_tourism.csv"
    final_df.to_csv(clean_output, index=False)

    plot_line_chart(final_df)
    plot_bar_chart(final_df)
    plot_scatter(final_df)
    plot_map(final_df)

    print("\nProject complete.")
    print(f"Rows in cleaned dataset: {len(final_df)}")
    print(f"Saved cleaned data: {clean_output}")
    print(f"Saved visualizations in: {OUTPUT_DIR}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()

