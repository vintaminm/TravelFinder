# Our code
import requests
import pandas as pd
from datetime import date, datetime
from calendar import monthrange
from dateutil.relativedelta import relativedelta
import time
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# =============================================================================
# CONFIGURATION
# =============================================================================

AMADEUS_API_KEY = "eh090xutJ5bDzDMvZ4tv5VQqisXv4gxA"
AMADEUS_API_SECRET = "cO8Yq5SDkvAqApVY"
DEFAULT_CURRENCY = "USD"


# =============================================================================
# FLIGHT FETCHER
# =============================================================================

class FlightFetcher:
    """
    Handles Amadeus authentication and flight price retrieval.
    """

    def __init__(self, api_key: str, api_secret: str, currency: str = DEFAULT_CURRENCY):
        self.api_key = api_key
        self.api_secret = api_secret
        self.currency = currency
        self.token = self._get_access_token()

    def _get_access_token(self):
        url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.api_secret,
        }
        resp = requests.post(url, headers=headers, data=data, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("access_token")
        else:
            raise RuntimeError(f"Amadeus auth failed: {resp.status_code} {resp.text}")

    def _ensure_token(self):
        if not self.token:
            raise RuntimeError("No Amadeus access token.")

    def get_price_one_day(self, origin: str, destination: str, date_str: str, max_results: int = 5):
        """
        Fetch the lowest price for a single date.
        """
        self._ensure_token()
        url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        headers = {"Authorization": f"Bearer {self.token}"}
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": date_str,
            "adults": 1,
            "currencyCode": self.currency,
            "max": max_results,
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            if resp.status_code != 200:
                return None
            data = resp.json().get("data", [])
            prices = []
            for offer in data:
                try:
                    prices.append(float(offer["price"]["total"]))
                except Exception:
                    continue
            return min(prices) if prices else None
        except Exception:
            return None

    def sample_month_price(self, origin: str, destination: str, year: int, month: int, sample_days=None):
        """
        Sample several days in given month and aggregate prices.
        """
        if sample_days is None:
            sample_days = [5, 15, 25]

        last_day = monthrange(year, month)[1]
        prices = []
        for d in sample_days:
            day = min(d, last_day)
            date_str = date(year, month, day).strftime("%Y-%m-%d")
            price = self.get_price_one_day(origin, destination, date_str)
            if price is not None:
                prices.append(price)
            time.sleep(0.2)  # be polite

        if not prices:
            return None

        return {
            "min_price": min(prices),
            "avg_price": sum(prices) / len(prices),
            "n_samples": len(prices),
        }

    def fetch_monthly_prices(
        self,
        origin: str,
        destination: str,
        months_ahead: int = 12,
        start_month: str | None = None,
    ) -> pd.DataFrame:
        """
        Build monthly price table starting from start_month (YYYY-MM)
        or from the current calendar month.
        """
        if not start_month:
            start_dt = datetime.today().replace(day=1)
        else:
            start_dt = datetime.strptime(start_month.strip(), "%Y-%m").replace(day=1)

        rows = []
        for i in range(months_ahead):
            month_dt = start_dt + relativedelta(months=i)
            year, month = month_dt.year, month_dt.month
            label = month_dt.strftime("%b %Y")

            stats = self.sample_month_price(origin, destination, year, month)
            if stats is None:
                continue

            rows.append(
                {
                    "Year": year,
                    "Month": month,
                    "Month_Label": label,
                    "Price_USD": stats["min_price"],
                    "Price_Avg": stats["avg_price"],
                    "Price_Samples": stats["n_samples"],
                    "Origin": origin,
                    "Destination": destination,
                }
            )

        return pd.DataFrame(rows)


# =============================================================================
# WEATHER FETCHER
# =============================================================================

class WeatherFetcher:
    def __init__(self):
        self.historical_url = "https://archive-api.open-meteo.com/v1/archive"

    def get_weather(self, latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
            "timezone": "auto",
        }
        try:
            resp = requests.get(self.historical_url, params=params, timeout=20)
            if resp.status_code != 200:
                return pd.DataFrame()
            return pd.DataFrame(resp.json().get("daily", {}))
        except Exception:
            return pd.DataFrame()


# =============================================================================
# MERGE FLIGHTS + WEATHER
# =============================================================================

def add_weather_to_flights(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Join monthly price table with historical monthly weather.
    """
    if df_prices is None or df_prices.empty:
        return df_prices

    airports = pd.read_csv("https://ourairports.com/data/airports.csv")
    airports = airports[airports["iata_code"].notnull()]
    airports = airports[["iata_code", "latitude_deg", "longitude_deg"]]

    df = df_prices.merge(
        airports, left_on="Destination", right_on="iata_code", how="left"
    )
    df.drop(columns=["iata_code"], inplace=True)

    wf = WeatherFetcher()
    today = date.today()
    most_recent_year = today.year - 1

    avg_max_temps = []
    avg_min_temps = []
    total_precips = []
    avg_wind_speeds = []
    weather_types = []

    for _, row in df.iterrows():
        lat, lon, month = row["latitude_deg"], row["longitude_deg"], int(row["Month"])

        if pd.isna(lat) or pd.isna(lon):
            avg_max_temps.append(None)
            avg_min_temps.append(None)
            total_precips.append(None)
            avg_wind_speeds.append(None)
            weather_types.append("missing")
            continue

        start_date_str = date(most_recent_year, month, 1).strftime("%Y-%m-%d")
        end_date_str = date(
            most_recent_year, month, monthrange(most_recent_year, month)[1]
        ).strftime("%Y-%m-%d")

        df_weather = wf.get_weather(lat, lon, start_date_str, end_date_str)
        if df_weather.empty:
            avg_max_temps.append(None)
            avg_min_temps.append(None)
            total_precips.append(None)
            avg_wind_speeds.append(None)
        else:
            avg_max_temps.append(df_weather["temperature_2m_max"].mean())
            avg_min_temps.append(df_weather["temperature_2m_min"].mean())
            total_precips.append(df_weather["precipitation_sum"].sum())
            avg_wind_speeds.append(df_weather["windspeed_10m_max"].mean())

        weather_types.append("historical")

    df["avg_max_temp"] = avg_max_temps
    df["avg_min_temp"] = avg_min_temps
    df["total_precip"] = total_precips
    df["avg_wind_speed"] = avg_wind_speeds
    df["weather_type"] = weather_types

    return df


# =============================================================================
# SCORING
# =============================================================================

def calculate_scores(
    budget: float,
    df: pd.DataFrame,
    trip_type: str = "general",
    price_weight: float = 0.6,
) -> pd.DataFrame:
    """
    Compute comfort_score, price_score and final_score.
    price_weight in [0,1], comfort_weight = 1 - price_weight.
    """

    try:
        price_weight = float(price_weight)
    except Exception:
        price_weight = 0.6
    price_weight = max(0.0, min(1.0, price_weight))
    comfort_weight = 1.0 - price_weight

    presets = {
        "beach": {"ideal_temp": 28, "min_temp": 22, "max_temp": 35, "rain_penalty": 0.3},
        "skiing": {"ideal_temp": -2, "min_temp": -15, "max_temp": 8, "rain_penalty": 0.0},
        "sightseeing": {"ideal_temp": 18, "min_temp": 8, "max_temp": 30, "rain_penalty": 0.5},
        "hiking": {"ideal_temp": 15, "min_temp": 5, "max_temp": 27, "rain_penalty": 0.6},
        "general": {"ideal_temp": 21, "min_temp": 5, "max_temp": 35, "rain_penalty": 0.5},
    }

    cfg = presets.get(trip_type.lower(), presets["general"])
    ideal = cfg["ideal_temp"]
    tmin = cfg["min_temp"]
    tmax = cfg["max_temp"]
    rain_mult = cfg["rain_penalty"]

    df = df.copy()
    T = df["avg_max_temp"]
    R = df["total_precip"]

    # comfort score
    comfort_scores = []
    for temp, rain in zip(T, R):
        if pd.isna(temp) or pd.isna(rain):
            comfort_scores.append(0.0)
            continue
        if temp < tmin or temp > tmax:
            comfort_scores.append(0.0)
            continue
        base = 100.0 - 2.5 * abs(temp - ideal)
        base -= rain_mult * rain
        comfort_scores.append(max(0.0, min(100.0, base)))
    df["comfort_score"] = comfort_scores

    # price score
    Pm = df["Price_USD"]
    Pmin = Pm.min()
    Pmax = Pm.max()
    price_scores = []
    for price in Pm:
        if Pmax > Pmin:
            if price > budget:
                score = 0.0
            else:
                score = 100.0 * (Pmax - price) / (Pmax - Pmin)
        else:
            score = 100.0
        price_scores.append(max(0.0, min(100.0, score)))
    df["price_score"] = price_scores

    # final score
    df["final_score"] = price_weight * df["price_score"] + comfort_weight * df["comfort_score"]
    df["price_weight_used"] = price_weight
    df["comfort_weight_used"] = comfort_weight

    return df


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def make_figure(df: pd.DataFrame, origin: str, destination: str, trip_type: str):
    """
    Build the 3-panel matplotlib figure and return it.
    """
    if df.empty:
        return None

    df_sorted = df.sort_values(["Year", "Month"]).reset_index(drop=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.35)

    # Chart 1: price vs comfort over time
    ax1 = axes[0]
    x_pos = range(len(df_sorted))
    ax1.bar(x_pos, df_sorted["Price_USD"], color="steelblue", alpha=0.75, label="Price (USD)")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Price (USD)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xticks(list(x_pos))
    ax1.set_xticklabels(df_sorted["Month_Label"], rotation=45, ha="right")

    ax1_t = ax1.twinx()
    ax1_t.plot(
        x_pos,
        df_sorted["comfort_score"],
        color="coral",
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="Comfort Score",
    )
    ax1_t.set_ylabel("Comfort Score", color="coral")
    ax1_t.tick_params(axis="y", labelcolor="coral")
    ax1_t.set_ylim(0, 100)

    ax1.set_title(f"Price vs Comfort Over Time\n{origin} → {destination}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    # Chart 2: decision scatter
    ax2 = axes[1]
    scatter = ax2.scatter(
        df_sorted["comfort_score"],
        df_sorted["Price_USD"],
        s=df_sorted["final_score"] * 3,
        c=df_sorted["final_score"],
        cmap="RdYlGn",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # annotate top-3 months by final score
    top_labels = set(
        df_sorted.sort_values("final_score", ascending=False).head(3)["Month_Label"]
    )
    for _, row in df_sorted.iterrows():
        if row["Month_Label"] not in top_labels:
            continue
        ax2.annotate(
            row["Month_Label"],
            (row["comfort_score"], row["Price_USD"]),
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax2.set_xlabel("Comfort Score")
    ax2.set_ylabel("Price (USD)")
    ax2.set_title(f"Price vs Comfort Decision Matrix\nTrip Type: {trip_type.capitalize()}")
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Final Score")

    ax2.axvline(x=50, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax2.axhline(y=df_sorted["Price_USD"].median(), color="gray", linestyle="--", alpha=0.3, linewidth=1)

    # Chart 3: ranking barh
    ax3 = axes[2]
    colors = plt.cm.RdYlGn(df_sorted["final_score"] / 100.0)
    ax3.barh(
        range(len(df_sorted)),
        df_sorted["final_score"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.set_yticks(range(len(df_sorted)))
    ax3.set_yticklabels(df_sorted["Month_Label"])
    ax3.set_xlabel("Final Score")
    ax3.set_title("Month Rankings by Final Score\n(Chronological Order)")
    ax3.set_xlim(0, 100)
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        score = row["final_score"]
        ax3.text(score + 1, i, f"{score:.1f}", va="center", fontsize=8)
    ax3.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# PIPELINE FOR ONE ROUTE
# =============================================================================

def run_pipeline_for_route(
    origin: str,
    destination: str,
    trip_type: str,
    start_month: str,
    months_ahead: int,
    budget: float,
    price_weight: float,
) -> pd.DataFrame | None:
    try:
        ff = FlightFetcher(AMADEUS_API_KEY, AMADEUS_API_SECRET)
    except Exception as e:
        st.error(f"Amadeus authentication failed: {e}")
        return None

    df_prices = ff.fetch_monthly_prices(origin, destination, months_ahead, start_month)
    if df_prices is None or df_prices.empty:
        st.warning(f"No flight price data for {origin} → {destination}.")
        return None

    df_full = add_weather_to_flights(df_prices)
    if df_full is None or df_full.empty:
        st.warning(f"No weather data for {origin} → {destination}.")
        return None

    df_scored = calculate_scores(budget, df_full, trip_type, price_weight)
    df_scored = df_scored.sort_values("final_score", ascending=False).reset_index(drop=True)
    return df_scored


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(page_title="Destination Price & Weather Optimizer", layout="wide")

    st.title("Destination Price & Weather Optimizer")
    st.markdown(
        "Find the best **month to travel** by balancing flight prices and weather comfort."
    )

    # Sidebar controls
    st.sidebar.header("Trip Settings")
    mode = st.sidebar.radio(
        "Mode",
        ["Single destination", "Compare multiple destinations"],
        index=0,
    )

    origin = st.sidebar.text_input("Origin airport code (IATA)", "JFK").upper()
    trip_type = st.sidebar.selectbox(
        "Trip type",
        ["general", "beach", "skiing", "sightseeing", "hiking"],
        index=0,
    )
    start_month = st.sidebar.text_input(
        "Start month (YYYY-MM, leave blank for current month)", ""
    )
    months_ahead = st.sidebar.slider("Months ahead to analyze", 3, 18, 12)

    budget_input = st.sidebar.text_input(
        "Budget in USD (leave blank for no limit)", ""
    )
    if budget_input.strip() == "":
        budget = float("inf")
    else:
        try:
            budget = float(budget_input)
        except ValueError:
            st.sidebar.warning("Invalid budget, treating as no limit.")
            budget = float("inf")

    price_weight = st.sidebar.slider(
        "Price importance vs comfort (0 = only comfort, 1 = only price)",
        0.0,
        1.0,
        0.6,
        0.05,
    )
    st.sidebar.markdown(
        f"**Current weights**: Price {price_weight*100:.0f}%, Comfort {100 - price_weight*100:.0f}%"
    )

    if mode == "Single destination":
        dest = st.sidebar.text_input("Destination airport code (IATA)", "LHR").upper()
        run_button = st.sidebar.button("Run analysis")
        if run_button:
            if not origin or not dest:
                st.error("Please enter both origin and destination airport codes.")
                return
            with st.spinner(f"Analyzing {origin} → {dest} ..."):
                df_scored = run_pipeline_for_route(
                    origin,
                    dest,
                    trip_type,
                    start_month,
                    months_ahead,
                    budget,
                    price_weight,
                )
            if df_scored is not None and not df_scored.empty:
                best = df_scored.iloc[0]
                st.subheader(f"Best month for {origin} → {dest}")
                st.write(
                    f"**{best['Month_Label']}**  ·  "
                    f"Estimated lowest fare: **${best['Price_USD']:.0f}**  ·  "
                    f"Comfort score: **{best['comfort_score']:.1f}/100**  ·  "
                    f"Final score: **{best['final_score']:.1f}/100**"
                )

                st.markdown("#### Top 3 recommended months")
                cols = [
                    "Month_Label",
                    "Price_USD",
                    "avg_max_temp",
                    "total_precip",
                    "comfort_score",
                    "price_score",
                    "final_score",
                ]
                st.dataframe(df_scored[cols].head(3))

                st.markdown("#### Full monthly table")
                st.dataframe(df_scored)

                fig = make_figure(df_scored, origin, dest, trip_type)
                if fig is not None:
                    st.markdown("#### Visualizations")
                    st.pyplot(fig)

    else:
        dest_string = st.sidebar.text_input(
            "Destination airport codes (comma separated)",
            "LHR,CDG,JFK",
        )
        run_button = st.sidebar.button("Run comparison")

        if run_button:
            dest_list = [d.strip().upper() for d in dest_string.split(",") if d.strip()]
            if not origin or not dest_list:
                st.error("Please enter origin and at least one destination.")
                return

            summary_rows = []
            tabs = st.tabs(dest_list)

            for dest, tab in zip(dest_list, tabs):
                with tab:
                    st.subheader(f"{origin} → {dest}")
                    with st.spinner(f"Analyzing {origin} → {dest} ..."):
                        df_scored = run_pipeline_for_route(
                            origin,
                            dest,
                            trip_type,
                            start_month,
                            months_ahead,
                            budget,
                            price_weight,
                        )
                    if df_scored is None or df_scored.empty:
                        st.warning("No data for this route.")
                        continue

                    best = df_scored.iloc[0]
                    summary_rows.append(
                        {
                            "Destination": dest,
                            "Best_Month": best["Month_Label"],
                            "Best_Price_USD": best["Price_USD"],
                            "Best_Comfort": best["comfort_score"],
                            "Best_Final_Score": best["final_score"],
                        }
                    )

                    st.write(
                        f"**Best month:** {best['Month_Label']}  ·  "
                        f"Price ≈ ${best['Price_USD']:.0f}  ·  "
                        f"Comfort {best['comfort_score']:.1f}/100  ·  "
                        f"Final {best['final_score']:.1f}/100"
                    )

                    st.markdown("**Top 3 months for this route**")
                    cols = [
                        "Month_Label",
                        "Price_USD",
                        "avg_max_temp",
                        "total_precip",
                        "comfort_score",
                        "price_score",
                        "final_score",
                    ]
                    st.dataframe(df_scored[cols].head(3))

                    fig = make_figure(df_scored, origin, dest, trip_type)
                    if fig is not None:
                        st.pyplot(fig)

            if summary_rows:
                st.markdown("### Multi-destination summary (sorted by best final score)")
                df_summary = pd.DataFrame(summary_rows).sort_values(
                    "Best_Final_Score", ascending=False
                )
                st.dataframe(df_summary)


if __name__ == "__main__":
    main()

