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
# CONFIG
# =============================================================================

AMADEUS_API_KEY = "eh090xutJ5bDzDMvZ4tv5VQqisXv4gxA"
AMADEUS_API_SECRET = "cO8Yq5SDkvAqApVY"
DEFAULT_CURRENCY = "USD"


# =============================================================================
# FLIGHT & AIRPORT FETCHER (Amadeus)
# =============================================================================

class FlightFetcher:
    """
    Handles:
    - Amadeus authentication
    - Flight price lookup
    - Airport coordinates lookup (via Amadeus Locations API)
    """

    def __init__(self, api_key, api_secret, currency=DEFAULT_CURRENCY):
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
        try:
            resp = requests.post(url, headers=headers, data=data, timeout=15)
        except Exception as e:
            st.sidebar.error(f"Amadeus auth request failed: {e}")
            return None

        if resp.status_code == 200:
            st.sidebar.success("Amadeus auth OK.")
            return resp.json().get("access_token")
        else:
            st.sidebar.error(f"Amadeus auth failed: {resp.status_code}")
            st.sidebar.write(resp.text)
            return None

    def _ensure_token(self):
        if not self.token:
            raise RuntimeError("No Amadeus access token. Check your API keys.")

    # ---------- Flight price (per day) ----------

    def get_price_one_day(self, origin, destination, date_str, max_results=5):
        """Lowest price for a given date."""
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
        except Exception:
            return None

        if resp.status_code != 200:
            return None

        data = resp.json().get("data", [])
        prices = []
        for offer in data:
            price_info = offer.get("price")
            if not price_info:
                continue
            try:
                prices.append(float(price_info["total"]))
            except Exception:
                continue
        return min(prices) if prices else None

    def sample_month_price(self, origin, destination, year, month, sample_days=None):
        """Sample 3 days in a month → min/avg price."""
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
            time.sleep(0.25)

        if not prices:
            return None

        return {
            "min_price": min(prices),
            "avg_price": sum(prices) / len(prices),
            "n_samples": len(prices),
        }

    def fetch_monthly_prices(self, origin, destination, months_ahead=12, start_month=None):
        """Loop future months and build monthly price table."""
        if start_month and start_month.strip():
            start_dt = datetime.strptime(start_month.strip(), "%Y-%m").replace(day=1)
        else:
            start_dt = datetime.today().replace(day=1)

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

    # ---------- Airport coordinates via Amadeus Locations API ----------

    def get_airport_coordinates(self, iata_code: str):
        """
        Query Amadeus reference data to get airport latitude/longitude.
        Uses: /v1/reference-data/locations?subType=AIRPORT&keyword=IATA
        """
        self._ensure_token()
        url = "https://test.api.amadeus.com/v1/reference-data/locations"
        headers = {"Authorization": f"Bearer {self.token}"}
        params = {
            "subType": "AIRPORT",
            "keyword": iata_code,
            "page[limit]": 5,
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code != 200:
                return None, None
            data = resp.json().get("data", [])
            # Prefer exact IATA match if present
            for item in data:
                if item.get("iataCode") == iata_code:
                    geo = item.get("geoCode", {})
                    return geo.get("latitude"), geo.get("longitude")
            # Fallback: first result
            if data:
                geo = data[0].get("geoCode", {})
                return geo.get("latitude"), geo.get("longitude")
        except Exception:
            pass
        return None, None


# =============================================================================
# WEATHER FETCHER (Open-Meteo Archive)
# =============================================================================

class WeatherFetcher:
    def __init__(self):
        self.historical_url = "https://archive-api.open-meteo.com/v1/archive"

    def get_weather(self, latitude, longitude, start_date, end_date):
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
        except Exception:
            return pd.DataFrame()

        if resp.status_code != 200:
            return pd.DataFrame()
        return pd.DataFrame(resp.json().get("daily", {}))


# =============================================================================
# MERGE FLIGHTS + WEATHER (NO CSV DATASET)
# =============================================================================

def add_weather_to_flights(df_prices):
    """
    Enrich monthly price table with weather statistics.
    Coordinates are obtained via Amadeus Locations API (no static CSV).
    """
    if df_prices is None or df_prices.empty:
        return df_prices

    # Use Amadeus to get airport coordinates
    ff = FlightFetcher(AMADEUS_API_KEY, AMADEUS_API_SECRET)
    if not ff.token:
        st.sidebar.error("Failed to fetch airport coordinates from Amadeus.")
        return df_prices

    wf = WeatherFetcher()
    today = date.today()
    most_recent_year = today.year - 1

    latitudes, longitudes = [], []
    avg_max_temps, avg_min_temps = [], []
    total_precips, avg_wind_speeds = [], []
    weather_types = []

    coord_cache = {}  # cache per IATA code

    for _, row in df_prices.iterrows():
        dest = row["Destination"]
        month = int(row["Month"])

        if dest in coord_cache:
            lat, lon = coord_cache[dest]
        else:
            lat, lon = ff.get_airport_coordinates(dest)
            coord_cache[dest] = (lat, lon)

        latitudes.append(lat)
        longitudes.append(lon)

        if lat is None or lon is None:
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

    df = df_prices.copy()
    df["latitude"] = latitudes
    df["longitude"] = longitudes
    df["avg_max_temp"] = avg_max_temps
    df["avg_min_temp"] = avg_min_temps
    df["total_precip"] = total_precips
    df["avg_wind_speed"] = avg_wind_speeds
    df["weather_type"] = weather_types

    return df


# =============================================================================
# SCORING
# =============================================================================

def calculate_scores(budget, df, trip_type="general", price_weight=0.6):
    """Compute comfort_score, price_score, final_score."""
    try:
        price_weight = float(price_weight)
    except Exception:
        price_weight = 0.6
    price_weight = max(0.0, min(1.0, price_weight))
    comfort_weight = 1.0 - price_weight

    presets = {
        "beach":      {"ideal_temp": 28, "min_temp": 22, "max_temp": 35, "rain_penalty": 0.3},
        "skiing":     {"ideal_temp": -2, "min_temp": -15, "max_temp": 8,  "rain_penalty": 0.0},
        "sightseeing":{"ideal_temp": 18, "min_temp": 8,  "max_temp": 30, "rain_penalty": 0.5},
        "hiking":     {"ideal_temp": 15, "min_temp": 5,  "max_temp": 27, "rain_penalty": 0.6},
        "general":    {"ideal_temp": 21, "min_temp": 5,  "max_temp": 35, "rain_penalty": 0.5},
    }

    cfg = presets.get(trip_type.lower(), presets["general"])
    ideal = cfg["ideal_temp"]
    tmin = cfg["min_temp"]
    tmax = cfg["max_temp"]
    rain_mult = cfg["rain_penalty"]

    df = df.copy()
    T = df["avg_max_temp"]
    R = df["total_precip"]

    # Comfort score
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

    # Price score
    Pm = df["Price_USD"]
    Pmin = Pm.min()
    Pmax = Pm.max()
    price_scores = []
    for price in Pm:
        if Pmax > Pmin:
            if budget is not None and budget > 0 and price > budget:
                score = 0.0
            else:
                score = 100.0 * (Pmax - price) / (Pmax - Pmin)
        else:
            score = 100.0
        price_scores.append(max(0.0, min(100.0, score)))
    df["price_score"] = price_scores

    # Final score
    df["final_score"] = price_weight * df["price_score"] + comfort_weight * df["comfort_score"]
    df["price_weight_used"] = price_weight
    df["comfort_weight_used"] = comfort_weight

    return df


# =============================================================================
# FIGURE (BIGGER & FIRST)
# =============================================================================

def make_figure(df, origin, destination, trip_type):
    """Return a big 3-panel matplotlib Figure."""
    if df.empty:
        return None

    sns.set_theme(style="whitegrid")
    df_sorted = df.sort_values(["Year", "Month"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    plt.subplots_adjust(wspace=0.35)

    # 1) Price vs Comfort over time
    ax1 = axes[0]
    x_pos = range(len(df_sorted))
    ax1.bar(x_pos, df_sorted["Price_USD"], color="steelblue", alpha=0.75, label="Price (USD)")
    ax1.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Price (USD)", fontsize=11, fontweight="bold", color="steelblue")
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
    ax1_t.set_ylabel("Comfort Score", fontsize=11, fontweight="bold", color="coral")
    ax1_t.tick_params(axis="y", labelcolor="coral")
    ax1_t.set_ylim(0, 100)

    ax1.set_title(f"Price vs Comfort Over Time\n{origin} → {destination}",
                  fontsize=13, fontweight="bold", pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    # 2) Decision scatter
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

    top_labels = set(df_sorted.sort_values("final_score", ascending=False).head(3)["Month_Label"])
    for _, row in df_sorted.iterrows():
        if row["Month_Label"] not in top_labels:
            continue
        short_label = row["Month_Label"].split()[0]
        ax2.annotate(
            short_label,
            (row["comfort_score"] + 1, row["Price_USD"] + 3),
            fontsize=8,
            ha="left",
            va="bottom",
        )

    ax2.set_xlabel("Comfort Score", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Price (USD)", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"Price vs Comfort Decision Matrix\nTrip Type: {trip_type.capitalize()}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Final Score", fontsize=10, fontweight="bold")

    ax2.axvline(x=50, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax2.axhline(
        y=df_sorted["Price_USD"].median(),
        color="gray",
        linestyle="--",
        alpha=0.3,
        linewidth=1,
    )

    # 3) Final score ranking
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
    ax3.set_xlabel("Final Score", fontsize=11, fontweight="bold")
    ax3.set_title(
        "Month Rankings by Final Score\n(Chronological Order)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax3.set_xlim(0, 100)

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        score = row["final_score"]
        ax3.text(score + 1, i, f"{score:.1f}", va="center", fontsize=9, fontweight="bold")

    ax3.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# PIPELINES WITH PROGRESS BAR
# =============================================================================

def single_route_analysis(origin, destination, trip_type, start_month, months_ahead, budget, price_weight):
    status = st.empty()
    progress = st.progress(0)

    status.info("Step 1/4: Authenticating with Amadeus ...")
    ff = FlightFetcher(AMADEUS_API_KEY, AMADEUS_API_SECRET)
    if not ff.token:
        status.error("Amadeus authentication failed. Check API keys.")
        progress.empty()
        return None
    progress.progress(20)

    status.info("Step 2/4: Fetching monthly flight prices ...")
    df_prices = ff.fetch_monthly_prices(origin, destination, months_ahead, start_month)
    if df_prices is None or df_prices.empty:
        status.error("No flight price data for this route (sandbox limitation).")
        progress.empty()
        return None
    progress.progress(45)

    status.info("Step 3/4: Fetching historical weather data ...")
    df_full = add_weather_to_flights(df_prices)
    if df_full is None or df_full.empty:
        status.error("No weather data available for this destination.")
        progress.empty()
        return None
    progress.progress(75)

    status.info("Step 4/4: Scoring months and building charts ...")
    eff_budget = float("inf") if (budget is None or budget <= 0) else budget
    df_scored = calculate_scores(eff_budget, df_full, trip_type, price_weight)
    progress.progress(100)
    status.success("Analysis complete.")
    time.sleep(0.3)
    progress.empty()
    status.empty()
    return df_scored


def multi_route_analysis(origin, dest_list, trip_type, start_month, months_ahead, budget, price_weight):
    status = st.empty()
    progress = st.progress(0)

    status.info("Step 1/3: Authenticating with Amadeus ...")
    ff = FlightFetcher(AMADEUS_API_KEY, AMADEUS_API_SECRET)
    if not ff.token:
        status.error("Amadeus authentication failed. Check API keys.")
        progress.empty()
        return None, None
    progress.progress(15)

    all_results = {}
    summary_rows = []
    total = len(dest_list)

    for idx, dest in enumerate(dest_list, start=1):
        status.info(f"Step 2/3: Route {idx}/{total} — fetching data for {origin} → {dest} ...")
        df_prices = ff.fetch_monthly_prices(origin, dest, months_ahead, start_month)
        if df_prices is None or df_prices.empty:
            continue

        df_full = add_weather_to_flights(df_prices)
        if df_full is None or df_full.empty:
            continue

        eff_budget = float("inf") if (budget is None or budget <= 0) else budget
        df_scored = calculate_scores(eff_budget, df_full, trip_type, price_weight)
        if df_scored is None or df_scored.empty:
            continue

        all_results[dest] = df_scored
        best = df_scored.sort_values("final_score", ascending=False).iloc[0]
        summary_rows.append(
            {
                "Destination": dest,
                "Best_Month": best["Month_Label"],
                "Best_Price_USD": best["Price_USD"],
                "Best_Comfort": best["comfort_score"],
                "Best_Final_Score": best["final_score"],
            }
        )
        frac = idx / float(total)
        progress.progress(min(15 + int(frac * 80), 95))

    if not summary_rows:
        status.error("No data collected for any destination.")
        progress.empty()
        return None, None

    status.info("Step 3/3: Finalizing comparison ...")
    df_summary = pd.DataFrame(summary_rows).sort_values("Best_Final_Score", ascending=False).reset_index(drop=True)
    progress.progress(100)
    status.success("Comparison complete.")
    time.sleep(0.3)
    progress.empty()
    status.empty()
    return df_summary, all_results


# =============================================================================
# STREAMLIT MAIN
# =============================================================================

def main():
    st.set_page_config(page_title="Destination Price & Weather Optimizer", layout="wide")
    st.title("Destination Price & Weather Optimizer")
    st.write(
        "Find the best **month to travel** by combining live flight prices (Amadeus) "
        "with historical weather comfort (Open‑Meteo)."
    )
    st.markdown("---")

    col_top = st.columns(3)
    with col_top[0]:
        mode = st.radio("Mode", ["Single destination", "Multi-destination comparison"])
    with col_top[1]:
        trip_type = st.selectbox("Trip type", ["general", "beach", "skiing", "sightseeing", "hiking"], index=0)
    with col_top[2]:
        months_ahead = st.slider("Months ahead to analyze", 3, 18, 12)

    col2 = st.columns(3)
    with col2[0]:
        origin = st.text_input("Origin airport (IATA)", value="JFK").upper()
    with col2[1]:
        start_default = datetime.today().strftime("%Y-%m")
        start_month = st.text_input("Start month (YYYY-MM, blank = current month)", value=start_default)
        if not start_month.strip():
            start_month = None
    with col2[2]:
        budget_str = st.text_input("Budget in USD (blank = no limit)", value="")
        if budget_str.strip():
            try:
                budget = float(budget_str)
            except Exception:
                budget = None
                st.warning("Invalid budget, treating as no limit.")
        else:
            budget = None

    price_weight = st.slider(
        "Price importance vs comfort (0 = only comfort, 1 = only price)",
        0.0, 1.0, 0.6, 0.05,
    )
    st.caption(f"Current weights → Price: {price_weight:.2f}, Comfort: {1 - price_weight:.2f}")
    st.markdown("---")

    if mode == "Single destination":
        dest = st.text_input("Destination airport (IATA)", value="LHR").upper()
        run_single = st.button("Run single-destination analysis", type="primary")

        if run_single:
            if not origin or not dest:
                st.error("Please enter both origin and destination codes.")
            else:
                df_scored = single_route_analysis(
                    origin, dest, trip_type, start_month, months_ahead, budget, price_weight
                )
                if df_scored is not None and not df_scored.empty:
                    df_sorted = df_scored.sort_values("final_score", ascending=False).reset_index(drop=True)
                    best = df_sorted.iloc[0]

                    st.subheader(f"Best month for {origin} → {dest}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Best month", best["Month_Label"])
                    m2.metric("Est. lowest fare", f"${best['Price_USD']:.0f}")
                    m3.metric("Final score", f"{best['final_score']:.1f} / 100")

                    st.markdown("### Visual summary")
                    fig = make_figure(df_scored, origin, dest, trip_type)
                    if fig is not None:
                        st.pyplot(fig, use_container_width=True)

                    st.markdown("### Top 3 recommended months")
                    cols = [
                        "Month_Label",
                        "Price_USD",
                        "avg_max_temp",
                        "total_precip",
                        "comfort_score",
                        "price_score",
                        "final_score",
                    ]
                    st.dataframe(df_sorted[cols].head(3), use_container_width=True)

                    with st.expander("Full monthly table"):
                        st.dataframe(df_sorted, use_container_width=True)

                    csv = df_sorted.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download full analysis as CSV",
                        csv,
                        file_name=f"travel_analysis_{origin}_{dest}.csv",
                        mime="text/csv",
                    )

    else:
        dest_str = st.text_input(
            "Destination airports (comma-separated IATA codes)",
            value="LHR, CDG, DEN",
        )
        dest_list = [d.strip().upper() for d in dest_str.split(",") if d.strip()]
        run_multi = st.button("Run multi-destination comparison", type="primary")

        if run_multi:
            if not origin or not dest_list:
                st.error("Please enter origin and at least one destination.")
            else:
                df_summary, all_results = multi_route_analysis(
                    origin, dest_list, trip_type, start_month, months_ahead, budget, price_weight
                )
                if df_summary is not None and all_results:
                    st.subheader(f"Summary: best month per destination from {origin}")
                    st.dataframe(
                        df_summary.style.format(
                            {
                                "Best_Price_USD": "{:.0f}",
                                "Best_Comfort": "{:.1f}",
                                "Best_Final_Score": "{:.1f}",
                            }
                        ),
                        use_container_width=True,
                    )

                    st.markdown("### Detailed view")
                    choices = ["(select destination)"] + list(df_summary["Destination"])
                    chosen = st.selectbox("Pick a destination for charts", choices)
                    if chosen != "(select destination)" and chosen in all_results:
                        df_chosen = all_results[chosen]
                        best = df_chosen.sort_values("final_score", ascending=False).iloc[0]

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Best month", best["Month_Label"])
                        c2.metric("Est. lowest fare", f"${best['Price_USD']:.0f}")
                        c3.metric("Final score", f"{best['final_score']:.1f} / 100")

                        fig = make_figure(df_chosen, origin, chosen, trip_type)
                        if fig is not None:
                            st.pyplot(fig, use_container_width=True)

                        with st.expander("Full monthly table for this route"):
                            st.dataframe(
                                df_chosen.sort_values("final_score", ascending=False).reset_index(drop=True),
                                use_container_width=True,
                            )

                        csv = df_chosen.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            f"Download CSV for {origin}→{chosen}",
                            csv,
                            file_name=f"travel_analysis_{origin}_{chosen}.csv",
                            mime="text/csv",
                        )


if __name__ == "__main__":
    main()
