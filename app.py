import time
from datetime import date, datetime
from calendar import monthrange

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dateutil.relativedelta import relativedelta

# =============================================================================
# CONFIG
# =============================================================================
AMADEUS_API_KEY = "eh090xutJ5bDzDMvZ4tv5VQqisXv4gxA"
AMADEUS_API_SECRET = "cO8Yq5SDkvAqApVY"
DEFAULT_CURRENCY = "USD"

AMADEUS_BASE = "https://test.api.amadeus.com"  # test environment base
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# =============================================================================
# SESSION STATE INIT
# =============================================================================
def init_state():
    st.session_state.setdefault("single_scored", None)
    st.session_state.setdefault("single_config", None)

    st.session_state.setdefault("multi_summary", None)
    st.session_state.setdefault("multi_results", None)
    st.session_state.setdefault("multi_config", None)

    st.session_state.setdefault("geo_cache", {})      # iata -> (lat, lon)
    st.session_state.setdefault("climate_cache", {})  # (lat, lon, year) -> monthly climate df
    st.session_state.setdefault("detail_dest", "Select a destination")


# =============================================================================
# AMADEUS CLIENT (token + GET helper)
# =============================================================================
class AmadeusClient:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.token = None
        self._auth()

    def _auth(self):
        url = f"{AMADEUS_BASE}/v1/security/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.api_secret,
        }
        resp = requests.post(url, headers=headers, data=data, timeout=20)
        if resp.status_code == 200:
            self.token = resp.json().get("access_token")
        else:
            self.token = None
            raise RuntimeError(f"Amadeus auth failed: {resp.status_code} {resp.text}")

    def get(self, path, params=None, timeout=25):
        if not self.token:
            self._auth()
        url = f"{AMADEUS_BASE}{path}"
        headers = {"Authorization": f"Bearer {self.token}"}
        resp = requests.get(url, headers=headers, params=params or {}, timeout=timeout)

        # token expired -> refresh once
        if resp.status_code == 401:
            self._auth()
            headers = {"Authorization": f"Bearer {self.token}"}
            resp = requests.get(url, headers=headers, params=params or {}, timeout=timeout)

        return resp


# =============================================================================
# FLIGHT FETCHER
# =============================================================================
class FlightFetcher:
    def __init__(self, client: AmadeusClient, currency=DEFAULT_CURRENCY):
        self.client = client
        self.currency = currency

    def get_price_one_day(self, origin, destination, date_str, max_results=5):
        """Return the lowest price for a given date, or None."""
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": date_str,
            "adults": 1,
            "currencyCode": self.currency,
            "max": max_results,
        }
        try:
            resp = self.client.get("/v2/shopping/flight-offers", params=params, timeout=30)
        except Exception:
            return None

        if resp.status_code != 200:
            return None

        data = resp.json().get("data", [])
        prices = []
        for offer in data:
            p = offer.get("price", {}).get("total", None)
            if p is None:
                continue
            try:
                prices.append(float(p))
            except Exception:
                pass
        return min(prices) if prices else None

    def sample_month_price(self, origin, destination, year, month, sample_days=(5, 15, 25)):
        """Sample a few days in the month -> min/avg price stats."""
        last_day = monthrange(year, month)[1]
        prices = []

        for d in sample_days:
            day = min(d, last_day)
            d_str = date(year, month, day).strftime("%Y-%m-%d")
            price = self.get_price_one_day(origin, destination, d_str)
            if price is not None:
                prices.append(price)
            time.sleep(0.2)

        if not prices:
            return None

        return {
            "min_price": float(min(prices)),
            "avg_price": float(sum(prices) / len(prices)),
            "n_samples": int(len(prices)),
        }

    def fetch_monthly_prices(self, origin, destination, months_ahead=12, start_month=None, status_cb=None):
        """
        Build a monthly price table for future months.
        """
        if start_month and str(start_month).strip():
            start_dt = datetime.strptime(str(start_month).strip(), "%Y-%m").replace(day=1)
        else:
            start_dt = datetime.today().replace(day=1)

        rows = []
        for i in range(months_ahead):
            month_dt = start_dt + relativedelta(months=i)
            y, m = month_dt.year, month_dt.month
            label = month_dt.strftime("%b %Y")

            stats = self.sample_month_price(origin, destination, y, m)
            if stats is None:
                if status_cb:
                    status_cb(f"{label}: no data")
                continue

            rows.append({
                "Year": y,
                "Month": m,
                "Month_Label": label,
                "Price_USD": stats["min_price"],
                "Price_Avg": stats["avg_price"],
                "Price_Samples": stats["n_samples"],
                "Origin": origin,
                "Destination": destination,
            })
            if status_cb:
                status_cb(f"{label}: {stats['n_samples']} sample day(s), min ≈ ${stats['min_price']:.0f}")

        return pd.DataFrame(rows)


# =============================================================================
# COORDINATES (IATA -> lat/lon) via AMADEUS LOCATIONS API
# =============================================================================
def get_airport_geocode(client: AmadeusClient, iata_code: str):
    """
    Uses Amadeus Locations API:
    GET /v1/reference-data/locations?keyword=JFK&subType=AIRPORT
    Returns (lat, lon) or (None, None).
    """
    iata_code = iata_code.strip().upper()
    cache = st.session_state["geo_cache"]
    if iata_code in cache:
        return cache[iata_code]

    params = {
        "keyword": iata_code,
        "subType": "AIRPORT",
        "page[limit]": 10
    }

    try:
        resp = client.get("/v1/reference-data/locations", params=params, timeout=25)
    except Exception:
        cache[iata_code] = (None, None)
        return (None, None)

    if resp.status_code != 200:
        cache[iata_code] = (None, None)
        return (None, None)

    data = resp.json().get("data", [])
    best = None
    for item in data:
        # Prefer exact IATA match
        if item.get("iataCode", "").upper() == iata_code:
            best = item
            break
    if best is None and data:
        best = data[0]

    if not best:
        cache[iata_code] = (None, None)
        return (None, None)

    geo = best.get("geoCode", {})
    lat = geo.get("latitude", None)
    lon = geo.get("longitude", None)

    cache[iata_code] = (lat, lon)
    return (lat, lon)


# =============================================================================
# WEATHER (Open-Meteo) -> monthly climate stats
# =============================================================================
def fetch_monthly_climate(lat: float, lon: float, year: int):
    """
    Fetch ONE year of daily weather, then aggregate by month.
    Returns df with columns:
    Month, avg_max_temp, avg_min_temp, total_precip, avg_wind_speed, weather_type
    """
    cache_key = (round(float(lat), 4), round(float(lon), 4), int(year))
    climate_cache = st.session_state["climate_cache"]
    if cache_key in climate_cache:
        return climate_cache[cache_key].copy()

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
    except Exception:
        df_empty = pd.DataFrame(columns=["Month", "avg_max_temp", "avg_min_temp", "total_precip", "avg_wind_speed", "weather_type"])
        climate_cache[cache_key] = df_empty
        return df_empty

    if resp.status_code != 200:
        df_empty = pd.DataFrame(columns=["Month", "avg_max_temp", "avg_min_temp", "total_precip", "avg_wind_speed", "weather_type"])
        climate_cache[cache_key] = df_empty
        return df_empty

    daily = resp.json().get("daily", {})
    if not daily or "time" not in daily:
        df_empty = pd.DataFrame(columns=["Month", "avg_max_temp", "avg_min_temp", "total_precip", "avg_wind_speed", "weather_type"])
        climate_cache[cache_key] = df_empty
        return df_empty

    df_daily = pd.DataFrame(daily)
    df_daily["time"] = pd.to_datetime(df_daily["time"])
    df_daily["Month"] = df_daily["time"].dt.month

    g = df_daily.groupby("Month", as_index=False).agg(
        avg_max_temp=("temperature_2m_max", "mean"),
        avg_min_temp=("temperature_2m_min", "mean"),
        total_precip=("precipitation_sum", "sum"),
        avg_wind_speed=("windspeed_10m_max", "mean"),
    )
    g["weather_type"] = "historical"

    climate_cache[cache_key] = g.copy()
    return g


def add_weather_to_flights(df_prices: pd.DataFrame, client: AmadeusClient, status_cb=None):
    """
    For each destination airport code:
      - get geocode via Amadeus
      - fetch one-year climate via Open-Meteo
      - merge onto df_prices by Month
    """
    if df_prices is None or df_prices.empty:
        return df_prices

    climate_year = date.today().year - 1  # last complete-ish year

    destinations = sorted(df_prices["Destination"].unique().tolist())
    climate_rows = []

    for dest in destinations:
        lat, lon = get_airport_geocode(client, dest)
        if status_cb:
            status_cb(f"{dest}: resolving location ... {'ok' if lat is not None else 'failed'}")

        if lat is None or lon is None:
            continue

        df_clim = fetch_monthly_climate(lat, lon, climate_year)
        if df_clim.empty:
            continue

        df_clim = df_clim.copy()
        df_clim["Destination"] = dest
        df_clim["latitude_deg"] = lat
        df_clim["longitude_deg"] = lon
        climate_rows.append(df_clim)

    if not climate_rows:
        df_prices["latitude_deg"] = None
        df_prices["longitude_deg"] = None
        df_prices["avg_max_temp"] = None
        df_prices["avg_min_temp"] = None
        df_prices["total_precip"] = None
        df_prices["avg_wind_speed"] = None
        df_prices["weather_type"] = "missing"
        return df_prices

    df_climate_all = pd.concat(climate_rows, ignore_index=True)
    df_out = df_prices.merge(df_climate_all, on=["Destination", "Month"], how="left")
    return df_out


# =============================================================================
# SCORING
# =============================================================================
def calculate_scores(df: pd.DataFrame, trip_type="general", budget=None, price_weight=0.6):
    """
    Adds comfort_score, price_score, final_score.
    price_weight in [0,1], comfort_weight = 1-price_weight
    """
    df = df.copy()
    try:
        price_weight = float(price_weight)
    except Exception:
        price_weight = 0.6
    price_weight = max(0.0, min(1.0, price_weight))
    comfort_weight = 1.0 - price_weight

    presets = {
        "beach":       {"ideal": 28, "tmin": 22,  "tmax": 35, "rain_pen": 0.30},
        "skiing":      {"ideal": -2, "tmin": -15, "tmax": 8,  "rain_pen": 0.00},
        "sightseeing": {"ideal": 18, "tmin": 8,   "tmax": 30, "rain_pen": 0.50},
        "hiking":      {"ideal": 15, "tmin": 5,   "tmax": 27, "rain_pen": 0.60},
        "general":     {"ideal": 21, "tmin": 5,   "tmax": 35, "rain_pen": 0.50},
    }
    cfg = presets.get(str(trip_type).lower(), presets["general"])

    # ---- comfort score ----
    T = df["avg_max_temp"]
    R = df["total_precip"]
    comfort_scores = []
    for temp, rain in zip(T, R):
        if pd.isna(temp) or pd.isna(rain):
            comfort_scores.append(0.0)
            continue
        if temp < cfg["tmin"] or temp > cfg["tmax"]:
            comfort_scores.append(0.0)
            continue

        score = 100.0 - 2.5 * abs(float(temp) - cfg["ideal"])
        score -= cfg["rain_pen"] * float(rain)
        comfort_scores.append(max(0.0, min(100.0, score)))
    df["comfort_score"] = comfort_scores

    # ---- price score ----
    P = df["Price_USD"].astype(float)
    Pmin, Pmax = P.min(), P.max()
    price_scores = []
    for p in P:
        if budget is not None and budget > 0 and p > budget:
            price_scores.append(0.0)
            continue
        if Pmax > Pmin:
            price_scores.append(100.0 * (Pmax - p) / (Pmax - Pmin))
        else:
            price_scores.append(100.0)
    df["price_score"] = [max(0.0, min(100.0, v)) for v in price_scores]

    df["final_score"] = price_weight * df["price_score"] + comfort_weight * df["comfort_score"]
    df["price_weight_used"] = price_weight
    df["comfort_weight_used"] = comfort_weight
    return df


# =============================================================================
# VISUALS
# =============================================================================
def make_figure(df, origin, destination, trip_type):
    if df is None or df.empty:
        return None

    sns.set_theme(style="whitegrid")
    df_sorted = df.sort_values(["Year", "Month"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    plt.subplots_adjust(wspace=0.35)

    # --- chart 1: price vs comfort over time ---
    ax1 = axes[0]
    x = list(range(len(df_sorted)))
    ax1.bar(x, df_sorted["Price_USD"], alpha=0.75, label="Price (USD)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_sorted["Month_Label"], rotation=45, ha="right")
    ax1.set_ylabel("Price (USD)")
    ax1.set_title(f"Price vs Comfort Over Time\n{origin} → {destination}", fontsize=12, fontweight="bold")

    ax1t = ax1.twinx()
    ax1t.plot(x, df_sorted["comfort_score"], marker="o", linewidth=2.5, label="Comfort Score")
    ax1t.set_ylim(0, 100)
    ax1t.set_ylabel("Comfort Score")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1t.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)

    # --- chart 2: decision matrix ---
    ax2 = axes[1]
    scatter = ax2.scatter(
        df_sorted["comfort_score"],
        df_sorted["Price_USD"],
        s=(df_sorted["final_score"] + 10) * 3,
        c=df_sorted["final_score"],
        cmap="RdYlGn",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.4,
    )
    ax2.set_xlabel("Comfort Score")
    ax2.set_ylabel("Price (USD)")
    ax2.set_title(f"Price vs Comfort Decision Matrix\nTrip Type: {str(trip_type).capitalize()}",
                  fontsize=12, fontweight="bold")

    # annotate top 3 only (avoid overlap)
    top3 = df_sorted.sort_values("final_score", ascending=False).head(3).copy()
    offsets = [(8, 8), (8, -12), (-35, 8)]
    for (i, row), (dx, dy) in zip(top3.iterrows(), offsets):
        ax2.annotate(
            row["Month_Label"],
            (row["comfort_score"], row["Price_USD"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"),
        )

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Final Score")

    ax2.axvline(50, linestyle="--", alpha=0.25, color="gray")
    ax2.axhline(df_sorted["Price_USD"].median(), linestyle="--", alpha=0.25, color="gray")

    # --- chart 3: ranking ---
    ax3 = axes[2]
    order = df_sorted["final_score"].values
    colors = plt.cm.RdYlGn(order / 100.0)
    ax3.barh(range(len(df_sorted)), df_sorted["final_score"], color=colors, edgecolor="black", linewidth=0.4)
    ax3.set_yticks(range(len(df_sorted)))
    ax3.set_yticklabels(df_sorted["Month_Label"])
    ax3.set_xlim(0, 100)
    ax3.set_xlabel("Final Score")
    ax3.set_title("Month Rankings by Final Score", fontsize=12, fontweight="bold")
    ax3.grid(axis="x", alpha=0.25)

    for i, v in enumerate(df_sorted["final_score"].values):
        ax3.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


# =============================================================================
# PIPELINES (with progress)
# =============================================================================
def run_single(origin, dest, trip_type, start_month, months_ahead, budget, price_weight):
    status = st.empty()
    progress = st.progress(0)

    try:
        status.info("Step 1/4: Authenticating ...")
        client = AmadeusClient(AMADEUS_API_KEY, AMADEUS_API_SECRET)
        progress.progress(15)

        status.info("Step 2/4: Fetching monthly flight prices ...")
        ff = FlightFetcher(client)

        log_lines = st.empty()
        logs = []
        def log(msg):
            logs.append(msg)
            if len(logs) > 12:
                logs.pop(0)
            log_lines.write("\n".join([f"- {x}" for x in logs]))

        df_prices = ff.fetch_monthly_prices(origin, dest, months_ahead, start_month, status_cb=log)
        if df_prices.empty:
            status.error("No flight price data found for this route.")
            progress.empty()
            return None
        progress.progress(55)

        status.info("Step 3/4: Fetching weather data ...")
        df_full = add_weather_to_flights(df_prices, client)
        progress.progress(80)

        status.info("Step 4/4: Scoring and building results ...")
        df_scored = calculate_scores(df_full, trip_type=trip_type, budget=budget, price_weight=price_weight)
        progress.progress(100)

        status.success("Done.")
        time.sleep(0.3)
        progress.empty()
        status.empty()
        return df_scored

    except Exception as e:
        status.error(f"Error: {e}")
        progress.empty()
        return None


def run_multi(origin, dest_list, trip_type, start_month, months_ahead, budget, price_weight):
    status = st.empty()
    progress = st.progress(0)

    try:
        status.info("Step 1/3: Authenticating ...")
        client = AmadeusClient(AMADEUS_API_KEY, AMADEUS_API_SECRET)
        ff = FlightFetcher(client)
        progress.progress(15)

        all_results = {}
        summary_rows = []
        total = len(dest_list)

        for idx, dest in enumerate(dest_list, start=1):
            status.info(f"Step 2/3: Route {idx}/{total} — {origin} → {dest}")
            df_prices = ff.fetch_monthly_prices(origin, dest, months_ahead, start_month)
            if df_prices.empty:
                continue

            df_full = add_weather_to_flights(df_prices, client)
            df_scored = calculate_scores(df_full, trip_type=trip_type, budget=budget, price_weight=price_weight)
            if df_scored.empty:
                continue

            all_results[dest] = df_scored
            best = df_scored.sort_values("final_score", ascending=False).iloc[0]
            summary_rows.append({
                "Destination": dest,
                "Best_Month": best["Month_Label"],
                "Best_Price_USD": float(best["Price_USD"]),
                "Best_Comfort": float(best["comfort_score"]),
                "Best_Final_Score": float(best["final_score"]),
            })

            progress.progress(min(15 + int(80 * idx / max(total, 1)), 95))

        if not summary_rows:
            status.error("No data collected for any destination.")
            progress.empty()
            return None, None

        status.info("Step 3/3: Finalizing comparison ...")
        df_summary = pd.DataFrame(summary_rows).sort_values("Best_Final_Score", ascending=False).reset_index(drop=True)
        progress.progress(100)
        status.success("Done.")
        time.sleep(0.3)
        progress.empty()
        status.empty()

        return df_summary, all_results

    except Exception as e:
        status.error(f"Error: {e}")
        progress.empty()
        return None, None


# =============================================================================
# STREAMLIT APP
# =============================================================================
def main():
    st.set_page_config(page_title="Destination Price & Weather Optimizer", layout="wide")
    init_state()

    st.title("Destination Price & Weather Optimizer")
    st.write("Compare future **flight prices** with historical **weather comfort** to recommend the best travel months.")
    st.markdown("---")

    # Inputs
    left, right = st.columns([1.2, 2.2])

    with left:
        st.subheader("Trip Settings")
        mode = st.radio("Mode", ["Single destination", "Multi-destination comparison"], index=0)

        origin = st.text_input("Origin airport code (IATA)", value="JFK").strip().upper()
        trip_type = st.selectbox("Trip type", ["general", "beach", "skiing", "sightseeing", "hiking"], index=0)

        start_month = st.text_input("Start month (YYYY-MM, blank = current month)", value="")
        start_month = start_month.strip() or None

        months_ahead = st.slider("Months ahead to analyze", 3, 18, 12)

        budget_str = st.text_input("Budget in USD (blank = no limit)", value="")
        budget = None
        if budget_str.strip():
            try:
                budget = float(budget_str.strip())
            except Exception:
                st.warning("Budget input is invalid. Using no limit.")
                budget = None

        price_weight = st.slider("Price importance vs comfort", 0.0, 1.0, 0.6, 0.05)
        st.caption(f"Current weights → Price: {price_weight:.2f}, Comfort: {1 - price_weight:.2f}")

        st.markdown("")

        clear = st.button("Clear saved results")
        if clear:
            st.session_state["single_scored"] = None
            st.session_state["single_config"] = None
            st.session_state["multi_summary"] = None
            st.session_state["multi_results"] = None
            st.session_state["multi_config"] = None
            st.session_state["detail_dest"] = "Select a destination"
            st.success("Cleared.")

    with right:
        if mode == "Single destination":
            st.subheader("Single destination analysis")
            dest = st.text_input("Destination airport code (IATA)", value="LHR").strip().upper()

            run_btn = st.button("Run analysis", type="primary", key="run_single_btn")

            if run_btn:
                config = {
                    "mode": "single",
                    "origin": origin,
                    "dest": dest,
                    "trip_type": trip_type,
                    "start_month": start_month,
                    "months_ahead": months_ahead,
                    "budget": budget,
                    "price_weight": price_weight,
                }
                df_scored = run_single(origin, dest, trip_type, start_month, months_ahead, budget, price_weight)
                st.session_state["single_scored"] = df_scored
                st.session_state["single_config"] = config

            df_scored = st.session_state.get("single_scored", None)
            if df_scored is not None and not df_scored.empty:
                df_sorted = df_scored.sort_values("final_score", ascending=False).reset_index(drop=True)
                best = df_sorted.iloc[0]

                c1, c2, c3 = st.columns(3)
                c1.metric("Best month", best["Month_Label"])
                c2.metric("Estimated lowest fare", f"${best['Price_USD']:.0f}")
                c3.metric("Final score", f"{best['final_score']:.1f} / 100")

                st.markdown("### Visual summary")
                fig = make_figure(df_scored, origin, dest, trip_type)
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)

                st.markdown("### Top 3 recommended months")
                cols = ["Month_Label", "Price_USD", "avg_max_temp", "total_precip", "comfort_score", "price_score", "final_score"]
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
            st.subheader("Multi-destination comparison")
            dest_str = st.text_input("Destination airports (comma-separated IATA codes)", value="LHR, CDG, DEN")
            dest_list = [d.strip().upper() for d in dest_str.split(",") if d.strip()]

            run_btn = st.button("Run comparison", type="primary", key="run_multi_btn")

            if run_btn:
                config = {
                    "mode": "multi",
                    "origin": origin,
                    "dest_list": dest_list,
                    "trip_type": trip_type,
                    "start_month": start_month,
                    "months_ahead": months_ahead,
                    "budget": budget,
                    "price_weight": price_weight,
                }
                df_summary, all_results = run_multi(origin, dest_list, trip_type, start_month, months_ahead, budget, price_weight)
                st.session_state["multi_summary"] = df_summary
                st.session_state["multi_results"] = all_results
                st.session_state["multi_config"] = config

                # Reset detail selector after a new run
                st.session_state["detail_dest"] = "Select a destination"

            # IMPORTANT: Display from session_state so selectbox reruns won't wipe results
            df_summary = st.session_state.get("multi_summary", None)
            all_results = st.session_state.get("multi_results", None)

            if df_summary is not None and all_results:
                st.markdown("### Summary (best month per destination)")
                st.dataframe(
                    df_summary.style.format({
                        "Best_Price_USD": "{:.0f}",
                        "Best_Comfort": "{:.1f}",
                        "Best_Final_Score": "{:.1f}"
                    }),
                    use_container_width=True
                )

                st.markdown("### Detail view")
                options = ["Select a destination"] + df_summary["Destination"].tolist()

                # Keep selection valid across reruns
                if st.session_state["detail_dest"] not in options:
                    st.session_state["detail_dest"] = "Select a destination"

                chosen = st.selectbox("Pick a destination for charts", options, key="detail_dest")

                if chosen != "Select a destination" and chosen in all_results:
                    df_chosen = all_results[chosen].sort_values("final_score", ascending=False).reset_index(drop=True)
                    best = df_chosen.iloc[0]

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Best month", best["Month_Label"])
                    c2.metric("Estimated lowest fare", f"${best['Price_USD']:.0f}")
                    c3.metric("Final score", f"{best['final_score']:.1f} / 100")

                    fig = make_figure(df_chosen, origin, chosen, trip_type)
                    if fig is not None:
                        st.pyplot(fig, use_container_width=True)

                    with st.expander("Full monthly table for this route"):
                        st.dataframe(df_chosen, use_container_width=True)

                    csv = df_chosen.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"Download CSV for {origin}→{chosen}",
                        csv,
                        file_name=f"travel_analysis_{origin}_{chosen}.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Run a comparison to see results here.")


if __name__ == "__main__":
    main()
