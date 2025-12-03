# Destination Price & Weather Optimizer

This project is a small decision tool built with Python and Streamlit.  
It helps travelers decide which month is the best time to visit a destination by combining:

- sampled flight prices from the Amadeus API
- historical weather data from the Open‑Meteo Archive API

The app analyzes future months for a given route and ranks them based on both cost and weather comfort.

---

## Demo

Deployed app (Streamlit Cloud):

https://naqtvtz64xsotb7rwrtzxu.streamlit.app/

The online version runs the same logic as app.py in this repository.

---

## Project Overview

Goal

Find the best travel month for a route by balancing:

- how cheap the flights are, and
- how comfortable the weather is

What the app does

For a chosen origin and one or more destinations, the app:

1. Samples one‑way flight prices for each month in a future window (for example, the next 12 months).
2. Requests historical daily weather data for each destination (most recent complete year), then aggregates it into monthly statistics.
3. Computes three scores per month:
   - Price score: lower prices mean higher score
   - Comfort score: temperatures near an ideal range with less rain mean higher score
   - Final score: a weighted combination of price and comfort (user controlled)
4. Ranks the months and shows:
   - a best month summary
   - a top‑3 table
   - a full monthly table
   - three charts: price vs time, price vs comfort, and final score ranking

The app supports:

- Single destination mode (one origin, one destination)
- Multi‑destination mode (one origin, multiple destinations)

---

## Data Sources & Citations

All data are obtained programmatically via APIs. There are no manual downloads or local CSV files.

1. Amadeus for Developers – Self‑Service APIs  
   - Flight Offers Search API: samples flight offers to estimate monthly prices  
   - Reference Data / Locations API: maps IATA airport codes to latitude/longitude for weather queries  
   - Site: https://developers.amadeus.com/

2. Open‑Meteo – Historical Weather API (Archive)  
   - Used to retrieve daily weather and aggregate it into monthly temperature and precipitation statistics  
   - Site: https://open-meteo.com/

---

## Project Structure

- app.py  
  Main Streamlit application (API calls, scoring, visualizations, and UI)

- README.md  
  Project summary, instructions, and data source citations

---

## Installation

Requirements

- Python 3.9+
- Internet connection (data are fetched from online APIs at runtime)

Install packages:

```bash
pip install streamlit requests pandas matplotlib seaborn python-dateutil
````

---

## How to Run

Local

1. Put app.py and README.md in the same folder
2. Install dependencies (see above)
3. Run:

```bash
streamlit run app.py
```

Open the URL printed in the terminal (typically [http://localhost:8501](http://localhost:8501)).

Google Colab (optional)

During development, the app was also tested in Google Colab by installing the same packages and running Streamlit from the notebook environment.

---

## Usage

Single destination mode

1. Choose Single destination
2. Enter:

   * origin airport (IATA code, e.g. JFK)
   * destination airport (IATA code, e.g. LHR)
   * start month (YYYY-MM) or leave blank for current month
   * number of months ahead to analyze
   * trip type (general, beach, skiing, sightseeing, hiking)
   * optional budget in USD
   * slider for price vs comfort weight
3. Click Run analysis

Multi‑destination mode

1. Choose Multi‑destination comparison
2. Enter:

   * origin airport
   * destination airports as comma‑separated IATA codes
   * other settings as above
3. Click Run comparison
4. Use the detail view to select a destination and view charts/tables

---

## Limitations

* Some origin–destination pairs or dates may not return flight offers.
* Flight prices are sampled snapshots and may not match real booking prices.
* Weather comfort scores are based on historical averages, not real‑time forecasts.
* Only one‑way ticket prices are considered; hotel costs, trip length, and round‑trip fares are not included.
