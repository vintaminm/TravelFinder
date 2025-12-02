# Destination Price & Weather Optimizer

This project is our Python final project. It is a Streamlit app that helps budget travelers decide **which month** is the best time to visit a destination by balancing flight prices and weather comfort.

The app combines:

- sampled flight prices from the Amadeus API (sandbox)
- historical weather data from the Open‑Meteo Archive API
- airport coordinates from the public OurAirports dataset

The application is implemented as a single Streamlit script: `app.py`.

---

## Demo

Deployed demo (Streamlit Cloud):

> https://naqtvtz64xsotb7rwrtzxu.streamlit.app/

This is the same code as `app.py` in this repository, running on Streamlit’s hosted service.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources & Citations](#data-sources--citations)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Usage](#usage)
- [Limitations](#limitations)
- [Team](#team)

---

## Project Overview

**Goal**

Provide a simple, data‑driven way to answer:

> “From my origin airport, which month offers the best balance between flight price and weather comfort for this destination?”

**What the app does**

For a chosen origin and one or more destinations, the app:

1. Samples one‑way flight prices for several future months (for example, the next 12 months).  
2. Looks up historical weather for the destination in each of those months.  
3. Computes:
   - a **price score** (cheaper months score higher)
   - a **comfort score** (temperature near an activity‑specific ideal range, with less rain)
   - a **final score** that combines price and comfort using a user‑controlled weight  
4. Ranks all months and visualizes the trade‑offs with charts and tables.

The app supports both a single‑destination view and a multi‑destination comparison view.

---

## Data Sources & Citations

All data are obtained programmatically in Python.

1. **Flight prices – Amadeus Self‑Service APIs (Test / Sandbox)**  
   - Product: Flight Offers Search API  
   - Use: sample one‑way flight prices for each route and month  
   - Site: https://developers.amadeus.com/

2. **Weather data – Open‑Meteo Archive API**  
   - Product: Historical Weather API  
   - Use: daily max temperature and precipitation, aggregated to monthly averages/totals  
   - Site: https://open-meteo.com/

3. **Airport coordinates – OurAirports public dataset**  
   - File: `airports.csv` (IATA codes with latitude / longitude)  
   - Use: map destination airport codes to coordinates for the weather API  
   - Site: https://ourairports.com/data/

These citations can also be copied into the final report or slides to satisfy the “Citation of all data sources” requirement.

---

## Project Structure

Key files:

- `app.py`  
  Main Streamlit application. Contains:
  - API calls (Amadeus, Open‑Meteo, OurAirports)
  - data cleaning and aggregation
  - scoring logic (price score, comfort score, final score)
  - visualizations and Streamlit UI

- `README.md`  
  Project summary, run instructions, and data source citations

---

## Installation

### Prerequisites

- Python 3.10 or later  
- Internet connection (all data are fetched from online APIs)

### Python packages

Install required packages:

```bash
pip install streamlit requests pandas matplotlib seaborn python-dateutil
````

The submitted `app.py` includes Amadeus **test** API key and secret at the top of the file, so the app runs in sandbox mode for demonstration.

---

## How to Run

### Local machine

1. Download or clone the project folder containing `app.py` and `README.md`.

2. Install the dependencies as shown above.

3. In a terminal, change into the project folder and run:

   ```bash
   streamlit run app.py
   ```

4. Open the URL shown in the terminal (usually `http://localhost:8501`).


## Usage

### Single destination mode

1. Select **Single destination** at the top of the app.
2. Enter:

   * origin airport (IATA code, e.g., `JFK`)
   * destination airport (IATA code, e.g., `PVG` or `LHR`)
   * start month (e.g., `2025-12`) and number of months to analyze
   * optional budget in USD
   * trip type: `general`, `beach`, `skiing`, `sightseeing`, or `hiking`
   * slider for **Price importance vs comfort**
3. Click **Run single‑destination analysis**.
4. Review:

   * best month summary (month, estimated lowest fare, final score)
   * top 3 recommended months table
   * full monthly table
   * three visualizations (price vs comfort over time, decision scatter, month ranking)
5. Optionally download the full results as a CSV file.

### Multi‑destination comparison mode

1. Select **Multi‑destination comparison**.
2. Enter:

   * origin airport
   * a list of destination airports (comma‑separated IATA codes)
   * other settings as above (start month, horizon, trip type, budget, weights)
3. Click **Run multi‑destination comparison**.
4. View:

   * summary table (for each destination: best month, price, comfort, final score)
   * pick any destination from the selector to see detailed charts and the full monthly table
   * download route‑level CSV files if needed.

---

## Limitations

* Amadeus data uses the **sandbox** environment, so some routes or dates may return no flight offers.
* Flight prices are sampled snapshots and may not exactly match live booking prices.
* Weather comfort scores are based on **historical averages**, not real‑time forecasts.
* Only one‑way flight prices are considered; hotel costs, trip length, and round‑trip fares are not modeled.

---

## Team

* Member 1 – flight data and Amadeus integration
* Member 2 – weather data and scoring design
* Member 3 – Streamlit UI and visualization
* Member 4 – documentation and presentation

