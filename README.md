# Destination Price & Weather Optimizer

This project is a small decision tool built with Python and Streamlit.  
It helps travelers decide **which month** is the best time to visit a destination by combining:

- sampled flight prices from the Amadeus API  
- historical weather data from the Open‑Meteo Archive API  

The app analyzes future months for a given route and ranks them based on both cost and weather comfort.

---

## Demo

Deployed app (Streamlit Cloud):

> https://naqtvtz64xsotb7rwrtzxu.streamlit.app/

The online version runs the same logic as `app.py` in this repository.

---

## Project Overview

**Goal**

Find the best travel month for a route by balancing:

- how cheap the flights are, and  
- how comfortable the weather is.

**What the app does**

For a chosen origin and one or more destinations, the app:

1. Samples one‑way flight prices for each month in a future window (for example, the next 12 months).  
2. Requests historical daily weather for each destination and month, then aggregates it to monthly statistics.  
3. Computes three scores per month:
   - **Price score** – lower prices → higher score  
   - **Comfort score** – temperatures near an “ideal” range with less rain → higher score  
   - **Final score** – a weighted combination of price and comfort; the user sets the weight  
4. Ranks the months and shows:
   - a “best month” summary  
   - a top‑3 table  
   - a full monthly table  
   - three charts: price vs time, price vs comfort, and final‑score ranking

The app supports:

- **Single destination mode** – one origin, one destination  
- **Multi‑destination mode** – one origin, several destinations

---

## Data Sources & Citations

All data are obtained programmatically via APIs. There are no manual downloads or local CSV files.

1. **Amadeus for Developers – Self‑Service APIs**  
   - Flight Offers Search API: used to sample one‑way flight prices for each origin–destination route and month.  
   - Reference Data / Locations API: used to retrieve airport latitude and longitude from IATA codes.  
   - Site: https://developers.amadeus.com/

2. **Open‑Meteo – Historical Weather API (Archive)**  
   - Used to retrieve daily maximum temperature and precipitation for each destination and month, then aggregated to monthly averages/totals.  
   - Site: https://open-meteo.com/


---

## Project Structure

- `app.py`  
  Main Streamlit application. Contains:
  - API calls (Amadeus and Open‑Meteo)  
  - data cleaning and aggregation  
  - scoring logic for price and comfort  
  - visualizations and the web user interface  

- `README.md`  
  This file: project description, instructions, and data source citations.


---

## Installation

### Requirements

- Python 3.9+  
- Internet connection (data are fetched from online APIs at runtime)

### Python packages

Install the required packages with:

```bash
pip install streamlit requests pandas matplotlib seaborn python-dateutil
````

---

## How to Run

### Local

1. Place `app.py` and `README.md` in a project folder.

2. Open a terminal in that folder.

3. Install dependencies as shown above.

4. Run:

   ```bash
   streamlit run app.py
   ```

5. Open the URL printed in the terminal (typically `http://localhost:8501`).

This will launch the same interface and logic as the online demo.

### Google Colab (optional)

During development, the app was also tested in Google Colab by:

1. Uploading `app.py` to a Colab notebook.
2. Installing the same Python packages with `pip`.
3. Running Streamlit inside Colab and exposing port 8501 via a tunneling tool.

To reproduce the project, running `streamlit run app.py` locally is sufficient.

---

## Usage

### Single destination mode

1. Choose **Single destination**.
2. Enter:

   * origin airport (IATA code, e.g. `JFK`)
   * destination airport (IATA code, e.g. `LHR`)
   * start month (e.g. `2025-12`)
   * number of months ahead to analyze
   * trip type: `general`, `beach`, `skiing`, `sightseeing`, or `hiking`
   * optional budget in USD
   * slider for “price importance vs comfort”
3. Click **Run single‑destination analysis**.
4. Review:

   * the best‑month metrics
   * the three charts
   * the top‑3 table and full monthly table
5. Optionally download the full results as CSV.

### Multi‑destination mode

1. Choose **Multi‑destination comparison**.
2. Enter:

   * origin airport
   * a comma‑separated list of destination airports (IATA codes)
   * other settings as above (start month, months ahead, trip type, budget, slider)
3. Click **Run multi‑destination comparison**.
4. Inspect:

   * the summary table (best month per destination)
   * detailed charts and tables for any selected route
   * optional CSV downloads for each route.

---

## Limitations

* Some origin–destination pairs or dates may not return flight offers.
* Flight prices are sampled snapshots and may not match real booking prices.
* Weather comfort scores are based on historical averages, not real‑time forecasts.
* Only one‑way ticket prices are considered; hotel costs, trip length, and round‑trip fares are not included.

---

## Team

(Replace with your actual names and roles.)

* Member 1 – flight data and Amadeus integration
* Member 2 – weather data and scoring design
* Member 3 – Streamlit UI and visualizations
* Member 4 – documentation and presentation


