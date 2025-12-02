# Destination Price & Weather Optimizer

A Python/Streamlit app built as a final project for our data programming course.

The app helps budget travelers decide **which month** is the best time to visit a destination by combining:

- sampled flight prices from the **Amadeus** APIs, and  
- historical weather data from the **Open‑Meteo** Archive API.

All data are obtained via APIs at runtime. We do **not** rely on any pre‑downloaded CSV datasets.

- Live demo (Streamlit Cloud): https://naqtvtz64xsotb7rwrtzxu.streamlit.app/

The project was developed in Google Colab, but anyone can reproduce it locally by running `app.py` with Streamlit and the required Python libraries.

---

## Project Overview

**Goal**

Help a user answer:

> “From my home airport, which month gives the best balance between cheap tickets and comfortable weather for this destination?”

**What the app does**

For a given origin and one or more destination airports, the app:

1. Samples one‑way flight prices for each month in a future window (e.g., the next 12 months) using the Amadeus Flight Offers Search API.  
2. Looks up historical daily weather for each destination and month using the Open‑Meteo Archive API and aggregates it to monthly climate statistics.  
3. Computes three scores for each month:
   - **Price score** – months with lower prices score higher.  
   - **Comfort score** – months closer to an activity‑specific ideal temperature with less rain score higher.  
   - **Final score** – a weighted combination of price and comfort; the user chooses how important price vs comfort should be.  
4. Ranks all months and presents:
   - a “best month” summary,  
   - a table of the top 3 recommended months,  
   - a full monthly table,  
   - three visualizations (price vs time, price vs comfort scatter, and final‑score ranking).

The app supports both **single‑destination analysis** and **multi‑destination comparison**.

---

## Data Sources and Citations

All data are obtained via **APIs** in Python. There are no manual downloads and no local CSV files.

1. **Amadeus for Developers – Self‑Service APIs (Test / Sandbox)**  
   - **Flight Offers Search API**  
     - Used to sample several days per month for each origin–destination route  
     - Provides estimated minimum/average one‑way ticket prices per month  
   - **Locations / Reference Data API**  
     - Used to look up airport metadata, including latitude and longitude, by IATA code  
   - Citation: Amadeus for Developers, “Flight Offers Search API (Test Environment)” and “Locations API”.

2. **Open‑Meteo – Archive API (Historical Weather)**  
   - Used to retrieve historical daily maximum temperature and precipitation for each destination and month.  
   - We aggregate this to:
     - average monthly maximum temperature, and  
     - total monthly precipitation.  
   - Citation: Open‑Meteo, “Archive API for Historical Weather Data”.

These sources satisfy the course requirement for online data sources obtained programmatically, with explicit citations.

---

## Project Structure

- `app.py`  
  Main Streamlit application. Contains:
  - Amadeus API calls for flight prices and airport coordinates
  - Open‑Meteo API calls for historical weather
  - data cleaning and aggregation logic
  - scoring functions for price and comfort
  - visualization code and the Streamlit user interface

- `README.md`  
  This file: project summary, citations, and instructions for running `app.py`.

---

## Installation

### Requirements

- Python 3.9 or later  
- Internet connection (APIs are called live at runtime)

### Python packages

Install the required packages:

```bash
pip install streamlit requests pandas matplotlib seaborn python-dateutil
