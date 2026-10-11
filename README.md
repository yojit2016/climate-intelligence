# Climate Intelligence — Heatwave Analytics Dashboard

A runnable Streamlit dashboard for exploring temperature patterns in Mumbai,
Delhi, Pune, Ahmedabad, Jaipur and Nagpur. Downloads real Open-Meteo data,
validates it, visualizes daily/monthly/seasonal patterns, compares cities, and
trains three direct Random Forest temperature models with held-out evaluation.
No synthetic weather, hardcoded metrics or fallback predictions are used.

## Run locally

Python 3.11+ is required (verified on Windows with Python 3.13).
Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

Open http://localhost:8501. Stop the foreground server with Ctrl+C. Activation
is optional: the commands invoke the virtual environment directly, avoiding
PowerShell execution-policy changes.

On macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

The first load downloads one year for Delhi and the default comparison cities
(Mumbai and Pune), then evaluates and trains the local models. It needs an
internet connection; the public noncommercial API needs no key. Choose cities,
dates, an exploratory threshold and smoothing settings in the sidebar, then
select **Apply filters**. Use the five tabs for trends, seasonal patterns,
regional comparisons, ML predictions and CSV downloads. Dates use Asia/Kolkata.
Requests are limited to ten years each to bound download and training costs.

## What the data represents

- **Historical:** Open-Meteo Historical Weather API reanalysis/model estimates,
  not direct station measurements. The archive is used through five days ago.
- **Recent completed days:** the Forecast API's past-days weather model output
  bridges the archive's reporting delay. These records carry their own source
  label. Mixing products can introduce differences; source counts are visible.
- **Current temperature/humidity:** current weather model estimates, with the
  provider timestamp displayed. Null or unavailable values show `Unavailable`.
- **Provider forecast:** Open-Meteo's next four calendar days, displayed as a
  separate product. These values are never local ML model features or outputs.
- **Local ML prediction:** three horizon-specific Random Forest regressors
  trained here, using daily maxima from the chosen historical range.

An arbitrary temperature threshold is **exploratory**, not an official heatwave
warning. The app does not implement an IMD heatwave classification. Its local
ML evaluation targets are historical reanalysis/model estimates, not independent
station observations. No measurement feed is currently ingested.

Sources: [Forecast API documentation](https://open-meteo.com/en/docs),
[Historical API documentation](https://open-meteo.com/en/docs/historical-weather-api).
Weather data is attributed to [Open-Meteo](https://open-meteo.com/) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The free API is for
noncommercial use under the provider's [terms](https://open-meteo.com/en/terms);
check those terms before commercial deployment.

## Processing and analytics

Raw API tables and processed tables are independent. Processing converts dates
to location time, removes duplicate city/day records (last record wins), coerces
non-numeric values to missing, and marks nonfinite values or temperatures outside
−90 to 65 °C invalid. Inconsistent minimum/mean/maximum rows become missing.
These bounds are broad sanity checks, not local climate thresholds. Missing
calendar dates appear explicitly with blank temperatures; no values are imputed.
The data quality expander reports counts and source labels.

Optional centred Savitzky–Golay smoothing uses polynomial order two on contiguous
valid daily runs. It never bridges gaps and leaves short runs blank. It is
descriptive and uses neighbouring values on both sides; it is **never used for ML**.
An hourly aggregation utility also computes daily maximum, minimum, mean and
valid-hour counts in location time, flagging incomplete days. The dashboard's
default API path requests native daily values rather than hourly downloads.

Charts include daily extremes, a seven-calendar-day rolling average (at least
four valid values), threshold exceedances, monthly aggregates, seasonal
aggregates and regional daily comparisons. Monthly and seasonal results cover
the selected dates only; partial periods are not climatological normals.
Cross-city summaries show valid-day counts to reveal incomplete coverage.

## Forecasting and evaluation

For an origin `t` after a completed local day, each model predicts `t+h` for
`h=1,2,3`. Features contain the latest daily maximum, lags 1/2/3/7, trailing
7/14-day means and the target date's month and cyclical day of year. Features use
no temperatures after `t`. Calendar gaps stay missing; feature/target rows with
missing values are excluded. The latest fourteen days must be complete to infer.

Each horizon requires at least 120 complete rows (select roughly six months or
more). Models use chronological 70/15/15 train/validation/test blocks, with labels
crossing block boundaries removed. Validation MAE selects a maximum tree depth
of six or unrestricted. Forests have 160 trees, a minimum leaf size of three and
a fixed random seed. The selected settings are refit on pre-test labels only
before scoring the test block. Test evaluation uses rolling origins with observed
past inputs and no test-period retraining. A persistence baseline repeats the
maximum at each origin; MAE/RMSE for both models and each horizon appear together.

Separate production models are then refit on all available labels. Their
predictions are not used to compute the reported historical test errors. The
dashboard explicitly warns when Random Forest fails to improve baseline MAE.
It also explains insufficient data and training failures instead of inventing
predictions.

Predicted dates follow the **selected range endpoint**, not necessarily today.
With yesterday as the endpoint, horizons 1–3 refer to today and the following two
days. Historical endpoints produce predictions following that historical date,
with a prominent warning. Models have no exogenous weather inputs, spatial
features, calibrated uncertainty bands or operational warning capability. Their
accuracy can be worse than persistence, especially during abrupt weather changes.

## Cache and exports

`data/raw/` stores ingested API tables as Parquet; `data/processed/` stores separate
quality-controlled tables. File writes use atomic replacement. Historical request
caches expire after six hours, current/provider forecasts after fifteen minutes.
Expired data is not silently served on network failure. Failed requests retry
transient server errors and rate limits three times with bounded backoff and
timeouts; errors remain visible. Cached successful history can be used offline
until expiry. Cache write failures are logged.

Trained model results are cached in memory by historical dataset for six hours,
with a bounded entry count. No model binaries are loaded from disk. Restarting
the server clears in-memory caches. To force a full refresh, stop the server,
remove the generated `data/` directory and restart.

The Data tab downloads both raw and processed CSV tables. Exports retain city,
source, retrieval timestamp and timezone offsets; missing values remain blank.
Generated datasets, logs, virtual environments, secrets and model artifacts are
excluded from Git. No remote is configured by this project.

## Validation

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m climate.check --city Delhi --days 365
```

The default tests are offline and cover processing, timezone aggregation,
smoothing gaps, causal features, purged splits, metrics, prediction failures,
API parsing/errors, cache expiry, chart serialization, CSV exports and Streamlit
rendering/filter submission. Synthetic values exist only in explicitly labelled
test fixtures, patched into offline tests; application paths always use real data.

The diagnostic command makes real network requests and prints data quality,
provider forecasts, local predictions and evaluation metrics. It returns a
nonzero exit status if ingestion or training fails. Live API access is deliberately
separate from deterministic tests. Server health is available at
http://localhost:8501/_stcore/health.

## Project layout

```text
app.py                  Streamlit entry point
climate/
  api.py                Retrying HTTP transport and explicit errors
  locations.py          City coordinates and timezones
  weather.py            API retrieval, parsing and provenance
  cache.py              Atomic Parquet caching
  processing.py         Quality controls and hourly aggregation
  smoothing.py          Gap-aware signal filter
  service.py            Raw/processed ingestion pipeline
  analytics.py          Statistics and seasonal/regional summaries
  features.py           Leakage-safe forecasting features
  evaluation.py         Chronological splits, baseline and metrics
  model.py              Direct model training and prediction
  charts.py             Plotly figures
  ui.py                 Filters, cards, caching and layout
  panels.py             Descriptive analysis panels
  forecast_panel.py     Local ML and separate provider forecast panels
  export.py             CSV serialization
  style.py              Compact neutral styling
  check.py              Live-data diagnostic CLI
tests/                  Offline unit and dashboard integration tests
.streamlit/config.toml  Theme and local server defaults
```
