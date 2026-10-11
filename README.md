# Climate Intelligence — Heatwave Analytics Dashboard

A runnable Streamlit dashboard for exploring temperature patterns in Mumbai,
Delhi, Pune, Ahmedabad, Jaipur and Nagpur. Downloads real Open-Meteo data,
validates it, visualizes daily/monthly/seasonal patterns, compares cities, and
trains three direct Random Forest temperature models with held-out evaluation.
No synthetic weather, hardcoded metrics or fallback predictions are used.

**Official IMD grids:** choose **IMD gridded temperature** in the sidebar for
validated annual 1° GRD import, recent daily 0.5° GRD + CTL import, geographic
heatmaps, CSV/Parquet exports, and separate historical grid-cell analytics/ML.
The supplied official **2025 GRD has been validated and cached locally**: 365 days,
129,552 valid readings, 5.3067–46.1526 °C. Select the imported 2025 dataset to
explore it, including nearest available 1° cells and coverage-aware statistics.
See the [actual-file validation record](docs/IMD_2025_VALIDATION.md).
Automatic IMD downloading remains unverified; local files use the manual workflow.
See [IMD format, import instructions and scientific limitations](docs/IMD.md).

To recreate the IMD cache from the supplied file (no network needed):

```powershell
.\.venv\Scripts\python -m climate.imd_import data/Maxtemp_MaxT_2025.GRD --year 2025 --byte-order little
```

If stored under `data/raw/imd/`, substitute that path. Raw and processed data are
excluded from Git, so other checkouts must obtain the official file and import it.

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

The first load automatically downloads **three calendar years for all six supported
cities**, then evaluates and trains the selected city's local models. It needs an
internet connection; the public noncommercial API needs no key. Choose cities,
dates, an exploratory threshold and smoothing settings in the sidebar, then
select **Apply filters**. Chart filters do not shorten or otherwise change the
automatic three-year training dataset. Both data workspaces use four sections:

- **Overview:** current Open-Meteo estimates and historical summary, or coverage
  and extremes of the selected imported IMD file.
- **Historical Analysis:** daily trends, monthly/seasonal patterns and city
  comparisons for Open-Meteo; geographic date maps and 1° cell analysis for IMD.
- **ML Forecasting:** upcoming Open-Meteo-based predictions when history is fresh,
  or explicitly labeled IMD historical endpoint experiments.
- **Data Processing:** actual record/missing/quality counts, processing steps,
  source interpretation, detailed reports, exports and IMD imports.

Each major chart includes units, source, date coverage, a calculated observation
and an optional **How to read this chart** guide. Technical methodology remains
available in expanders. The current conditions section includes a separate
expandable provider forecast. Dates use Asia/Kolkata. See the
[dashboard reading guide](docs/DASHBOARD_GUIDE.md) for interpretation and limits.
Requests are limited to ten years each to bound download and training costs.

**Refresh weather data** immediately bypasses the on-disk caches for current
conditions, the primary/comparison chart ranges and all six training datasets, and clears
trained-model results. You do not need to download CSVs or supply training files.
Current conditions and provider forecasts update automatically every fifteen
minutes while the dashboard is open. Automatic training acquisition is checked
every six hours; its end date always follows the latest completed local day.
These are Streamlit session timers, not an operating-system scheduled service:
closing the dashboard stops timed refreshes; reopening it checks the caches.

## What the data represents

- **Historical:** Open-Meteo Historical Weather API reanalysis/model estimates,
  not direct station measurements. All completed days, including recent days,
  are requested exclusively from `/v1/archive` using its default Best Match
  product. No past weather is requested from the Forecast API. If recent archive
  values are unavailable, they remain missing and the model explains why it
  cannot predict. There is no substituted forecast-history bridge.
- **Current temperature/humidity:** current weather model estimates, with the
  provider timestamp displayed. Null or unavailable values show `Unavailable`.
- **Provider forecast:** Open-Meteo's next four calendar days, displayed as a
  separate product. These values are never local ML model features or outputs.
- **Local ML prediction:** three horizon-specific Random Forest regressors
  trained here, using daily maxima from the automatically acquired three-year
  history for the primary city, independently of chart filters.

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

Each horizon requires at least 120 complete rows. Automatic acquisition normally
provides three years; missing values can reduce usable rows. Models use chronological
70/15/15 train/validation/test blocks, with labels
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

Predicted dates follow the **automatic training history endpoint**, which is
yesterday, independently of chart date filters. Horizons 1–3 therefore refer to
today and the following two days. Missing recent historical values prevent
inference; older values are never passed off as current forecasts.
Models have no exogenous weather inputs, spatial
features, calibrated uncertainty bands or operational warning capability. Their
accuracy can be worse than persistence, especially during abrupt weather changes.

## Cache and exports

`data/raw/` stores ingested API tables as Parquet; `data/processed/` stores separate
quality-controlled tables. File writes use atomic replacement. Historical request
caches expire after six hours, current/provider forecasts after fifteen minutes.
Current conditions and forecasts have their own raw and processed Parquet tables
under `data/raw/current/` and `data/processed/current/`. Temperatures and humidity
are validated, and current timestamps are normalized to the location timezone.
Cache keys include the provider identity and schema version; older mixed-source
history caches cannot be reused as archive-only training data.
Expired data is not silently served on network failure. Failed requests retry
transient server errors and rate limits three times with bounded backoff and
timeouts; errors remain visible. Cached successful history can be used offline
until expiry. Cache write failures are logged.

Trained model results are cached in memory by historical dataset for six hours,
with a bounded entry count. No model binaries are loaded from disk. Restarting
the server clears in-memory caches. Use **Refresh weather data** to force fresh
downloads and model retraining without deleting files or restarting the server.
Acquisition continues across all cities if individual requests fail. The ML tab
shows each city's available maximum-temperature count and explicit download errors;
partial failures can be retried with the refresh button. Failed refresh requests
leave existing files intact but do not silently return them as fresh data.

The Data Processing section downloads both raw and processed CSV tables. Exports retain city,
source, retrieval timestamp and timezone offsets; missing values remain blank.
Generated datasets, logs, virtual environments, secrets and model artifacts are
excluded from Git. No remote is configured by this project.

## Validation

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m climate.check --all-cities --refresh
```

The default tests are offline and cover processing, timezone aggregation,
smoothing gaps, causal features, purged splits, metrics, prediction failures,
API parsing/errors, provider-isolated caching, cache expiry, manual refresh,
multi-city acquisition failures, chart serialization, CSV exports and Streamlit
rendering/filter submission. Synthetic values exist only in explicitly labelled
test fixtures, patched into offline tests; application paths always use real data.

The diagnostic command bypasses caches with `--refresh`, makes real network
requests for every supported city and prints data quality, historical sources,
provider forecasts, local predictions and evaluation metrics. It returns a
nonzero exit status if ingestion or training fails. Live API access is deliberately
separate from deterministic tests. Server health is available at
http://localhost:8501/_stcore/health.

For a smaller first download or network troubleshooting, run:

```powershell
.\.venv\Scripts\python -m climate.check --city Delhi --years 3 --refresh
```

Use `--years 4` or `--years 5` for longer CLI acquisition, or omit `--refresh` to
reuse valid local caches. `--days` remains available for a custom single-city
diagnostic; it does not change the dashboard's automatic three-year training range.
If API access is blocked, run this command locally with internet access, allowing
HTTPS to `archive-api.open-meteo.com` and `api.open-meteo.com`. No dataset import or
API key is required. API failures are reported explicitly rather than counted as
live verification.

## Adding another provider

IMD has a dedicated grid ingestion path (`imd_format`, `imd_parser`, `imd_store`,
`imd_charts`, `imd_ui`). It reuses analytics/ML only after extracting an explicitly
selected historical grid cell. It is not treated as an Open-Meteo point adapter;
its grids and forecasts are never blended with point estimates. The following
contract is for additional compatible point-weather providers.

`climate.sources.WeatherSource` defines the provider contract. Implement a unique,
versioned `cache_id`, `history(city, start, end)`, `current(city)` and `close()`.
Historical tables use timezone-aware `date`, `city`, `tmax`, `tmin`, `tmean`,
`source` and preferably `retrieved_at`; temperatures must use Celsius. Current
values use `time`, `temperature_2m`, `relative_humidity_2m` and an explicit `source`.
Return the current mapping together with a daily forecast table. Null values stay
missing. Label direct observations and modeled estimates accurately.

Pass an adapter to `load_history(..., source=adapter)` or
`load_current(..., source=adapter)` to reuse validation and isolated local caches.
Supplied adapters remain caller-owned and should be closed by the caller.
`OpenMeteoSource` is the working automatic point adapter. IMD manual gridded
support is described above; NASA POWER remains an unimplemented extension target.

## Project layout

```text
app.py                  Streamlit entry point
climate/
  api.py                Retrying HTTP transport and explicit errors
  locations.py          City coordinates and timezones
  weather.py            API retrieval, parsing and provenance
  sources.py            Provider contract and Open-Meteo adapter
  acquisition.py        Automatic three-year acquisition for all cities
  imd_format.py         Official grid contracts and CTL validation
  imd_parser.py         Strict NumPy GRD decoding and quality checks
  imd_store.py          Separate raw IMD files, Parquet and metadata
  imd_charts.py         Geographic grid heatmaps
  imd_ui.py             Manual import, maps and isolated cell analytics/ML
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
  forecast_panel.py     Local ML forecasting and held-out evaluation
  export.py             CSV serialization
  style.py              Compact neutral styling
  check.py              Live-data diagnostic CLI
tests/                  Offline unit and dashboard integration tests
.streamlit/config.toml  Theme and local server defaults
```
