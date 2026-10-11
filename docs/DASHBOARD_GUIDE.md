# Reading Climate Intelligence

Choose a **Data workspace** first. Open-Meteo estimates describe individual city
locations; IMD imports describe a regular spatial grid. Their provenance,
processing, caches and model inputs stay separate.

## Four questions, four sections

| Section | Question | What to look for |
| --- | --- | --- |
| Overview | What is happening with temperatures? | Current modeled estimates where available, selected historical coverage, available readings and temperature extremes |
| Historical Analysis | How do temperatures vary over time and space? | Daily charts, seasonal summaries, city comparisons, or date-selectable IMD maps and grid-cell histories |
| ML Forecasting | What does our model predict, and how well did it perform? | Prediction dates, simple baseline, held-out errors and explicit historical-experiment labels |
| Data Processing | Where did the data come from, and what changed? | Actual quality counts, source interpretation, preprocessing, detailed reports, imports and exports |

Use **Explore history** to switch Open-Meteo historical views. Optional threshold,
smoothing and comparison controls are in the sidebar. The IMD map and cell
controls live in Historical Analysis; optional cell modeling lives in ML
Forecasting. When an IMD map uses a recent daily import, historical cell modeling
still requires a separately selected annual 1° dataset.

## Reading the charts

Every major chart has a descriptive title, measurement unit, source name and
displayed date coverage. The observation below it is computed from the displayed
values, not generated as a speculative explanation. **How to read this chart**
adds optional detail without crowding the main view.

Daily maximum means the hottest temperature in a day. The average of daily
maxima describes daytime peaks; it is not the average across all hours. A
monthly or seasonal average covers the selected available dates. Partial months,
missing days and changing geographic coverage can make comparisons uneven.
These summaries alone cannot establish long-term climate change.

Missing readings remain blank. An entirely missing period cannot tell us whether
any threshold was exceeded; the overview reports that result as unavailable.
User thresholds are exploratory and are never labeled official IMD heatwave
criteria or warnings. Optional smoothing is a descriptive filter using neighboring
values and never enters training. The daily trend's separate 7-day rolling mean
is also explicitly labeled.

The IMD map's color scale is fixed within a dataset so dates are comparable.
Blank cells have no valid temperature. The rectangular grid domain is not a
political boundary mask. Nearest-available lookup reports actual cell coordinates,
distance, resolution and valid-date count; it does not estimate weather at the
requested city coordinate. Grid summaries are unweighted available cell-day
statistics, not an area-weighted India average.

## What the forecast means

Random Forest predicts daily maximum temperature one, two and three days after
the last completed day in its input history. Inputs include the latest maximum,
past-day maxima, trailing 7-/14-day averages and the target date's seasonal cycle.
No provider future forecast or chart smoothing is used as a model input.

Open-Meteo's chart range does not shorten its independent automatic three-year
training history. When history ends before yesterday, the dashboard warns that
the prediction dates are historical rather than upcoming. Imported IMD models
are always presented as historical endpoint experiments, with the input endpoint
shown explicitly. The 2025 IMD file is not a live stream.

**Persistence** repeats the most recent daily maximum at each forecast origin.
It is a useful simple baseline, not a second machine-learning model. **MAE** is
the average size of an error in °C. **RMSE** is also in °C but penalizes large
errors more heavily. Lower values are better. The dashboard compares both
measures at each test horizon and reports wins, losses and ties from actual
metrics; it never assumes Random Forest performed better.

The one-day evaluation chart shows held-out targets and predictions. Its reading
guide distinguishes this single horizon from the table of all three horizons.
Chronological splits, purged boundary-crossing labels, validation-based selection
and separate production refitting remain unchanged. Detailed splits and metrics
are available in expanders. Accuracy against modeled/interpolated targets is not
independent validation against station measurements. No calibrated uncertainty
interval or guarantee of future performance is implied.

## Understanding processing counts

Open-Meteo summaries describe the selected historical chart dataset. A record is
a city-day; valid/missing maximum-temperature counts describe the processed daily
calendar. Missing max/min/mean counts are temperature values, not records. Raw
invalid-row counts combine applicable quality flags per row, while duplicate
removal is reported separately. These overlapping/raw-vs-processed categories
must not be added as though they partition the input.

The original processing report's `invalid_values_or_rows` combines value and row
events. It is preserved in detailed metadata and is explicitly not presented as
a distinct-invalid-record count. The summary calculates distinct flagged raw
records from the original acquired table, including unusable dates, missing city
identifiers, nonnumeric/impossible values and inconsistent temperature ordering.

For IMD, a record is a grid cell-day. Valid and missing counts come directly from
the parser's stored metadata. Invalid values cause full-file rejection, so
accepted imports retain zero invalid cells; that does not mean the source has
perfect observational accuracy. IMD performs spatial interpolation in its
source product. This application performs no additional spatial interpolation
and fills no missing temperatures. Raw GRD and processed Parquet remain separate.

## Architecture and verification

The redesign reuses `analytics`, Plotly chart builders, the daily cleaning report,
IMD parser/store metadata, acquisition caches, and existing `ModelResult` outputs.
`storytelling.py` contains pure calculated facts; `presentation.py` renders shared
explanations. UI panels arrange those existing results into the four sections.
Ingestion, feature engineering and ML training/evaluation were not rewritten.

Tests cover actual-data-derived summaries, mixed baseline results and ties,
invalid-event-versus-row counts, enabled/disabled smoothing explanations,
all-missing states, four-section navigation, history subviews, refresh behavior,
source isolation and actual IMD 2025 map/model execution. Streamlit AppTest checks
interactive behavior; visual browser inspection requires a connected browser.
