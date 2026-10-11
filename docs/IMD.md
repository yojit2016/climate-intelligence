# Official IMD gridded maximum temperature

The **IMD gridded temperature** workspace is separate from Open-Meteo. It accepts
official maximum-temperature GRD files, validates and converts them, displays
date-selected geographic heatmaps, and evaluates historical grid-cell models.
It does not contain demo weather, generate substitute grids, or merge the two
providers' temperatures.

## Official sources and download investigation

- [Historical yearly 1° maximum temperature](https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html)
- [Recent daily 0.5° maximum temperature](https://www.imdpune.gov.in/cmpg/Realtimedata/max/Max_Download.html)
- [Recent product's linked GrADS control file](https://www.imdpune.gov.in/cmpg/Realtimedata/max/TEMP.CTL)

Investigation on 11 October 2026: indexed official historical documentation
provides grid coordinates, Fortran/C binary-reading examples and a sample GrADS
CTL. The recent page advertises a date-selected binary download, a control file
and a 61×61 grid file. Direct HTTPS requests from the development environment to
both supplied pages and the recent control file timed out. The live form action,
field names, response file and access conditions therefore could not be verified.
No automated downloader, guessed URL pattern or undocumented form POST is used.
No authentication, CAPTCHA or access restriction is bypassed.

Use the official site in your browser and follow its normal download procedure.
If it requires sign-in or another access step, complete that yourself through the
official page. Upload the downloaded GRD through the dashboard or import locally
with the command below. **Automated IMD downloads and a live IMD stream remain
unverified.** The user-supplied official 2025 GRD has now been tested directly,
converted to Parquet and exercised through the dashboard and historical model.
See the [actual-file validation record](IMD_2025_VALIDATION.md).
Unit tests also use explicitly labeled binary fixtures for corruption/edge cases;
those fixtures are never installed as user-visible weather data.

## Supported binary contracts

| Property | Historical annual | Recent daily |
| --- | --- | --- |
| Resolution | 1° | 0.5° |
| Grid | 31 latitudes × 31 longitudes | 61 × 61, matching CTL required |
| Coordinates | Latitude 7.5…37.5°N; longitude 67.5…97.5°E | Matching CTL must confirm the same extent |
| Records | 365 or 366 completed calendar days | One product date |
| Values | Headerless 4-byte floats in °C | Headerless 4-byte floats in °C, validated against CTL layout |
| Missing value | `99.9` | `UNDEF` from the matching CTL |
| Annual size | 1,403,060 bytes; leap year 1,406,904 bytes | 14,884 bytes per date |

The record layout is **day → latitude increasing south to north → longitude
increasing west to east**, with longitude changing fastest. This follows the
historical page's direct-record examples and increasing XDEF/YDEF axes. NumPy
decodes to `(days, latitude, longitude)` in C order. No transposition, flipping,
header stripping, record-marker removal or resolution guessing is performed.

The official historical sample programs read native 4-byte floats; the sample
CTL does not explicitly declare byte order. The app therefore requires the user
to declare **little endian or big endian** rather than claiming the source
documents a universal endianness. If CTL OPTIONS specifies an order, it must
agree with the selection. Confirm the actual file's format before importing it.

The recent control file could not be retrieved here, so a recent daily import
**requires a matching official CTL uploaded alongside the GRD**. It validates
XDEF, YDEF, UNDEF, layout options and TDEF. A historical CTL is optional because
the historical geometry and sentinel are documented directly on the source page.
If supplied, its time axis must also match the chosen year and day count. IMD's
illustrative historical CTL is an example, not a universally matching control
file; its displayed time count/date should not override calendar leap-year rules.

Supported CTL axes are LINEAR, with daily `1DY` time steps, one variable and one
vertical level. Reversed axes, sequential records, file headers, alternate data
types and template/multifile layouts are rejected rather than silently decoded.
DSET paths are never opened or executed. Files requiring an unsupported layout
must be obtained/exported using a documented supported format first.

## Import and validation

1. Start the app, choose **IMD gridded temperature** under **Data workspace**,
   then open **Data Processing**.
2. Select the annual historical or recent daily product.
3. Declare the year/product date and byte order. Annual imports require a
   completed year from 1951 onward; incomplete current-year files are rejected.
4. Upload the GRD and, for recent data, its matching CTL.
5. Confirm the product and documented axis orientation, then select
   **Validate and import IMD data**.

For an annual file already in this workspace, run from the repository root:

```powershell
.\.venv\Scripts\python -m climate.imd_import data/Maxtemp_MaxT_2025.GRD --year 2025 --byte-order little
```

Change the path to `data/raw/imd/Maxtemp_MaxT_2025.GRD` if appropriate. The command
requires an explicit year and byte order, validates before writing, prints the
actual counts/range/SHA-256 and Parquet path, and preserves the original input.
Its imported dataset is immediately listed in the dashboard's IMD workspace.
It never downloads data or invokes Open-Meteo. Repeating it reuses dataset identity.

`ParsedGrid.cube()` exposes a labeled NumPy analysis structure with dimensions
`(date, latitude, longitude)`, timezone-aware dates, coordinate arrays, units,
source and a float32 temperature cube. This is the coordinate-aware equivalent
of an xarray Dataset without introducing another dependency. Parquet retains the
complete long-form representation, including null cells.

Before accepting the data, the parser checks exact byte count, calendar record
count, control-file consistency, coordinate extent and increasing axis order.
It recognizes only the declared missing sentinel. Other nonfinite floats,
subnormal numbers suggestive of incorrect byte order, or values outside the broad
−90…65 °C sanity range reject the entire file. All-missing and constant-value
files are also rejected. These are format/quality checks, not official IMD
heatwave thresholds. Missing grid cells remain null; no spatial or temporal
imputation is applied.

**Headerless binary files cannot independently prove their variable, calendar
year, byte order or physical orientation.** A wrong year with the same day count,
or a transposed field whose values remain plausible, cannot be reliably detected
from the bytes alone. Metadata declarations must match the official download;
the app validates the declared contract and requires confirmation. A SHA-256
identifies the imported content; it is not a certificate of IMD authenticity.

## Storage and exports

Accepted imports preserve:

```text
data/imd/raw/<content-sha256>.grd
data/imd/processed/<dataset-id>.parquet
data/imd/metadata/<dataset-id>.json
```

Dataset identity includes the raw hash, grid contract, dates, byte order and
validation metadata. Import writes use temporary files and atomic replacement;
metadata is written last so incomplete conversions are not listed as datasets.
These generated files are ignored by Git. The raw file remains independent of
the structured table. Identical imports reuse their dataset identity.

Structured tables have exactly:

```text
date, latitude, longitude, maximum_temperature_c, data_source
```

Dates use Asia/Kolkata and denote the IMD **product date**. This does not assert
that a recent reported maximum has the same accumulation window as Open-Meteo's
calendar-day maximum. Imported metadata records the original filename, source
product, date coverage, byte order, orientation convention, valid/missing counts,
temperature extrema, content hash and import timestamp.

Download the complete long-format Parquet in **Data Processing**. The map
exports the selected date as CSV; the analytics panel exports the selected cell
series as CSV. Missing temperatures remain blank. Parquet reads use date/cell
filters so normal exploration does not load all imported years into memory.

## Maps and scientifically appropriate analysis

The map plots actual imported values on longitude/latitude axes, preserves
south-to-north orientation, and leaves missing cells transparent. It uses no
basemap service or additional geographic download. The rectangular IMD grid
domain includes surrounding territory; no political boundary mask or invented
India-only national average is applied. City markers show locations only.
Color limits use the selected dataset's full range for consistent date
comparisons. No spatial interpolation is added by the plot.

Select the grid product date once above the four primary sections. Overview's
valid-cell count, minimum/maximum and hottest coordinate refer to that selected
date, while whole-file cell-day counts are separately labeled. Historical
Analysis adds the top five available cells, retaining original coordinates and
product dates. Map hover includes the date, latitude, longitude and Tmax.

The optional **Compare two IMD dates** expander uses the selected product date
as Date A and a second date from the same imported dataset as Date B. Temperature
maps retain identical dataset-wide color limits. Differences are B minus A at
exact coordinate matches; a missing value on either date leaves the difference
missing. The difference scale is symmetric around zero, with increased Tmax red
and decreased Tmax blue. Findings report actual largest positive/negative
differences, including their coordinates, and handle no increases, no decreases
and no joint coverage explicitly. The CSV retains both dates and original values.
Neither this operation nor hotspot rankings changes spatial resolution or
converts grids into station readings. Two-date differences do not establish a
long-term climate trend.

For analytics, select one version per historical year and an exact 1° grid
coordinate. City names help locate a nearby cell, but the cell's coordinates
are always displayed. **Nearest available to location** uses great-circle distance
to a cell with at least one valid date in the selected history. The requested
location, actual cell coordinates, distance in km, valid-date count and 1°
resolution are shown. Users may enter custom coordinates within the grid domain.
This is a spatial lookup, not interpolation to a city or a guarantee of complete
daily coverage. **Exact grid coordinate** keeps the requested cell even when it
is entirely missing. No mode fills absent temperatures.
Overlapping annual versions are rejected. Historical 1°
and recent 0.5° files are never joined into one temperature time series.

The map reports valid-cell count and lowest/highest available Tmax for the selected
date. Its grid-statistics expander provides downloadable daily, monthly and
calendar-year seasonal tables with sample mean, minimum/maximum Tmax and
valid/missing cell-day counts. These are unweighted summaries of the available
grid domain, not area-weighted India averages; geographic coverage can change.
Minimum Tmax means the lowest daily maximum, not Tmin. Winter combines January,
February and December within each file year. The cell panel also exposes the
daily series alongside monthly/seasonal summaries.

The extracted historical cell reuses the application's summary, rolling,
monthly/seasonal and Random Forest pipeline, including causal lag features,
chronological purged splits, validation selection, held-out MAE/RMSE and
persistence comparison. Tmax-only imports contain no daily minimum or daily mean;
those remain missing rather than being estimated from Tmax.

Model evaluation is optional in **ML Forecasting**, using the cell selected in
**Historical Analysis**. Predictions follow the imported history endpoint,
usually an old year-end, and are prominently labeled a **historical endpoint
experiment**, not live city forecasts. There are no Open-Meteo inputs in these
models. Targets are IMD spatially interpolated, station-derived grid estimates,
not independent measurements at each grid coordinate. The source notes that
station availability changes after 2008; assess coverage and product consistency
before drawing long-term conclusions. Arbitrary thresholds are exploratory and
are not IMD-issued warnings.

## Attribution

Source: **India Meteorological Department, Climate Research & Services, Pune,
Ministry of Earth Sciences, Government of India**. Retain the original product's
terms, attribution and disclaimer when sharing derived data; Open-Meteo's license
does not automatically apply to IMD imports.

The historical product requests citation of:

Srivastava, A. K., Rajeevan, M., and Kshirsagar, S. R. (2009), *Development of High
Resolution Daily Gridded Temperature Data Set (1969–2005) for the Indian Region*,
Atmospheric Science Letters, [doi:10.1002/asl.232](https://doi.org/10.1002/asl.232).

## Verification

```powershell
.\.venv\Scripts\python -m pytest tests/test_imd_parser.py tests/test_imd_store.py tests/test_imd_ui.py -q
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m streamlit run app.py
```

Tests check documented grid corners and ordering, sentinel masking, leap-year
sizes, explicit byte orders, malformed control files, corruption, conversion,
Parquet filtering, source-isolated model execution, empty-state source isolation,
date changes and dashboard model evaluation. `tests/test_imd_actual.py` additionally
checks the supplied actual 2025 file, including raw byte offsets and content hash,
Parquet round trips, maps, statistics, nearest-cell selection and dashboard model
execution. It explicitly skips if the official file is absent from another
checkout; it never substitutes generated weather for actual-file validation.
