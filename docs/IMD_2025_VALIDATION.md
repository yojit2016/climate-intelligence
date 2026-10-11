# Actual IMD 2025 maximum-temperature validation

Validated locally on 11 October 2026 using the user's file
`data/Maxtemp_MaxT_2025.GRD`, obtained by the user from the
[official IMD annual Tmax page](https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html).
The application did not download this file. Source authenticity rests on its
official download provenance; binary validation alone cannot certify authenticity.

| Check | Actual result |
| --- | --- |
| SHA-256 | `673775eafe685d180c5d7a19018287599697a3f6800332470ab27ca512bd4b88` |
| Bytes | 1,403,060 |
| Encoding | User-declared little-endian float32; values pass validation |
| Shape | 365 × 31 × 31 = 350,765 cell-days |
| Dates | 1 January–31 December 2025, Asia/Kolkata product dates |
| Latitude | 7.5–37.5°N, increasing at 1° intervals |
| Longitude | 67.5–97.5°E, increasing at 1° intervals |
| Valid cell-days | 129,552 |
| Missing cell-days | 221,213; float32 `99.9` decoded to NaN |
| Lowest valid Tmax | 5.30673885345459 °C |
| Highest valid Tmax | 46.15256118774414 °C |
| Dataset ID | `5c87e8972fc16a6283bda4679fcf4a812b05c25fa22f077f8bb2977cbf299139` |

The processed file is
`data/imd/processed/5c87e8972fc16a6283bda4679fcf4a812b05c25fa22f077f8bb2977cbf299139.parquet`.
Raw content-addressed GRD and JSON metadata are retained separately under
`data/imd/raw/` and `data/imd/metadata/`. These are local caches, excluded from Git.

## Spatial ordering verification

The indexed official Tmax page was rechecked against its C sample, Fortran sample
and embedded GrADS control example. The C sample reads each daily `t[31][31]`
and traverses the second index in the inner loop; the Fortran reader likewise
traverses J inside I in each daily record. The control file gives increasing
XDEF from 67.5°E and YDEF from 7.5°N, without reversed-axis options. Together these
specify longitude changing fastest, then south-to-north latitude, then date.
The control file's illustrative calendar is not copied onto the 2025 data.

The actual-file test independently unpacks float32 values at
`4 * ((day_index * 31 + latitude_index) * 31 + longitude_index)` for every cell on
1 January, 15 April and 31 December and compares them with the coordinate-aware
cube. The same dates are checked after Parquet round trips. A map's full 15 April
31×31 temperature matrix is checked against the decoded cube, including NaNs.
This verifies implementation of the documented orientation. Headerless bytes
cannot independently prove their declared physical axes or calendar year.

## Analysis and dashboard checks

The actual-file regression suite checks 365 daily, 12 monthly and four seasonal
summary groups; valid/missing counts reconcile to the binary for every grouping.
The nearest available cell for Delhi's reference location is 28.5°N, 77.5°E with
365 valid dates. A lookup from the masked southwest corner also confirms missing
cells are skipped only in the explicitly selected nearest-available mode.

Streamlit AppTest opens the actual-file workspace, changes the map date, inspects
nearest/exact cell modes, and evaluates the historical Random Forest pipeline.
The model is trained on the real extracted cell series. Predictions follow the
31 December 2025 history endpoint and are labeled historical experiments. This
one-year file provides no multi-year climate baseline or live observation stream.
No Open-Meteo values or synthetic fixtures are used by these actual-file tests.

Reproduce from the repository root:

```powershell
.\.venv\Scripts\python -m climate.imd_import data/Maxtemp_MaxT_2025.GRD --year 2025 --byte-order little
.\.venv\Scripts\python -m pytest tests/test_imd_actual.py -q
.\.venv\Scripts\python -m streamlit run app.py
```

Choose **IMD gridded temperature** in the sidebar, then select the 2025 dataset.
See [IMD documentation](IMD.md) for attribution and scientific limitations.
