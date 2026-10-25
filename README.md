# Exploring vessel movement near Pakistan using AIS data

**Python · Pandas · Geospatial analysis · DBSCAN · Machine learning · Plotly**

I used historical AIS reports to explore where vessels were observed near Karachi and the northern Arabian Sea, how movement varied, and whether a model could predict speed at the **next received report** better than simply using the last recorded speed.

**At a glance:** 43,283 MarineTraffic AIS records collected approximately hourly (18 August–1 October 2023) · 42,722 records analysed after screening · 1,638 vessels · 66-minute median gap between successive reports.

**Research question:** What can I learn about vessel activity from irregularly sampled AIS data, and how reliable are simple spatial and predictive methods on that data?

## What I explored

1. **Data quality and trajectories:** Cleaned timestamps and AIS units, removed duplicate and unsuitable records, and calculated distances and time gaps between reports from the same vessel.
2. **Spatial patterns:** Mapped Gaussian-smoothed **reporting density** on a geographic basemap and tested DBSCAN across 1, 2, 5, 10 and 20 km neighbourhood radii. The DBSCAN approximation uses 0.5 km weighted cells and haversine distance.
3. **Unusual movement screening:** Applied interpretable movement rules and separate Isolation Forest models for larger vessel categories. These are unlabelled screening examples, not verified incidents.
4. **Speed prediction:** Compared persistence, squared-error gradient boosting, absolute-error boosting of speed changes, and a stationary-vessel hybrid using three chronological evaluation windows.

## Results from the original research export

### Spatial clustering

| Neighbourhood radius | Clusters | Reports classified as noise |
|---|---:|---:|
| 1 km | 100 | 34.8% |
| 2 km | 379 | 26.0% |
| 5 km | 101 | 1.8% |
| 10 km | 7 | 0.1% |
| 20 km | 1 | 0.0% |

The number and size of clusters depend strongly on the radius. I would not treat a cluster as a confirmed shipping corridor without checking it against geography and vessel activity.

### Forecasting

| Test window | Persistence MAE | Squared-error model MAE | Absolute-error change model MAE | Hybrid MAE |
|---|---:|---:|---:|---:|
| 1 | 0.912 | 1.053 | **0.835** | 0.837 |
| 2 | 1.065 | 1.125 | **0.918** | 0.918 |
| 3 | 0.827 | 1.041 | **0.736** | 0.738 |

MAE is in knots; lower is better. The absolute-error change model had lower MAE than persistence in all three windows. The squared-error model had lower RMSE than persistence, illustrating that model rankings depend on the error metric. See [`outputs/rolling_forecast_metrics.csv`](outputs/rolling_forecast_metrics.csv) for MAE, RMSE, and vessel-bootstrap confidence intervals for the model-versus-persistence MAE differences.

Training uses only observations whose **next-report targets were already available before the test cutoff**. Test periods follow training periods in time. Some vessels occur in both, so these results do not measure generalization to entirely new vessels. Next-report timing is variable; this is not a fixed 30- or 60-minute prediction.

### Coverage and limitations

The reports are approximately hourly but not evenly spaced. Complete UTC days range from about 520 to 2,118 received records, so the density map reflects reporting coverage as well as vessel activity. The data cover Pakistan's coastal approaches and the northern Arabian Sea; I have not verified that every record is inside Pakistan's EEZ. Anomaly screening has no labelled ground truth, so I do not report detection accuracy.

## Maps and example output

The interactive maps in [`demo_outputs/`](demo_outputs/) were generated from **synthetic demonstration data**, not the commercial MarineTraffic export. They show how the code works, not the original study's geography or findings.

- [Synthetic AIS reporting density map](demo_outputs/kde_density.html)
- [Synthetic spatial cluster map](demo_outputs/cluster_centres.html)
- [Original-data aggregate forecast metrics](outputs/rolling_forecast_metrics.csv)
- [Original-data aggregate spatial sensitivity](outputs/dbscan_sensitivity.csv)

## Reproduce the code

The repository contains a **fully synthetic** dataset (`data/ais_demo.csv`) with AIS-like columns and simulated tracks. It lets anyone run the entire pipeline without redistributing commercial vessel records.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m src.research --input data/ais_demo.csv --output my_demo_results
```

To regenerate the demo data with the same random seed:

```bash
python -m src.make_demo --output data/ais_demo.csv
```

The published aggregate findings were calculated on a separate, non-public MarineTraffic export of 43,283 observations. The synthetic dataset is for **code reproducibility**, not numerical reproduction of those original-data results. Researchers with their own authorized AIS export can pass its path to `--input`.

## Repository structure

- `src/pipeline.py`: parsing, unit conversions and movement calculations
- `src/research.py`: spatial analysis, screening, forecasts and evaluation
- `src/make_demo.py`: synthetic AIS-like data generator
- `tests/`: unit tests, leakage checks, JSON checks and a CI smoke test
- `data/`: synthetic sample and full synthetic demo
- `outputs/`: original-data **aggregate** results only
- `demo_outputs/`: synthetic-data interactive examples

## What I learned

Working with AIS data showed me why the time between reports matters, why clustering parameters need testing, and why a simple baseline can be competitive with machine learning. I also learned to keep the prediction target and evaluation metric aligned, and to check that future labels do not enter the training period.

This is an exploratory research project, not a maritime monitoring or incident-detection product.
