# AutoML for Time Series Forecasting of Dissolved Oxygen

This repository contains the data and notebooks used in the paper **"Does Preprocessing Quality Matter for AutoML? A Case Study on Time Series Forecasting with Environmental Data"**. 
The study evaluates automated machine learning frameworks for forecasting hourly Dissolved Oxygen (DO, mg/L) across six monitoring stations on the Tietê and Piracicaba river basins (São Paulo, Brazil), comparing three input data variants produced by a hierarchical gap-filling pipeline.

---

## Raw Data

### `EF01 - OD.csv` · `EF02 - OD.csv` · `EF03 - OD.csv` · `EF06 - OD.csv` · `EF07 - OD.csv` · `EF08 - OD.csv`

Raw hourly Dissolved Oxygen time series exported from CETESB's ANA/QUALAR monitoring network. Each file corresponds to one water quality monitoring station.

**Format**

| Column | Description |
|---|---|
| `Data` | Datetime string — `DD/MM/YYYY HH:MM` |
| `Minimo` | Minimum DO reading in the hour (mg/L) |
| `Maximo` | Maximum DO reading in the hour (mg/L) |
| `Media` | Mean DO reading in the hour (mg/L) — **used as the target variable** |
| `Mediana` | Median DO reading in the hour (mg/L) |
| `Desvio padrao` | Standard deviation within the hour |
| `Variancia` | Variance within the hour |

- Separator: `;` — Decimal separator: `,`
- Series span: ~2011–2025 (varies by station)
- The series contain structural gaps (sensor outages, maintenance) and constancy periods (sensor fouling), which are addressed by `DO_gap_filling_pipeline.ipynb`.

---

## Notebooks

### `DO_gap_filling_pipeline.ipynb`

Applies a quality-control and hierarchical gap-filling pipeline to all six raw CSV files and produces the three dataset variants used as input to the forecasting notebooks.

**Pipeline stages**

1. **Physical QC** — removes values outside the physically plausible range (0–14.5 mg/L).
2. **Sparse-start trimming** — discards the initial portion of each series where data density is too low for reliable modelling.
3. **Constancy detection** — flags extended flat-line segments caused by sensor fouling or drift and marks them as missing.
4. **Hierarchical gap-filling** — selects the filling method based on gap length:
   - ≤ 10 h → linear time interpolation
   - 11–72 h → centred moving average (MA) or STL decomposition
   - > 72 h → Prophet seasonal model
5. **Baseline construction** — builds two naive baselines from the QC-cleaned series:
   - **Baseline I** — forward-fill then backward-fill (`ffill().bfill()`)
   - **Baseline II** — global linear interpolation
6. **Validation** — DTW-based holdout validation (100 synthetic gaps) and time series cross-validation (expanding window, tsCV).

**Outputs (per station)**

| File | Description |
|---|---|
| `EF{xx}_DO_imputed.csv` | Timestamps and imputed values only (filled positions) |
| `EF{xx}_DO_baseline_I.csv` | Full hourly series — Baseline I (ffill/bfill) |
| `EF{xx}_DO_baseline_II.csv` | Full hourly series — Baseline II (linear interpolation) |
| `EF{xx}_DO_complete_series.csv` | Full hourly series — Hierarchical Filling (pipeline output) |
| `EF{xx}_DO_imputed.png` | Per-station imputation overview plot |
| `all_stations_DO_imputed.png` | Combined 3×2 overview of all six stations |
| `station_map.png` | Geographic map of the six monitoring stations |

---

### `sktime.ipynb`

Forecasting notebook using the **sktime** library. Trains four models on each of the three dataset variants (Baseline I, Baseline II, Hierarchical Filling) and evaluates them on a held-out test period.

**Models**

| Model | Implementation |
|---|---|
| Prophet | `sktime` wrapper around Facebook Prophet |
| XGBoost | `RecursiveTabularRegressionForecaster` + XGBRegressor |
| SARIMA | `ARIMA` with seasonal order (0,1,1,24) |
| DeepAR | neuralforecast `NHITS`/LSTM — manual integration |

**Experimental setup**

- Training data: 2015-01-01 → 2024-12-31 (hourly, ~87 000 points per station)
- Test period: 2025-01-01 → 2025-03-31 (2 184 hours, 91 days)
- Forecast horizon: 2 184 steps ahead (one-shot)
- Seasonality: m = 24 h (daily)

**Metrics** (all seasonally scaled at m = 24): MASE, RMSSE, R², DA, MDA, MAED, CETrend

**Key outputs**

| File | Description |
|---|---|
| `sktime_metrics_FINAL.csv` | Mean metrics across stations — all models × datasets |
| `sktime_metrics_FINAL_completa.csv` | Full per-station × model × dataset metrics table |
| `sktime_metrics_{Model}.csv` | Per-model detailed metrics |
| `sktime_testes_estatisticos.csv` | Wilcoxon signed-rank tests: Baselines vs Hierarchical Filling |
| `sktime_tempos.csv` | Fit and prediction wall-clock times |
| `plots_sktime/EF{xx}_forecast.png` | 3-panel forecast plot per station (300 dpi) |

---

### `AutoGluonTS.ipynb`

Forecasting notebook using the **AutoGluon-TimeSeries** library. Mirrors the sktime experimental setup exactly, enabling direct model-to-model comparison.

**Models**

| Model | Implementation |
|---|---|
| DeepAR | AutoGluon native `DeepAR` (hidden size 64, 2 layers, 50 epochs) |
| XGBoost | AutoGluon `DirectTabular` + XGBRegressor |
| Prophet | Facebook Prophet — manual implementation outside AutoGluon |
| SARIMA | statsmodels `SARIMAX` — manual implementation outside AutoGluon |

**Experimental setup**

- Identical temporal split to sktime (training up to 2024-12-31, test Jan–Mar 2025)
- DeepAR and XGBoost run through `TimeSeriesPredictor`; Prophet and SARIMA are fitted manually per station to maintain comparability
- Context length: 24 × 7 h (one week) for neural/gradient-boosted models

**Metrics**: same set as sktime (MASE, RMSSE, R², DA, MDA, MAED, CETrend)

**Key outputs**

| File | Description |
|---|---|
| `AutoGluonTS_metrics_FINAL.csv` | Mean metrics across stations — all models × datasets |
| `AutoGluonTS_metrics_FINAL_completa.csv` | Full per-station × model × dataset metrics table |
| `AutoGluonTS_metrics_{Model}.csv` | Per-model detailed metrics |
| `AutoGluonTS_testes_estatisticos.csv` | Wilcoxon signed-rank tests: Baselines vs Hierarchical Filling |
| `AutoGluonTS_tempos.csv` | Fit and prediction wall-clock times |
| `plots_autogluon/EF{xx}_forecast.png` | 3-panel forecast plot per station (300 dpi) |

---

## Dataset Variants

All three forecasting notebooks use the same three input variants produced by `DO_gap_filling_pipeline.ipynb`:

| Variant | Description |
|---|---|
| **Baseline I** | QC-cleaned series filled with `ffill().bfill()` — no temporal interpolation |
| **Baseline II** | QC-cleaned series filled with global linear interpolation |
| **Hierarchical Filling** | QC-cleaned series filled by the full hierarchical pipeline (Interp. / MA+STL / Prophet) |

Test-set evaluation is always performed against the **raw original observations** to avoid any leakage from the filling procedure.

---

## Reproducibility

| Requirement | Version |
|---|---|
| Python | 3.10 |
| sktime | ≥ 0.30 |
| AutoGluon-TimeSeries | ≥ 1.2 |
| prophet | 1.3 |
| statsmodels | ≥ 0.14 |
| neuralforecast | ≥ 1.7 |

All notebooks are self-contained. Running `DO_gap_filling_pipeline.ipynb` first generates the CSV inputs required by `sktime.ipynb` and `AutoGluonTS.ipynb`.

## Computational Requirements

All experiments are CPU-compatible. No GPU is required to reproduce the results, though a GPU accelerates DeepAR training in `sktime.ipynb` and `AutoGluonTS.ipynb`.

All experiments in the paper were run on local hardware — no external cluster or cloud provider was used.

---

### `DO_gap_filling_pipeline.ipynb` — CPU only

The heaviest operation is fitting Prophet independently per gap > 72 h per station. Each Prophet fit runs in seconds on a single core. STL and MA are lightweight pandas/statsmodels operations.

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| GPU | Not required | Not required |

**Estimated wall-clock time:** ~5–15 min total.

> Memory note: Prophet fits can spike RAM briefly when processing long gaps across all six stations sequentially.

---

### `sktime.ipynb` — CPU only (GPU optional for DeepAR)

| Model | Compute | Notes |
|---|---|---|
| Prophet | CPU | 1 fit per station × 3 datasets = 18 fits |
| XGBoost | CPU, multi-core | `n_jobs=-1`, window 24×7 h, 300 trees |
| SARIMA | CPU | Window truncated to last 60 days (1 440 points) |
| DeepAR | CPU or GPU | neuralforecast NF_DeepAR, `max_steps=50`, LSTM hidden=64, 2 layers, input 24×7 h; trains globally on all 6 stations |

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 8 cores | 8 cores |
| RAM | 16 GB | 16 GB |
| GPU | Not required | ≥ 6 GB VRAM (e.g. NVIDIA RTX 3060) |

**Estimated wall-clock time:** ~30–60 min CPU-only (DeepAR dominates). Without a GPU, DeepAR trains via PyTorch/Lightning on CPU.

---

### `AutoGluonTS.ipynb` — CPU only (GPU strongly recommended for DeepAR)

| Model | Compute | Notes |
|---|---|---|
| DeepAR | GPU recommended | AutoGluon native DeepAR, hidden=64, 2 layers, `max_epochs=50`, context 24×7 h, `time_limit=1800 s` per fit |
| XGBoost | CPU | AutoGluon DirectTabular, 300 trees |
| Prophet | CPU | Manual fit per station × 3 datasets |
| SARIMA | CPU | statsmodels SARIMAX, manual fit per station |

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 8 cores | 8 cores |
| RAM | 16 GB | 16 GB |
| GPU | Not required | ≥ 8 GB VRAM (e.g. NVIDIA RTX 3070 / A10G) |

AutoGluon auto-detects CUDA and falls back to CPU training for DeepAR if no GPU is available. With a GPU, DeepAR training per dataset fits comfortably within the 1 800 s time limit.

**Estimated wall-clock time:** ~60–120 min CPU-only; ~20–40 min with GPU.

---

### Summary

| Notebook | GPU required | Min RAM | Recommended |
|---|---|---|---|
| `DO_gap_filling_pipeline` | No | 8 GB | 16 GB, CPU only |
| `sktime` | No (optional) | 16 GB | GPU ≥ 6 GB VRAM |
| `AutoGluonTS` | No (strongly recommended) | 16 GB | GPU ≥ 8 GB VRAM |
