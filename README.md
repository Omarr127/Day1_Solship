# 🔋 Energy AI Hackathon 2026 — Solship
**Team | Zewail City of Science and Technology**

---

## 📋 Problem Overview

Residential site in Italy with:
- **9 kWp** rooftop solar PV
- **16 kWh** lithium battery (max 8 kW charge/discharge, 90% round-trip efficiency)
- **6 kW** grid connection limit
- Italian Time-of-Use tariff (F1/F2/F3 bands)

**Challenge:** Build a load forecasting model + rolling-horizon battery dispatch controller to minimize electricity bill vs two baselines.

---

## 🗂️ Repository Structure

```
solship_hackathon/
│
├── README.md                          ← This file
├── requirements.txt                   ← All dependencies
│
├── Notebook1_EDA_Features.ipynb       ← EDA + Feature Engineering (2024 & 2025)
├── Notebook2_Forecaster.ipynb         ← XGBoost + LightGBM + Ensemble models
│
├── data/
│   ├── ENERGY_Hackathon_DataSet_Sheet1_.csv   ← Raw dataset (2024–2025)
│   ├── Weather_2024.csv                       ← Sondrio weather 2024
│   └── Weather_2025.csv                       ← Sondrio weather 2025
│
├── outputs/
│   ├── features_2024.csv              ← Engineered features (2024 training)
│   ├── features_2025.csv              ← Engineered features (2025 test)
│   ├── feature_cols.json              ← Feature column names
│   └── forecast_2025_apr_sep.csv      ← Final forecast (April + September 2025)
│
└── plots/
    ├── eda_01_overview.png
    ├── eda_02_heatmap.png
    ├── eda_08_weather.png
    ├── lgbm_loss_curve.png
    ├── lgbm_feature_importance.png
    ├── forecast_vs_actual_apr_sep.png
    ├── scatter_actual_vs_predicted.png
    └── nrmse_comparison.png
```

---

## ⚙️ Setup

### Option 1 — Google Colab (Recommended)
Upload all files to `/content/` and run notebooks top to bottom. Install dependencies in the first cell:
```python
!pip install -r requirements.txt
```

### Option 2 — Local
```bash
git clone https://github.com/<your-team>/solship_hackathon
cd solship_hackathon
pip install -r requirements.txt
jupyter notebook
```

---

## 🔬 Methodology

### Forecasting Design Philosophy

> **Optimizer window = 1 day (96 × 15-min steps)**
> All features use lags ≥ 96 steps (≥ 24 hours) — fully causal, zero intra-day leakage.
> The forecaster predicts the full next day at once; the optimizer plans battery dispatch for the whole day.

### Data Split
| Split | Period | Purpose |
|---|---|---|
| Training | Jan–Sep 2024 | Model training |
| Validation | Oct–Dec 2024 | Hyperparameter tuning, early stopping |
| Test | April & September 2025 | Final reported metrics |

> ⚠️ **No 2025 data was used during training or tuning.**

---

## 📊 Feature Groups

### Group 1 — Time / Calendar
| Feature | Description |
|---|---|
| `hour`, `dow`, `month` | Basic time components |
| `step_of_day` | 0–95 (15-min slot within the day) |
| `is_weekend`, `is_holiday`, `is_working_day` | Calendar flags |
| `days_until_holiday`, `days_after_holiday` | Holiday proximity |
| `sin/cos hour/dow/month/doy` | Fourier cyclical encoding |

### Group 2 — Domain / Tariff
| Feature | Description |
|---|---|
| `band_int` | F1=1, F2=2, F3=3 |
| `sell_price_lag_96` | Yesterday's sell price (known before today) |
| `price_spread` | Buy − sell price spread |
| `pv_lag_96`, `pv_lag_672` | PV from yesterday and last week |
| `mins_until_band_change` | Minutes until next tariff band change |

### Group 3 — Lag & Rolling (all ≥ 24h)
| Feature | Description |
|---|---|
| `load_lag_96` | Same time yesterday |
| `load_lag_192` | Same time 2 days ago |
| `load_lag_672` | Same time last week |
| `rolling_mean_96/192/672` | Rolling averages over 1/2/7 days |
| `rolling_std/max/min_96` | Rolling stats over yesterday |
| `rolling_mean_same_hour_7d` | Avg same slot last 7 days |
| `daily_load_total` | Yesterday's total consumption |
| `daily_peak_yesterday` | Yesterday's peak load |

### Group 4 — Extra / Advanced
| Feature | Description |
|---|---|
| `temp_x_step` | Temperature × step_of_day interaction |
| `temp_x_iswork` | Temperature × working day flag |
| `load_z_step` | Z-score of load at this step (using 7-day lag) |
| `load_ewma_24h`, `load_ewma_7d` | Exponentially weighted load |
| `net_load_lag_96/672` | (Load − PV) from yesterday/last week |
| `pv_surplus_lag_96` | PV surplus from yesterday |
| `load_trend_week` | Load trend vs last week |

### Weather Features (Sondrio, Italy)
| Feature | Description |
|---|---|
| `temp_c` | Current temperature (exogenous — known from forecast) |
| `temp_lag_24h`, `temp_lag_48h` | Temperature 1/2 days ago |
| `temp_ewma_24h` | Thermal inertia proxy |
| `is_hot_day`, `is_cold_day` | AC/heating usage flags |
| `temp_mean/max/min_yesterday` | Daily temperature summary |
| `radiation_sum_yesterday` | Total solar radiation yesterday |
| `cloud_mean_yesterday` | Average cloud cover yesterday |

---

## 🤖 Models

### LightGBM (Primary Model)
```python
LGBMRegressor(
    n_estimators      = 2000,
    learning_rate     = 0.02,
    max_depth         = 8,
    num_leaves        = 127,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_samples = 20,
    reg_alpha         = 0.05,
    reg_lambda        = 0.5,
    random_state      = 42,
)
```

### XGBoost (Secondary Model)
```python
XGBRegressor(
    n_estimators     = 2000,
    learning_rate    = 0.02,
    max_depth        = 8,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
)
```

### Ensemble
Weighted average using inverse-NRMSE weights from validation:
```
ensemble = w_xgb × XGBoost + w_lgbm × LightGBM
w_xgb  = (1/NRMSE_xgb)  / (1/NRMSE_xgb + 1/NRMSE_lgbm)
w_lgbm = (1/NRMSE_lgbm) / (1/NRMSE_xgb + 1/NRMSE_lgbm)
```

---

## 📈 Results — Day 1 Submission

### Validation (Oct–Dec 2024)

| Model | RMSE (kW) | MAE (kW) | NRMSE (%) |
|---|---|---|---|
| XGBoost | — | — | — |
| LightGBM | 0.0569 | 0.0185 | **3.5589** |
| Ensemble | — | — | — |

### Test Results

| Model | Period | RMSE (kW) | MAE (kW) | NRMSE (%) |
|---|---|---|---|---|
| LightGBM | April 2025 | 0.0746 | — | — |
| LightGBM | September 2025 | 0.0635 | — | — |

> Fill in remaining values after running Notebook 2.

---

## 📐 Evaluation Metrics

| Metric | Formula | Unit |
|---|---|---|
| RMSE | √(mean((actual − predicted)²)) | kW |
| MAE | mean(|actual − predicted|) | kW |
| NRMSE | RMSE / mean(actual) × 100 | % |

> **NRMSE is the primary ranking metric** per the hackathon problem statement.

---

## 📤 Day 1 Submission Checklist

- [x] Working forecaster (LightGBM + XGBoost + Ensemble)
- [x] RMSE / MAE / NRMSE on 2025 test set (April & September)
- [x] No training or tuning on 2025 data
- [x] All features fully causal (lags ≥ 24 hours)
- [ ] Written optimizer formulation (to be submitted by 17:00)

### Written Optimizer Formulation (Day 1 Required)

**Objective:** Minimize 2025 electricity bill

```
minimize  Σ [ P_grid(t) × buy_price(t) × 𝟙(P_grid>0)
             − |P_grid(t)| × sell_price(t) × 𝟙(P_grid<0) ] × Δt
```

**Method:** Model Predictive Control (MPC) with rolling horizon H = 96 steps (1 day)

**At each day d:**
1. Load full-day forecast from forecaster (96 × 15-min predictions)
2. Solve LP/MILP over 96-step horizon using CVXPY or PuLP
3. Execute all 96 dispatch decisions for the day
4. Advance to next day and repeat

**Hard Constraints:**
```
−6 kW  ≤ P_grid(t)    ≤  6 kW
−8 kW  ≤ P_battery(t) ≤  8 kW
 0     ≤ SoC(t)        ≤  1
 SoC(0) = 0.5
```

**Battery SoC update:**
```
If P_battery < 0 (charging):
    SoC(t+1) = SoC(t) + (−P_battery × √0.9 × Δt) / 16

If P_battery ≥ 0 (discharging):
    SoC(t+1) = SoC(t) − (P_battery / √0.9 × Δt) / 16
```

**Energy balance (enforced at every step):**
```
load(t) = pv(t) + P_grid(t) + P_battery(t)
```

---

## 👥 Team
- Zewail City of Science and Technology
- Nanotechnology and Electronics Engineering

## 📅 Event
- **Day 1:** EDA, Forecasting, Optimizer Formulation
- **Day 2:** Optimizer Implementation, Surprise Dataset, Presentations

---

*© 2026 — Energy AI Hackathon | Solship*
