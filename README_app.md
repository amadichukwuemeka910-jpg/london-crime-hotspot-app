# London Crime Hotspot Forecasting Streamlit App

## Project
**Structural Breaks and Hotspot Forecasting: Analysing Post-Pandemic Crime Patterns in London (2019–2024)**

This app is the working prototype for Francis's system integration and modelling contribution. It displays precomputed burglary and robbery hotspot forecasts by LSOA for a policy/government user.

## What the app does

The app loads validated modelling outputs and allows the user to:

- compare Seasonal Naive vs Random Forest tuned models;
- select crime type: Burglary or Robbery;
- select forecast month;
- filter by borough;
- view top predicted hotspot LSOAs;
- inspect prediction history for a selected LSOA;
- view monthly Precision@10% and Recall@10%;
- inspect error by borough and IMD decile;
- compare feature importance and permutation importance;
- review governance and responsible-use guidance.

## Why predictions are precomputed

The app does not train models live. It loads precomputed outputs from the modelling notebook. This makes the app:

- more reproducible;
- more stable for demonstration;
- consistent with the values reported in the group report;
- safer for policy use because the outputs are validated before display.

## Folder structure

```text
london_crime_hotspot_streamlit_app/
├── app.py
├── requirements.txt
├── README_app.md
├── .streamlit/
│   └── config.toml
└── data/
    ├── enhanced_predictions_for_app.csv
    ├── enhanced_model_results_table.csv
    ├── monthly_precision_recall_at_10.csv
    ├── rf_internal_feature_importance.csv
    ├── rf_permutation_importance.csv
    ├── rf_hyperparameter_tuning_results.csv
    ├── train_test_overfitting_check.csv
    ├── spatial_lag_sensitivity_check.csv
    ├── error_by_borough.csv
    ├── error_by_imd_decile.csv
    └── best_rf_hyperparameters.json
```

## How to run locally

1. Unzip the app folder.
2. Open a terminal inside the folder.
3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

## How to deploy on Streamlit Community Cloud

1. Upload the app folder to a GitHub repository.
2. Make sure `app.py`, `requirements.txt`, and the `data/` folder are committed.
3. Go to Streamlit Community Cloud.
4. Choose the repository.
5. Set the main file path as:

```text
app.py
```

6. Deploy.

## Responsible use

This app is intended for strategic planning and interpretation. It should not be used as a sole basis for enforcement, surveillance, or individual-level policing decisions.