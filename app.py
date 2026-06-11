
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# London Crime Hotspot Forecasting App
# COMP1884 Group Project
# Francis - Technical Lead / Modelling
# ============================================================

st.set_page_config(
    page_title="London Crime Hotspot Forecasting",
    page_icon="🚨",
    layout="wide"
)

DATA_DIR = Path(__file__).parent / "data"


# -------------------------------
# Data loading functions
# -------------------------------
@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        st.error(f"Missing file: {filename}. Please check the data folder.")
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def load_json(filename: str) -> dict:
    path = DATA_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


predictions = load_csv("enhanced_predictions_for_app.csv")
model_results = load_csv("enhanced_model_results_table.csv")
monthly_metrics = load_csv("monthly_precision_recall_at_10.csv")
internal_importance = load_csv("rf_internal_feature_importance.csv")
permutation_importance = load_csv("rf_permutation_importance.csv")
tuning_results = load_csv("rf_hyperparameter_tuning_results.csv")
overfitting_check = load_csv("train_test_overfitting_check.csv")
spatial_lag_sensitivity = load_csv("spatial_lag_sensitivity_check.csv")
error_by_borough = load_csv("error_by_borough.csv")
error_by_imd = load_csv("error_by_imd_decile.csv")
best_params = load_json("best_rf_hyperparameters.json")

# Standardise date columns
predictions["date"] = pd.to_datetime(predictions["date"])
monthly_metrics["date"] = pd.to_datetime(monthly_metrics["date"])

# Friendly labels
MODEL_COLUMN_MAP = {
    "Random Forest tuned": "random_forest_tuned_pred",
    "Seasonal Naive": "seasonal_naive_pred"
}


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.title("Forecast controls")

crime_types = sorted(predictions["crime_type"].dropna().unique().tolist())
selected_crime = st.sidebar.selectbox("Crime type", crime_types)

models = ["Random Forest tuned", "Seasonal Naive"]
selected_model = st.sidebar.selectbox("Prediction model", models)
prediction_column = MODEL_COLUMN_MAP[selected_model]

available_months = (
    predictions.loc[predictions["crime_type"] == selected_crime, "date"]
    .drop_duplicates()
    .sort_values()
)
month_labels = [d.strftime("%B %Y") for d in available_months]
month_lookup = dict(zip(month_labels, available_months))
selected_month_label = st.sidebar.selectbox("Forecast month", month_labels)
selected_month = month_lookup[selected_month_label]

boroughs = ["All boroughs"] + sorted(
    predictions.loc[predictions["crime_type"] == selected_crime, "Actual_Borough_Name"]
    .dropna()
    .unique()
    .tolist()
)
selected_borough = st.sidebar.selectbox("Borough filter", boroughs)

top_n = st.sidebar.slider("Number of hotspot LSOAs to display", 5, 50, 15, step=5)


# -------------------------------
# Main header
# -------------------------------
st.title("London Crime Hotspot Forecasting App")
st.caption(
    "COMP1884 Group Project | Structural Breaks and Hotspot Forecasting: "
    "Analysing Post-Pandemic Crime Patterns in London (2019–2024)"
)

st.markdown(
    """
This prototype displays **precomputed hotspot forecasts** for burglary and robbery across London LSOAs.
It is designed for a government or policy user who wants to compare model outputs, identify predicted
high-risk LSOAs, and understand the limitations of predictive hotspot forecasting.

The app does **not** retrain models live. It loads validated outputs from the modelling notebook so that
the displayed predictions match the report results.
"""
)


# -------------------------------
# Filter data
# -------------------------------
crime_data = predictions[predictions["crime_type"] == selected_crime].copy()

month_data = crime_data[crime_data["date"] == selected_month].copy()
if selected_borough != "All boroughs":
    month_data = month_data[month_data["Actual_Borough_Name"] == selected_borough].copy()

if month_data.empty:
    st.warning("No data is available for the selected filters.")
    st.stop()


# -------------------------------
# KPI row
# -------------------------------
metric_row = model_results[
    (model_results["Crime type"] == selected_crime) &
    (model_results["Model"] == selected_model)
].iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("RMSE", f"{metric_row['RMSE']:.3f}")
col2.metric("MAE", f"{metric_row['MAE']:.3f}")
col3.metric("R²", f"{metric_row['R2']:.3f}")
col4.metric("Precision@10%", f"{metric_row['Precision@10%']:.3f}")
col5.metric("Recall@10%", f"{metric_row['Recall@10%']:.3f}")

st.info(
    f"Selected view: **{selected_crime}**, **{selected_model}**, **{selected_month_label}**, "
    f"borough filter: **{selected_borough}**."
)


# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Hotspot forecast",
    "Model comparison",
    "Prediction lookup",
    "Fairness and error checks",
    "Feature importance",
    "Governance guidance"
])


# ============================================================
# Tab 1: Hotspot forecast
# ============================================================
with tab1:
    st.header("Predicted hotspot LSOAs")

    st.markdown(
        """
The table ranks LSOAs by the selected model's predicted crime count for the selected month.
For policy use, this should be interpreted as **relative hotspot risk**, not a guaranteed future crime count.
"""
    )

    display_cols = [
        "LSOA_Code",
        "Actual_Borough_Name",
        "IMD_decile_1_most_deprived",
        "actual",
        "seasonal_naive_pred",
        "random_forest_tuned_pred",
        "absolute_error_rf"
    ]

    hotspot_table = (
        month_data
        .sort_values(prediction_column, ascending=False)
        .head(top_n)[display_cols]
        .rename(columns={
            "LSOA_Code": "LSOA code",
            "Actual_Borough_Name": "Borough",
            "IMD_decile_1_most_deprived": "IMD decile (1=most deprived)",
            "actual": "Actual count",
            "seasonal_naive_pred": "Seasonal Naive prediction",
            "random_forest_tuned_pred": "Random Forest prediction",
            "absolute_error_rf": "RF absolute error"
        })
    )

    st.dataframe(
        hotspot_table,
        use_container_width=True,
        hide_index=True
    )

    fig_hotspots = px.bar(
        hotspot_table.sort_values("Random Forest prediction", ascending=True),
        x="Random Forest prediction" if selected_model == "Random Forest tuned" else "Seasonal Naive prediction",
        y="LSOA code",
        orientation="h",
        hover_data=["Borough", "Actual count", "IMD decile (1=most deprived)"],
        title=f"Top predicted {selected_crime.lower()} hotspot LSOAs — {selected_month_label}"
    )
    fig_hotspots.update_layout(xaxis_title="Predicted count", yaxis_title="LSOA")
    st.plotly_chart(fig_hotspots, use_container_width=True)

    st.markdown(
        """
**Observation:** The highest-ranked LSOAs are the model's strongest hotspot candidates for the selected month.
The app intentionally reports LSOA codes and boroughs rather than individual-level information, supporting safer
area-level planning rather than person-level prediction.
"""
    )


# ============================================================
# Tab 2: Model comparison
# ============================================================
with tab2:
    st.header("Random Forest vs Seasonal Naive")

    st.markdown(
        """
This section compares the tuned Random Forest model against the Seasonal Naive baseline. Seasonal Naive is a
simple benchmark that repeats the equivalent month from the previous year. Random Forest should only be preferred
if it improves prediction accuracy and hotspot detection enough to justify its added complexity.
"""
    )

    crime_results = model_results[model_results["Crime type"] == selected_crime].copy()
    st.subheader("Overall model metrics")
    st.dataframe(crime_results.round(4), use_container_width=True, hide_index=True)

    metric_choice = st.selectbox(
        "Choose metric to compare",
        ["RMSE", "MAE", "R2", "Precision@10%", "Recall@10%", "Monthly Precision@10%", "Monthly Recall@10%"]
    )

    fig_metric = px.bar(
        crime_results,
        x="Model",
        y=metric_choice,
        text=crime_results[metric_choice].round(3),
        title=f"{selected_crime}: model comparison by {metric_choice}"
    )
    fig_metric.update_traces(textposition="outside")
    fig_metric.update_layout(yaxis_title=metric_choice)
    st.plotly_chart(fig_metric, use_container_width=True)

    st.subheader("Actual vs predicted monthly trend")
    monthly_agg = (
        crime_data
        .groupby("date", as_index=False)[["actual", "seasonal_naive_pred", "random_forest_tuned_pred"]]
        .sum()
    )

    fig_trend = px.line(
        monthly_agg,
        x="date",
        y=["actual", "seasonal_naive_pred", "random_forest_tuned_pred"],
        markers=True,
        title=f"{selected_crime}: actual vs predicted monthly total counts"
    )
    fig_trend.update_layout(xaxis_title="Month", yaxis_title="Monthly count", legend_title="Series")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Monthly hotspot detection performance")
    monthly_filtered = monthly_metrics[
        (monthly_metrics["crime_type"] == selected_crime) &
        (monthly_metrics["model"] == selected_model)
    ].copy()

    fig_monthly = px.line(
        monthly_filtered,
        x="date",
        y=["Precision@10%", "Recall@10%"],
        markers=True,
        title=f"{selected_crime}: monthly Precision@10% and Recall@10% ({selected_model})"
    )
    fig_monthly.update_layout(xaxis_title="Month", yaxis_title="Score", yaxis_range=[0, 1])
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.markdown(
        """
**Observation:** A stronger model should show lower RMSE/MAE and higher Precision@10%/Recall@10%.
The monthly hotspot chart is important because the app is intended to support monthly planning, not only one
aggregate test-period summary.
"""
    )


# ============================================================
# Tab 3: Prediction lookup
# ============================================================
with tab3:
    st.header("Prediction based on user inputs")

    st.markdown(
        """
Use this panel to inspect a specific LSOA. This is a **lookup of precomputed predictions**, not live retraining.
That makes the app reproducible and consistent with the validated notebook outputs.
"""
    )

    lookup_boroughs = sorted(crime_data["Actual_Borough_Name"].dropna().unique().tolist())
    lookup_borough = st.selectbox("Select borough for lookup", lookup_boroughs, key="lookup_borough")

    lookup_subset = crime_data[crime_data["Actual_Borough_Name"] == lookup_borough]
    lookup_lsoas = sorted(lookup_subset["LSOA_Code"].dropna().unique().tolist())
    selected_lsoa = st.selectbox("Select LSOA", lookup_lsoas)

    lsoa_rows = (
        crime_data[crime_data["LSOA_Code"] == selected_lsoa]
        .sort_values("date")
        .copy()
    )

    fig_lsoa = px.line(
        lsoa_rows,
        x="date",
        y=["actual", "seasonal_naive_pred", "random_forest_tuned_pred"],
        markers=True,
        title=f"{selected_crime}: prediction history for {selected_lsoa}"
    )
    fig_lsoa.update_layout(xaxis_title="Month", yaxis_title="Crime count", legend_title="Series")
    st.plotly_chart(fig_lsoa, use_container_width=True)

    selected_lsoa_month = lsoa_rows[lsoa_rows["date"] == selected_month]
    if not selected_lsoa_month.empty:
        row = selected_lsoa_month.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual count", f"{row['actual']:.0f}")
        c2.metric("Seasonal Naive", f"{row['seasonal_naive_pred']:.2f}")
        c3.metric("Random Forest", f"{row['random_forest_tuned_pred']:.2f}")
        c4.metric("IMD decile", f"{int(row['IMD_decile_1_most_deprived'])}")

    st.markdown(
        """
**Observation:** This lookup helps users understand how one neighbourhood's prediction changes across the
test period. It should be used for interpretation and prioritisation discussions, not as an automated trigger
for enforcement.
"""
    )


# ============================================================
# Tab 4: Fairness and error checks
# ============================================================
with tab4:
    st.header("Fairness and error checks")

    st.markdown(
        """
This section supports the fairness audit by showing where model errors are higher and whether error varies across
boroughs or deprivation deciles. It does not prove discrimination by itself, but it helps identify areas requiring
careful interpretation.
"""
    )

    st.subheader("Error by borough")
    borough_error = error_by_borough[error_by_borough["crime_type"] == selected_crime].copy()
    borough_error_top = borough_error.sort_values("mean_absolute_error", ascending=False).head(15)

    fig_borough_error = px.bar(
        borough_error_top.sort_values("mean_absolute_error", ascending=True),
        x="mean_absolute_error",
        y="Actual_Borough_Name",
        orientation="h",
        title=f"{selected_crime}: boroughs with highest mean absolute error"
    )
    fig_borough_error.update_layout(xaxis_title="Mean absolute error", yaxis_title="Borough")
    st.plotly_chart(fig_borough_error, use_container_width=True)

    st.dataframe(
        borough_error.sort_values("mean_absolute_error", ascending=False).round(3),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Error by IMD deprivation decile")
    imd_error = error_by_imd[error_by_imd["crime_type"] == selected_crime].copy()

    fig_imd_error = px.line(
        imd_error.sort_values("IMD_decile_1_most_deprived"),
        x="IMD_decile_1_most_deprived",
        y="mean_absolute_error",
        markers=True,
        title=f"{selected_crime}: prediction error by IMD decile"
    )
    fig_imd_error.update_layout(
        xaxis_title="IMD decile (1 = most deprived, 10 = least deprived)",
        yaxis_title="Mean absolute error"
    )
    st.plotly_chart(fig_imd_error, use_container_width=True)

    st.dataframe(imd_error.round(3), use_container_width=True, hide_index=True)

    st.subheader("Spatial lag sensitivity check")
    lag_sensitivity = spatial_lag_sensitivity[spatial_lag_sensitivity["Crime type"] == selected_crime].copy()

    fig_lag = px.bar(
        lag_sensitivity,
        x="Model",
        y="RMSE",
        text=lag_sensitivity["RMSE"].round(3),
        title=f"{selected_crime}: performance with and without Spatial_Lag_Predictor"
    )
    fig_lag.update_traces(textposition="outside")
    fig_lag.update_layout(xaxis_title="", yaxis_title="RMSE")
    st.plotly_chart(fig_lag, use_container_width=True)

    st.markdown(
        """
**Observation:** Borough and IMD error charts help identify whether the model is less reliable in particular
places or deprivation groups. The spatial-lag sensitivity check is included because a contemporaneous spatial lag
can be problematic for real future forecasting. If removing the variable causes only a small performance drop,
the model is more robust than if performance collapses.
"""
    )


# ============================================================
# Tab 5: Feature importance
# ============================================================
with tab5:
    st.header("Feature importance")

    st.markdown(
        """
Feature importance explains which inputs most influenced Random Forest predictions. Two forms are shown:

- **Internal Random Forest importance:** importance from the tree model itself.
- **Permutation importance:** how much performance worsens when a feature is shuffled.

Permutation importance is especially useful because it checks whether a feature really contributes to prediction
performance on held-out data.
"""
    )

    st.subheader("Internal Random Forest importance")
    imp = internal_importance[internal_importance["crime_type"] == selected_crime].copy()
    fig_imp = px.bar(
        imp.sort_values("importance", ascending=True).tail(12),
        x="importance",
        y="feature",
        orientation="h",
        title=f"{selected_crime}: internal Random Forest feature importance"
    )
    fig_imp.update_layout(xaxis_title="Importance", yaxis_title="")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Permutation importance")
    perm = permutation_importance[permutation_importance["crime_type"] == selected_crime].copy()
    fig_perm = px.bar(
        perm.sort_values("permutation_importance_mean", ascending=True).tail(12),
        x="permutation_importance_mean",
        y="feature",
        orientation="h",
        error_x="permutation_importance_std",
        title=f"{selected_crime}: permutation feature importance"
    )
    fig_perm.update_layout(xaxis_title="Mean importance", yaxis_title="")
    st.plotly_chart(fig_perm, use_container_width=True)

    st.dataframe(perm.sort_values("permutation_importance_mean", ascending=False).round(5), use_container_width=True, hide_index=True)

    st.markdown(
        """
**Observation:** If lag variables and spatial lag appear near the top, the model is mainly learning from recent
crime history and nearby-area crime pressure. This supports the project's spatial-temporal argument while also
showing why predictions should be interpreted carefully: the model reflects patterns in recorded crime data.
"""
    )


# ============================================================
# Tab 6: Governance guidance
# ============================================================
with tab6:
    st.header("Responsible-use and governance guidance")

    st.markdown(
        """
### What goes in
The app uses LSOA-month level features including historical burglary/robbery counts, lag variables,
spatial lag, IMD deprivation score, mobility proxy, high street distance and pandemic phase indicators.

### What comes out
The app outputs precomputed predicted counts and hotspot rankings for burglary and robbery during the
post-pandemic test period.

### How outputs should be interpreted
Predictions should be treated as **decision-support evidence**, not as automatic policing instructions.
A high predicted value means the LSOA has a higher relative hotspot risk according to the model, not that
crime is certain to occur or that residents are responsible for crime.

### Main limitations
- The model uses police-recorded crime, which may reflect reporting behaviour and enforcement patterns.
- LSOA-level aggregation avoids individual prediction but can still hide local variation.
- IMD is useful for fairness auditing but should not be interpreted as proving deprivation causes crime.
- Spatial lag may encode neighbouring recorded crime pressure and should be handled carefully for future deployment.
- The app uses a validated post-pandemic holdout period, but future crime patterns may change again.

### Recommended use
Use the app for strategic planning, resource discussion, prevention targeting and transparency review.
Do not use it as a sole basis for enforcement, surveillance or individual-level decision-making.
"""
    )

    st.subheader("Best tuned Random Forest hyperparameters")
    st.json(best_params)

    st.subheader("Overfitting check")
    st.dataframe(overfitting_check.round(4), use_container_width=True, hide_index=True)

    st.markdown(
        """
**Final insight:** The tuned Random Forest is the recommended model because it improves hotspot detection
over Seasonal Naive while remaining explainable through feature-importance and error-audit outputs.
However, the governance checks show that prediction accuracy must be balanced against fairness, uncertainty
and responsible interpretation.
"""
    )


# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption(
    "Prototype created for COMP1884 Group Project. Predictions are precomputed from the validated modelling notebook."
)
