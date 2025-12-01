# dashboard.py
import pandas as pd
import joblib
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# === Paths ===
DATA_PATH = "data/Fraud.csv"
MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"

# === Load raw data ===
df = pd.read_csv(DATA_PATH)

# === Load trained model and feature list ===
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

# === Dashboard preprocessing (same as training) ===
df["amount_to_oldbalance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1e-9)
df["amount_to_newbalance_ratio"] = df["amount"] / (df["newbalanceOrig"] + 1e-9)
df["balance_diff_org"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

# One-hot encode transaction types
df = pd.get_dummies(df, columns=["type"], prefix="type")

# Ensure all features exist for model
for col in features:
    if col not in df.columns:
        df[col] = 0

# Keep only features used by the model
df_features = df[features]

# === Predict fraud ===
df["fraud_prob"] = model.predict_proba(df_features)[:, 1]
df["fraud_pred"] = model.predict(df_features)

# === Initialize Dash app ===
app = Dash(__name__)
app.title = "Fraud Detection Dashboard"

# === Layout ===
app.layout = html.Div(
    [
        html.H1("Fraud Detection Dashboard", style={"textAlign": "center"}),
        html.Div(
            [
                html.Label("Minimum Fraud Probability Threshold:"),
                dcc.Slider(
                    id="prob_threshold",
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.5,
                    marks={0: "0", 0.5: "0.5", 1: "1"},
                ),
                html.Label("Minimum Transaction Amount:"),
                dcc.Slider(
                    id="amount_filter",
                    min=df["amount"].min(),
                    max=df["amount"].max(),
                    step=1000,
                    value=df["amount"].min(),
                    marks={
                        int(df["amount"].min()): str(int(df["amount"].min())),
                        int(df["amount"].max()): str(int(df["amount"].max())),
                    },
                ),
            ],
            style={"margin": "20px"},
        ),
        html.Div(
            [
                dcc.Graph(id="fraud_hist"),
                dcc.Graph(id="transaction_type_pie"),
                dcc.Graph(id="top_risky_transactions"),
            ]
        ),
    ]
)


# === Callbacks ===
@app.callback(
    Output("fraud_hist", "figure"),
    Output("transaction_type_pie", "figure"),
    Output("top_risky_transactions", "figure"),
    Input("prob_threshold", "value"),
    Input("amount_filter", "value"),
)
def update_dashboard(prob_threshold, amount_filter):
    # Filter based on sliders
    df_filtered = df[
        (df["fraud_prob"] >= prob_threshold) & (df["amount"] >= amount_filter)
    ]

    # Fraud probability histogram
    hist_fig = px.histogram(
        df_filtered,
        x="fraud_prob",
        nbins=50,
        title="Fraud Probability Distribution",
        labels={"fraud_prob": "Predicted Fraud Probability"},
    )

    # Transaction type pie chart
    type_cols = [c for c in df_filtered.columns if c.startswith("type_")]
    type_counts = df_filtered[type_cols].sum().reset_index()
    type_counts.columns = ["type", "count"]
    type_counts["type"] = type_counts["type"].str.replace("type_", "")
    pie_fig = px.pie(
        type_counts,
        names="type",
        values="count",
        title="Transaction Types Distribution",
    )

    # Top 10 risky transactions
    top_risky = df_filtered.sort_values(by="fraud_prob", ascending=False).head(10)
    top_fig = px.bar(
        top_risky,
        x="nameOrig",
        y="fraud_prob",
        hover_data=["amount", "balance_diff_org", "balance_diff_dest"],
        title="Top 10 Risky Transactions",
        labels={
            "nameOrig": "Origin Account",
            "fraud_prob": "Predicted Fraud Probability",
        },
    )

    return hist_fig, pie_fig, top_fig


# === Run server ===
if __name__ == "__main__":
    app.run(debug=True)
