# ------------------------------------------------------------
#  04_prediction.py
#  Stock Price Prediction using ML + Classical Time Series
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (TICKER, PREDICTION_HORIZONS,
                    TEST_SPLIT_RATIO, RANDOM_STATE, OUTPUT_DIR)

os.makedirs(os.path.join(OUTPUT_DIR, "charts"), exist_ok=True)


# ─────────────────────────────────────────
#  1. FEATURE ENGINEERING
# ─────────────────────────────────────────

def engineer_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates technical indicator features from OHLCV price history.
    These are the inputs (X) for the ML models.
    """
    df = price_df.copy()
    close = df["Close"]

    # --- Moving Averages ---
    df["MA_10"]  = close.rolling(10).mean()
    df["MA_20"]  = close.rolling(20).mean()
    df["MA_50"]  = close.rolling(50).mean()

    #  Price relative to MAs 
    df["Price_to_MA20"]  = close / df["MA_20"]
    df["Price_to_MA50"]  = close / df["MA_50"]

    #  Returns 
    df["Return_1d"]   = close.pct_change(1)
    df["Return_5d"]   = close.pct_change(5)
    df["Return_20d"]  = close.pct_change(20)

    #  Volatility 
    df["Volatility_20d"] = close.rolling(20).std() / close.rolling(20).mean()

    #  RSI (14-day) 
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    #  Bollinger Band Width 
    rolling_std  = close.rolling(20).std()
    upper_band   = df["MA_20"] + 2 * rolling_std
    lower_band   = df["MA_20"] - 2 * rolling_std
    df["BB_width"] = (upper_band - lower_band) / df["MA_20"]

    #  Volume ratio 
    df["Volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    #  Distance from 52-week high/low 
    df["Pct_from_52w_high"] = close / close.rolling(252).max() - 1
    df["Pct_from_52w_low"]  = close / close.rolling(252).min() - 1

    df.dropna(inplace=True)
    return df


def create_target(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """
    Target variable: future return over horizon_days.
    We predict % return, then convert to price.
    """
    return df["Close"].shift(-horizon_days) / df["Close"] - 1


FEATURE_COLS = [
    "MA_10", "MA_20", "MA_50",
    "Price_to_MA20", "Price_to_MA50",
    "Return_1d", "Return_5d", "Return_20d",
    "Volatility_20d", "RSI_14", "BB_width",
    "Volume_ratio", "Pct_from_52w_high", "Pct_from_52w_low"
]


# ─────────────────────────────────────────
#  2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────

def split_data(df: pd.DataFrame, target: pd.Series):
    """Time-series aware train/test split (no shuffling)."""
    df_clean = df[FEATURE_COLS].copy()
    df_clean["target"] = target
    df_clean.dropna(inplace=True)

    n        = len(df_clean)
    split    = int(n * (1 - TEST_SPLIT_RATIO))
    train    = df_clean.iloc[:split]
    test     = df_clean.iloc[split:]

    X_train = train[FEATURE_COLS]
    y_train = train["target"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["target"]

    return X_train, y_train, X_test, y_test, test.index


# ─────────────────────────────────────────
#  3. MODELS
# ─────────────────────────────────────────

def train_models(X_train, y_train):
    """Trains Linear Regression, Random Forest, and Gradient Boosting."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(
                                n_estimators=100,
                                max_depth=5,
                                random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
                                n_estimators=100,
                                max_depth=3,
                                learning_rate=0.05,
                                random_state=RANDOM_STATE),
    }

    trained = {}
    for name, model in models.items():
        if name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        trained[name] = model

    return trained, scaler


def evaluate_models(trained_models, scaler, X_test, y_test):
    """Returns metrics dataframe for all models."""
    X_test_scaled = scaler.transform(X_test)
    results = []

    for name, model in trained_models.items():
        if name == "Linear Regression":
            preds = model.predict(X_test_scaled)
        else:
            preds = model.predict(X_test)

        results.append({
            "Model":  name,
            "MAE":    round(mean_absolute_error(y_test, preds), 4),
            "RMSE":   round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            "R²":     round(r2_score(y_test, preds), 4),
        })

    return pd.DataFrame(results).set_index("Model")


def predict_future(trained_models, scaler, df_features, current_price):
    """
    Uses the most recent row of features to predict future returns,
    then converts to price targets.
    """
    latest = df_features[FEATURE_COLS].iloc[[-1]]
    latest_scaled = scaler.transform(latest)

    predictions = {}
    for name, model in trained_models.items():
        if name == "Linear Regression":
            pred_return = model.predict(latest_scaled)[0]
        else:
            pred_return = model.predict(latest)[0]
        pred_price = current_price * (1 + pred_return)
        predictions[name] = round(pred_price, 2)

    # Ensemble: average of all models
    predictions["Ensemble (Avg)"] = round(
        np.mean(list(predictions.values())), 2)

    return predictions


def get_feature_importance(trained_models):
    """Returns feature importances from tree-based models."""
    importances = {}
    for name in ["Random Forest", "Gradient Boosting"]:
        if name in trained_models:
            imp = pd.Series(
                trained_models[name].feature_importances_,
                index=FEATURE_COLS
            ).sort_values(ascending=False)
            importances[name] = imp
    return importances


# ─────────────────────────────────────────
#  4. CHARTS
# ─────────────────────────────────────────

def plot_predictions(price_df, trained_models, scaler,
                     df_features, horizon, ticker_symbol):
    """Plots actual vs predicted returns on test set."""
    target   = create_target(df_features, horizon)
    X_train, y_train, X_test, y_test, test_index = split_data(df_features, target)
    X_test_scaled = scaler.transform(X_test)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_index, y_test.values, label="Actual Return",
            color="#1E293B", linewidth=1.5)

    colors = ["#2563EB", "#16A34A", "#DC2626"]
    for (name, model), color in zip(trained_models.items(), colors):
        if name == "Linear Regression":
            preds = model.predict(X_test_scaled)
        else:
            preds = model.predict(X_test)
        ax.plot(test_index, preds, label=f"{name} (predicted)",
                color=color, alpha=0.7, linewidth=1)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_title(f"Predicted vs Actual Returns ({horizon}d horizon) — {ticker_symbol}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "charts",
        f"{ticker_symbol.replace('.','_')}_prediction_{horizon}d.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"   Prediction chart saved ({horizon}d): {path}")


def plot_feature_importance(importances, ticker_symbol):
    """Bar chart of top 10 most important features."""
    for model_name, imp in importances.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.head(10).sort_values().plot(kind="barh", ax=ax, color="#2563EB")
        ax.set_title(f"Feature Importance — {model_name} ({ticker_symbol})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        path = os.path.join(OUTPUT_DIR, "charts",
            f"{ticker_symbol.replace('.','_')}_feature_importance.png")
        fig.savefig(path, dpi=150)
        plt.close()
        print(f"   Feature importance chart saved: {path}")
        break   # save only one


# ─────────────────────────────────────────
#  5. MASTER PREDICTION FUNCTION
# ─────────────────────────────────────────

def run_full_prediction(cleaned_data: dict) -> dict:
    """
    Runs feature engineering, trains all models for each horizon,
    and returns predictions + metrics.
    """
    print("\n Running ML Prediction Engine...")

    price_df     = cleaned_data["price_history"]
    current_price = cleaned_data["info"].get("currentPrice")

    if price_df.empty or current_price is None:
        return {"error": "Insufficient price data for prediction."}

    df_features = engineer_features(price_df)
    print(f"  Features engineered: {df_features.shape[0]} rows × {len(FEATURE_COLS)} features")

    all_results = {}
    primary_horizon = PREDICTION_HORIZONS[-1]   # Use longest for training primary model

    # Train on longest horizon for feature importance
    target   = create_target(df_features, primary_horizon)
    X_train, y_train, X_test, y_test, _ = split_data(df_features, target)
    trained_models, scaler = train_models(X_train, y_train)
    metrics = evaluate_models(trained_models, scaler, X_test, y_test)
    importances = get_feature_importance(trained_models)

    print(f"\n  Model Performance (horizon: {primary_horizon} days):")
    print(metrics.to_string())

    # Predict for each horizon
    horizon_predictions = {}
    for h in PREDICTION_HORIZONS:
        target_h = create_target(df_features, h)
        Xtr, ytr, Xte, yte, _ = split_data(df_features, target_h)
        models_h, scaler_h = train_models(Xtr, ytr)
        preds_h = predict_future(models_h, scaler_h, df_features, current_price)
        horizon_predictions[h] = preds_h

        months = round(h / 21)
        print(f"  {h}d (~{months}m) Ensemble Target: ₹{preds_h['Ensemble (Avg)']}")

        # Charts for each horizon
        plot_predictions(price_df, models_h, scaler_h,
                         df_features, h, TICKER)

    plot_feature_importance(importances, TICKER)

    return {
        "horizon_predictions": horizon_predictions,
        "metrics":             metrics,
        "importances":         importances,
        "current_price":       current_price,
        "trained_models":      trained_models,
        "scaler":              scaler,
        "df_features":         df_features,
    }
