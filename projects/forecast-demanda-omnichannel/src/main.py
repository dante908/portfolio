from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"
RANDOM_STATE = 42


def make_synthetic_demand(start: str = "2024-01-01", end: str = "2025-12-31", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    channels = ["site", "marketplace", "loja_fisica"]
    skus = [f"SKU_{i:02d}" for i in range(1, 13)]

    rows = []
    for channel in channels:
        channel_factor = {"site": 1.0, "marketplace": 1.25, "loja_fisica": 0.85}[channel]
        for sku in skus:
            base = rng.uniform(30, 140) * channel_factor
            trend = rng.uniform(-0.02, 0.08)
            season_amp = rng.uniform(5, 25)
            price = rng.uniform(35, 320)

            for i, d in enumerate(dates):
                dow = d.dayofweek
                week_season = 1.0 + (season_amp / 100.0) * np.sin(2 * np.pi * dow / 7)
                year_pos = d.timetuple().tm_yday
                year_season = 1.0 + 0.08 * np.cos(2 * np.pi * year_pos / 365)
                promo = int(rng.uniform() < 0.08)
                promo_lift = 1.35 if promo else 1.0
                noise = rng.normal(0, 4)

                demand = (base + trend * i) * week_season * year_season * promo_lift + noise
                rows.append(
                    {
                        "date": d,
                        "channel": channel,
                        "sku": sku,
                        "unit_price": round(price, 2),
                        "is_promo": promo,
                        "demand_units": max(1, round(demand, 0)),
                    }
                )

    return pd.DataFrame(rows)


def forecast_series(train_values: np.ndarray, horizon: int) -> np.ndarray:
    # Seasonal naive (weekly) + linear trend from recent window
    recent = train_values[-56:] if len(train_values) >= 56 else train_values
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0] if len(recent) > 1 else 0.0

    preds = []
    for h in range(1, horizon + 1):
        seasonal_base = train_values[-7 + ((h - 1) % 7)] if len(train_values) >= 7 else train_values[-1]
        pred = seasonal_base + slope * h
        preds.append(max(0.0, pred))
    return np.array(preds)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1.0, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    df = make_synthetic_demand(seed=RANDOM_STATE)
    df = df.sort_values(["channel", "sku", "date"]).reset_index(drop=True)

    horizon = 30
    forecasts = []

    for (channel, sku), grp in df.groupby(["channel", "sku"], sort=False):
        values = grp["demand_units"].to_numpy(dtype=float)
        dates = grp["date"].to_numpy()

        train_values = values[:-horizon]
        test_values = values[-horizon:]
        test_dates = dates[-horizon:]

        pred = forecast_series(train_values, horizon=horizon)

        for d, y, yhat in zip(test_dates, test_values, pred):
            forecasts.append(
                {
                    "date": pd.Timestamp(d),
                    "channel": channel,
                    "sku": sku,
                    "actual_demand": round(float(y), 2),
                    "forecast_demand": round(float(yhat), 2),
                    "abs_error": round(abs(float(y - yhat)), 2),
                }
            )

    fcst_df = pd.DataFrame(forecasts)

    overall_mae = mae(fcst_df["actual_demand"].to_numpy(), fcst_df["forecast_demand"].to_numpy())
    overall_mape = mape(fcst_df["actual_demand"].to_numpy(), fcst_df["forecast_demand"].to_numpy())

    channel_metrics_rows = []
    for channel, g in fcst_df.groupby("channel"):
        channel_metrics_rows.append(
            {
                "channel": channel,
                "mae": round(mae(g["actual_demand"].to_numpy(), g["forecast_demand"].to_numpy()), 4),
                "mape": round(mape(g["actual_demand"].to_numpy(), g["forecast_demand"].to_numpy()), 4),
            }
        )
    channel_metrics = pd.DataFrame(channel_metrics_rows)

    metrics = {
        "overall_mae": round(overall_mae, 4),
        "overall_mape": round(overall_mape, 4),
        "series_count": int(df.groupby(["channel", "sku"]).ngroups),
        "forecast_horizon_days": horizon,
        "channel_metrics": channel_metrics.to_dict(orient="records"),
    }

    df.to_csv(DATA_DIR / "demand_history_synthetic.csv", index=False)
    fcst_df.to_csv(DATA_DIR / "forecast_backtest.csv", index=False)

    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    model_info = {
        "method": "seasonal_naive_plus_trend",
        "seasonality_days": 7,
        "trend_window_days": 56,
    }
    with (MODELS_DIR / "model_info.json").open("w", encoding="utf-8") as fp:
        json.dump(model_info, fp, indent=2)

    notes = (
        "# Forecast Demanda Omnichannel - Analysis Notes\n\n"
        f"- MAE geral: {metrics['overall_mae']}\n"
        f"- MAPE geral: {metrics['overall_mape']}%\n"
        "- Backtest realizado nos ultimos 30 dias por serie channel+sku.\n"
    )
    (NOTEBOOKS_DIR / "analysis_notes.md").write_text(notes, encoding="utf-8")

    print(f"MAE geral: {metrics['overall_mae']}")
    print(f"MAPE geral: {metrics['overall_mape']}%")
    print(f"Arquivos gerados em {DATA_DIR}, {MODELS_DIR} e {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    main()
