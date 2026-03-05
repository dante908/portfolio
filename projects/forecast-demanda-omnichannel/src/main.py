from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    import optuna  # type: ignore
except Exception:
    optuna = None


def make_synthetic_demand(start: str = "2022-01-01", end: str = "2025-12-31", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    channels = ["site", "marketplace", "loja_fisica"]
    skus = [f"SKU_{i:02d}" for i in range(1, 13)]

    rows: list[dict[str, Any]] = []
    holiday_md = {
        "01-01",  # ano novo
        "04-21",  # tiradentes
        "05-01",  # dia do trabalho
        "09-07",  # independencia
        "10-12",  # nossa senhora aparecida
        "11-02",  # finados
        "11-15",  # proclamacao da republica
        "12-25",  # natal
    }

    for channel in channels:
        channel_factor = {"site": 1.0, "marketplace": 1.22, "loja_fisica": 0.88}[channel]
        for sku in skus:
            base = rng.uniform(45, 185) * channel_factor
            trend = rng.uniform(-0.005, 0.03)
            week_amp = rng.uniform(0.08, 0.26)
            year_amp = rng.uniform(0.03, 0.14)
            promo_sensitivity = rng.uniform(1.18, 1.55)
            price_base = rng.uniform(25, 480)

            for idx, date in enumerate(dates):
                dow = date.dayofweek
                month = date.month
                day_of_year = date.timetuple().tm_yday

                is_weekend = int(dow >= 5)
                is_holiday = int(date.strftime("%m-%d") in holiday_md)

                # clima sintetico: temperatura + chuva com sazonalidade anual
                temp = 23.0 + 7.0 * np.sin(2 * np.pi * day_of_year / 365.25) + rng.normal(0, 1.8)
                rain = max(0.0, 6.0 + 6.0 * np.cos(2 * np.pi * day_of_year / 365.25) + rng.normal(0, 2.2))

                promo_prob = 0.08 + 0.08 * is_weekend + 0.06 * is_holiday
                is_promo = int(rng.uniform() < promo_prob)

                week_season = 1.0 + week_amp * np.sin(2 * np.pi * dow / 7)
                year_season = 1.0 + year_amp * np.cos(2 * np.pi * day_of_year / 365.25)
                holiday_lift = 1.14 if is_holiday else 1.0
                promo_lift = promo_sensitivity if is_promo else 1.0
                rain_impact = 1.0 - min(0.11, rain / 250.0)

                price = price_base * (1.0 + 0.02 * np.sin(2 * np.pi * month / 12))
                noise = rng.normal(0.0, 4.4)

                demand = (base + trend * idx) * week_season * year_season * holiday_lift * promo_lift * rain_impact + noise
                rows.append(
                    {
                        "date": date,
                        "channel": channel,
                        "sku": sku,
                        "unit_price": round(float(price), 2),
                        "is_promo": is_promo,
                        "is_holiday": is_holiday,
                        "avg_temp_c": round(float(temp), 2),
                        "rainfall_mm": round(float(rain), 2),
                        "demand_units": max(1.0, round(float(demand), 0)),
                    }
                )

    return pd.DataFrame(rows)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().sort_values(["channel", "sku", "date"]).reset_index(drop=True)

    work["dow"] = work["date"].dt.dayofweek
    work["month"] = work["date"].dt.month
    work["week_of_year"] = work["date"].dt.isocalendar().week.astype(int)
    work["day_of_year"] = work["date"].dt.dayofyear
    work["is_weekend"] = (work["dow"] >= 5).astype(int)

    for lag in [1, 7, 14, 28]:
        work[f"lag_{lag}"] = work.groupby(["channel", "sku"], sort=False)["demand_units"].shift(lag)

    grouped = work.groupby(["channel", "sku"], sort=False)["demand_units"]
    work["rolling_mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(window=7).mean())
    work["rolling_mean_28"] = grouped.transform(lambda s: s.shift(1).rolling(window=28).mean())
    work["rolling_std_7"] = grouped.transform(lambda s: s.shift(1).rolling(window=7).std())

    work = work.dropna().reset_index(drop=True)
    return work


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1.0, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def temporal_split(work: pd.DataFrame, test_horizon_days: int = 30, val_horizon_days: int = 90):
    max_date = work["date"].max()
    test_start = max_date - pd.Timedelta(days=test_horizon_days - 1)
    val_start = test_start - pd.Timedelta(days=val_horizon_days)

    train = work[work["date"] < val_start].copy()
    val = work[(work["date"] >= val_start) & (work["date"] < test_start)].copy()
    test = work[work["date"] >= test_start].copy()
    return train, val, test


def generate_reports(backtest: pd.DataFrame) -> list[str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if plt is None:
        return []

    generated: list[str] = []

    # 1) Curva agregada por dia: atual vs previsto
    daily = (
        backtest.groupby("date", as_index=False)[["actual_demand", "forecast_demand", "baseline_lag7"]]
        .sum()
        .sort_values("date")
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(daily["date"], daily["actual_demand"], label="Actual", linewidth=2.0)
    ax.plot(daily["date"], daily["forecast_demand"], label="Forecast", linewidth=1.8)
    ax.plot(daily["date"], daily["baseline_lag7"], label="Baseline lag7", linewidth=1.4, alpha=0.8)
    ax.set_title("Backtest Diario Agregado - Actual vs Forecast")
    ax.set_xlabel("Data")
    ax.set_ylabel("Demanda (unidades)")
    ax.legend()
    ax.grid(alpha=0.25)
    p1 = REPORTS_DIR / "daily_actual_vs_forecast.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    generated.append(p1.name)

    # 2) Distribuicao de erro absoluto
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(backtest["abs_error"], bins=35, alpha=0.85)
    ax.set_title("Distribuicao do Erro Absoluto")
    ax.set_xlabel("Erro absoluto")
    ax.set_ylabel("Frequencia")
    ax.grid(alpha=0.2)
    p2 = REPORTS_DIR / "abs_error_distribution.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    generated.append(p2.name)

    # 3) Top series por volume
    top_series = (
        backtest.groupby(["channel", "sku"], as_index=False)["actual_demand"].sum()
        .sort_values("actual_demand", ascending=False)
        .head(6)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for row in top_series.itertuples(index=False):
        f = backtest[(backtest["channel"] == row.channel) & (backtest["sku"] == row.sku)].sort_values("date")
        ax.plot(f["date"], f["actual_demand"], linewidth=1.2, alpha=0.95, label=f"{row.channel}-{row.sku}")
    ax.set_title("Top 6 Series por Volume - Demanda Real")
    ax.set_xlabel("Data")
    ax.set_ylabel("Demanda")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    p3 = REPORTS_DIR / "top_series_actual_demand.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    generated.append(p3.name)

    return generated


def design_matrix(df: pd.DataFrame, fit_columns: list[str] | None = None):
    y = df["demand_units"].to_numpy(dtype=float)
    x = df.drop(columns=["demand_units"])
    x = x.drop(columns=["date"])
    x = pd.get_dummies(x, columns=["channel", "sku", "dow", "month", "week_of_year"], drop_first=False)

    if fit_columns is not None:
        x = x.reindex(columns=fit_columns, fill_value=0.0)
    else:
        fit_columns = x.columns.tolist()

    return x.to_numpy(dtype=float), y, fit_columns


def fit_ridge_closed_form(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x_aug = np.hstack([np.ones((x.shape[0], 1)), x])
    ident = np.eye(x_aug.shape[1])
    ident[0, 0] = 0.0
    w = np.linalg.solve(x_aug.T @ x_aug + alpha * ident, x_aug.T @ y)
    return w


def predict_ridge(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([np.ones((x.shape[0], 1)), x])
    return x_aug @ w


def tune_xgboost(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int):
    if xgb is None:
        return None, None, None

    if optuna is None:
        params = {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": seed,
            "n_jobs": 1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        return model, params, mape(y_val, pred)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 900),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "reg:squarederror",
            "random_state": seed,
            "n_jobs": 1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        return mape(y_val, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=24, show_progress_bar=False)

    best_params = study.best_params | {
        "objective": "reg:squarederror",
        "random_state": seed,
        "n_jobs": 1,
    }
    model = xgb.XGBRegressor(**best_params)
    model.fit(x_train, y_train)
    best_pred = model.predict(x_val)
    return model, best_params, mape(y_val, best_pred)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = make_synthetic_demand(seed=RANDOM_STATE)
    work_df = build_features(raw_df)

    train_df, val_df, test_df = temporal_split(work_df)

    x_train, y_train, cols = design_matrix(train_df)
    x_val, y_val, _ = design_matrix(val_df, fit_columns=cols)
    x_test, y_test, _ = design_matrix(test_df, fit_columns=cols)

    # baseline simples usando lag semanal
    baseline_pred_test = test_df["lag_7"].to_numpy(dtype=float)

    backend = "ridge_numpy"
    tuned = False
    val_mape_model = None
    model_artifact: dict[str, Any] = {}

    model, best_params, val_mape_xgb = tune_xgboost(x_train, y_train, x_val, y_val, RANDOM_STATE)
    if model is not None:
        backend = "xgboost"
        tuned = optuna is not None
        test_pred = model.predict(x_test)
        val_mape_model = float(val_mape_xgb)
        model_artifact = {
            "backend": backend,
            "tuned_with_optuna": tuned,
            "best_params": best_params,
        }
    else:
        w = fit_ridge_closed_form(x_train, y_train, alpha=1.5)
        test_pred = predict_ridge(x_test, w)
        val_pred = predict_ridge(x_val, w)
        val_mape_model = mape(y_val, val_pred)
        model_artifact = {
            "backend": backend,
            "alpha": 1.5,
            "weights": w.tolist(),
            "feature_columns": cols,
        }

    test_pred = np.clip(test_pred, 0.0, None)
    baseline_pred_test = np.clip(baseline_pred_test, 0.0, None)

    metrics = {
        "validation_mape_model": round(float(val_mape_model), 4),
        "test_model": {
            "mape": round(mape(y_test, test_pred), 4),
            "mae": round(mae(y_test, test_pred), 4),
            "rmse": round(rmse(y_test, test_pred), 4),
        },
        "test_baseline_lag7": {
            "mape": round(mape(y_test, baseline_pred_test), 4),
            "mae": round(mae(y_test, baseline_pred_test), 4),
            "rmse": round(rmse(y_test, baseline_pred_test), 4),
        },
        "backend": backend,
        "optuna_used": tuned,
        "years_of_data": 4,
        "series_count": int(raw_df.groupby(["channel", "sku"]).ngroups),
        "test_horizon_days": 30,
    }

    backtest = test_df[["date", "channel", "sku", "demand_units", "lag_7", "is_promo", "is_holiday", "avg_temp_c", "rainfall_mm"]].copy()
    backtest = backtest.rename(columns={"demand_units": "actual_demand", "lag_7": "baseline_lag7"})
    backtest["forecast_demand"] = np.round(test_pred, 2)
    backtest["abs_error"] = np.round(np.abs(backtest["actual_demand"] - backtest["forecast_demand"]), 2)

    raw_df.to_csv(DATA_DIR / "demand_history_synthetic.csv", index=False)
    backtest.to_csv(DATA_DIR / "forecast_backtest.csv", index=False)
    generated_reports = generate_reports(backtest)

    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        metrics["reports_generated"] = generated_reports
        metrics["matplotlib_available"] = plt is not None
        json.dump(metrics, fp, indent=2)

    with (MODELS_DIR / "model_info.json").open("w", encoding="utf-8") as fp:
        json.dump(model_artifact, fp, indent=2)

    notes = (
        "# Forecast Demanda Omnichannel - Analysis Notes\n\n"
        f"- Backend: {backend}\n"
        f"- Optuna usado: {tuned}\n"
        f"- MAPE validacao (modelo): {metrics['validation_mape_model']}%\n"
        f"- MAPE teste (modelo): {metrics['test_model']['mape']}%\n"
        f"- MAPE teste (baseline lag7): {metrics['test_baseline_lag7']['mape']}%\n"
        f"- Relatorios visuais: {', '.join(generated_reports) if generated_reports else 'nao gerados (matplotlib ausente)'}\n"
        "- Pipeline preparado para execucao semanal e comparacao com baseline.\n"
    )
    (NOTEBOOKS_DIR / "analysis_notes.md").write_text(notes, encoding="utf-8")

    print(f"Backend: {backend}")
    print(f"MAPE validacao (modelo): {metrics['validation_mape_model']}%")
    print(f"MAPE teste (modelo): {metrics['test_model']['mape']}%")
    print(f"MAPE teste (baseline lag7): {metrics['test_baseline_lag7']['mape']}%")


if __name__ == "__main__":
    main()
