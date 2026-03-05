from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache-fraud")

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    import optuna  # type: ignore
except Exception:
    optuna = None

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = float(ranks[y_true == 1].sum())
    return float((sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def make_transactions(n: int = 42000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="min")

    amount = np.round(np.exp(rng.normal(4.45, 1.02, size=n)), 2)
    hour = np.array([d.hour for d in dates])
    is_night = ((hour <= 5) | (hour >= 23)).astype(int)
    velocity_1h = rng.poisson(1.5, size=n)
    device_risk = np.clip(rng.beta(2.0, 5.0, size=n), 0, 1)
    merchant_risk = np.clip(rng.beta(2.3, 3.6, size=n), 0, 1)
    is_international = (rng.uniform(size=n) < 0.18).astype(int)
    card_age_days = rng.integers(1, 2500, size=n)
    chargeback_history = (rng.uniform(size=n) < 0.09).astype(int)
    amount_over_1k = (amount > 1000).astype(int)

    linear = (
        -4.2
        + 0.0024 * amount
        + 0.53 * is_night
        + 0.30 * velocity_1h
        + 2.4 * device_risk
        + 1.55 * merchant_risk
        + 0.92 * is_international
        - 0.00045 * card_age_days
        + 1.15 * chargeback_history
        + 0.35 * amount_over_1k
    )
    fraud_prob = sigmoid(linear)
    is_fraud = (rng.uniform(size=n) < fraud_prob).astype(int)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "amount": amount,
            "hour": hour,
            "velocity_1h": velocity_1h,
            "device_risk": np.round(device_risk, 4),
            "merchant_risk": np.round(merchant_risk, 4),
            "is_international": is_international,
            "card_age_days": card_age_days,
            "chargeback_history": chargeback_history,
            "amount_over_1k": amount_over_1k,
            "is_fraud": is_fraud,
        }
    )


def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-12:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)


def compute_risk_score(df: pd.DataFrame) -> np.ndarray:
    s = (
        0.28 * minmax(df["amount"].to_numpy(dtype=float))
        + 0.08 * df["hour"].isin([0, 1, 2, 3, 4, 5, 23]).astype(float).to_numpy()
        + 0.15 * minmax(df["velocity_1h"].to_numpy(dtype=float))
        + 0.20 * df["device_risk"].to_numpy(dtype=float)
        + 0.12 * df["merchant_risk"].to_numpy(dtype=float)
        + 0.09 * df["is_international"].to_numpy(dtype=float)
        + 0.05 * (1.0 - minmax(df["card_age_days"].to_numpy(dtype=float)))
        + 0.03 * df["amount_over_1k"].to_numpy(dtype=float)
    )
    return np.clip(s, 0.0, 1.0)


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, len(y_true))

    return {
        "roc_auc": round(roc_auc_score_np(y_true, y_score), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def select_threshold(y_val: np.ndarray, score_val: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.15, 0.9, 76)
    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        pred = (score_val >= t).astype(int)
        tp = ((y_val == 1) & (pred == 1)).sum()
        fp = ((y_val == 0) & (pred == 1)).sum()
        fn = ((y_val == 1) & (pred == 0)).sum()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-12, precision + recall)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def build_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [
        "amount",
        "hour",
        "velocity_1h",
        "device_risk",
        "merchant_risk",
        "is_international",
        "card_age_days",
        "chargeback_history",
        "amount_over_1k",
    ]
    return df[cols].to_numpy(dtype=float)


def tune_xgb(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    if xgb is None:
        return None, None, None

    if optuna is None:
        params = {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.07,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train)
        val_score = model.predict_proba(x_val)[:, 1]
        return model, params, roc_auc_score_np(y_val, val_score)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train)
        val_score = model.predict_proba(x_val)[:, 1]
        return roc_auc_score_np(y_val, val_score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=18, show_progress_bar=False)
    params = study.best_params | {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train)
    val_score = model.predict_proba(x_val)[:, 1]
    return model, params, roc_auc_score_np(y_val, val_score)


def generate_reports(scored: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, feature_importance: pd.DataFrame) -> list[str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if plt is None:
        return []

    generated: list[str] = []

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(scored.loc[scored["is_fraud"] == 0, "risk_score"], bins=35, alpha=0.75, label="No fraud")
    ax.hist(scored.loc[scored["is_fraud"] == 1, "risk_score"], bins=35, alpha=0.75, label="Fraud")
    ax.set_title("Distribuicao de Risk Score")
    ax.set_xlabel("risk_score")
    ax.legend()
    ax.grid(alpha=0.2)
    p1 = REPORTS_DIR / "score_distribution.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    generated.append(p1.name)

    cm = np.array(
        [
            [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
            [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["Real 0", "Real 1"])
    ax.set_title("Matriz de Confusao")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    p2 = REPORTS_DIR / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    generated.append(p2.name)

    imp = feature_importance.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["importance"])
    ax.set_title("Feature Importance")
    ax.grid(axis="x", alpha=0.2)
    p3 = REPORTS_DIR / "feature_importance.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    generated.append(p3.name)

    return generated


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = make_transactions(seed=RANDOM_STATE).sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    score_val_rule = compute_risk_score(val_df)
    score_test_rule = compute_risk_score(test_df)

    x_train = build_matrix(train_df)
    x_val = build_matrix(val_df)
    x_test = build_matrix(test_df)
    y_train = train_df["is_fraud"].to_numpy(dtype=int)
    y_val = val_df["is_fraud"].to_numpy(dtype=int)
    y_test = test_df["is_fraud"].to_numpy(dtype=int)

    xgb_model, xgb_params, xgb_val_auc = tune_xgb(x_train, y_train, x_val, y_val)
    rule_val_auc = roc_auc_score_np(y_val, score_val_rule)

    if xgb_model is not None and xgb_val_auc is not None and xgb_val_auc >= rule_val_auc:
        selected_model = "xgboost"
        score_val = xgb_model.predict_proba(x_val)[:, 1]
        score_test = xgb_model.predict_proba(x_test)[:, 1]
        feature_importance = pd.DataFrame(
            {
                "feature": [
                    "amount",
                    "hour",
                    "velocity_1h",
                    "device_risk",
                    "merchant_risk",
                    "is_international",
                    "card_age_days",
                    "chargeback_history",
                    "amount_over_1k",
                ],
                "importance": xgb_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
    else:
        selected_model = "rule_based"
        score_val = score_val_rule
        score_test = score_test_rule
        feature_importance = pd.DataFrame(
            {
                "feature": [
                    "amount",
                    "device_risk",
                    "velocity_1h",
                    "merchant_risk",
                    "is_international",
                    "card_age_days",
                    "hour",
                ],
                "importance": [0.28, 0.20, 0.15, 0.12, 0.09, 0.05, 0.08],
            }
        ).sort_values("importance", ascending=False)

    threshold, val_f1 = select_threshold(y_val, score_val)
    pred_test = (score_test >= threshold).astype(int)

    metrics = metrics_from_preds(y_test, pred_test, score_test)
    metrics["selected_threshold"] = round(float(threshold), 4)
    metrics["validation_f1"] = round(float(val_f1), 4)
    metrics["model_selected"] = selected_model
    metrics["val_auc_rule"] = round(float(rule_val_auc), 4)
    metrics["val_auc_xgboost"] = round(float(xgb_val_auc), 4) if xgb_val_auc is not None else None
    metrics["xgboost_available"] = xgb is not None
    metrics["optuna_used"] = optuna is not None and selected_model == "xgboost"

    scored = test_df.copy()
    scored["risk_score"] = np.round(score_test, 6)
    scored["pred_fraud"] = pred_test

    report_files = generate_reports(scored, y_test, pred_test, feature_importance.head(15))
    metrics["reports_generated"] = report_files
    metrics["matplotlib_available"] = plt is not None

    df.to_csv(DATA_DIR / "transactions_synthetic.csv", index=False)
    scored.to_csv(DATA_DIR / "transactions_test_scored.csv", index=False)

    with (MODELS_DIR / "model_info.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "model": selected_model,
                "selected_threshold": threshold,
                "xgboost_params": xgb_params if selected_model == "xgboost" else None,
                "features": [
                    "amount",
                    "hour",
                    "velocity_1h",
                    "device_risk",
                    "merchant_risk",
                    "is_international",
                    "card_age_days",
                    "chargeback_history",
                    "amount_over_1k",
                ],
            },
            fp,
            indent=2,
        )

    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    notes = (
        "# Fraude Pagamentos RTR - Analysis Notes\n\n"
        f"- Modelo selecionado: {selected_model}\n"
        f"- Threshold selecionado: {metrics['selected_threshold']}\n"
        f"- ROC-AUC teste: {metrics['roc_auc']}\n"
        f"- Recall teste: {metrics['recall']}\n"
        f"- Reports: {', '.join(report_files) if report_files else 'not generated'}\n"
    )
    (NOTEBOOKS_DIR / "analysis_notes.md").write_text(notes, encoding="utf-8")

    print(f"Modelo: {selected_model}")
    print(f"Threshold selecionado: {metrics['selected_threshold']}")
    print(f"ROC-AUC teste: {metrics['roc_auc']}")
    print(f"F1 teste: {metrics['f1']}")


if __name__ == "__main__":
    main()
