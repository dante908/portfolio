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


def make_transactions(n: int = 40000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="min")

    amount = np.round(np.exp(rng.normal(4.4, 1.0, size=n)), 2)
    hour = np.array([d.hour for d in dates])
    is_night = ((hour <= 5) | (hour >= 23)).astype(int)
    velocity_1h = rng.poisson(1.4, size=n)
    device_risk = np.clip(rng.beta(2, 5, size=n), 0, 1)
    merchant_risk = np.clip(rng.beta(2.5, 3.5, size=n), 0, 1)
    is_international = (rng.uniform(size=n) < 0.18).astype(int)
    card_age_days = rng.integers(1, 2500, size=n)
    chargeback_history = (rng.uniform(size=n) < 0.09).astype(int)

    linear = (
        -4.1
        + 0.0025 * amount
        + 0.55 * is_night
        + 0.32 * velocity_1h
        + 2.3 * device_risk
        + 1.6 * merchant_risk
        + 0.85 * is_international
        - 0.0005 * card_age_days
        + 1.2 * chargeback_history
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
        0.30 * minmax(df["amount"].to_numpy(dtype=float))
        + 0.08 * df["hour"].isin([0, 1, 2, 3, 4, 5, 23]).astype(float).to_numpy()
        + 0.16 * minmax(df["velocity_1h"].to_numpy(dtype=float))
        + 0.20 * df["device_risk"].to_numpy(dtype=float)
        + 0.12 * df["merchant_risk"].to_numpy(dtype=float)
        + 0.08 * df["is_international"].to_numpy(dtype=float)
        + 0.06 * (1.0 - minmax(df["card_age_days"].to_numpy(dtype=float)))
    )
    return np.clip(s, 0.0, 1.0)


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
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
    candidates = np.linspace(0.2, 0.9, 71)
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


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    df = make_transactions(seed=RANDOM_STATE)
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    score_train = compute_risk_score(train_df)
    score_val = compute_risk_score(val_df)
    score_test = compute_risk_score(test_df)

    threshold, val_f1 = select_threshold(val_df["is_fraud"].to_numpy(), score_val)

    pred_test = (score_test >= threshold).astype(int)
    metrics = metrics_from_preds(test_df["is_fraud"].to_numpy(), pred_test, score_test)
    metrics["selected_threshold"] = round(float(threshold), 4)
    metrics["validation_f1"] = round(float(val_f1), 4)

    scored = test_df.copy()
    scored["risk_score"] = np.round(score_test, 6)
    scored["pred_fraud"] = pred_test

    df.to_csv(DATA_DIR / "transactions_synthetic.csv", index=False)
    scored.to_csv(DATA_DIR / "transactions_test_scored.csv", index=False)

    with (MODELS_DIR / "model_info.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "model": "rule_based_risk_scoring",
                "selected_threshold": threshold,
                "features": [
                    "amount",
                    "hour",
                    "velocity_1h",
                    "device_risk",
                    "merchant_risk",
                    "is_international",
                    "card_age_days",
                ],
            },
            fp,
            indent=2,
        )

    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    notes = (
        "# Fraude Pagamentos RTR - Analysis Notes\n\n"
        f"- Threshold selecionado: {metrics['selected_threshold']}\n"
        f"- ROC-AUC teste: {metrics['roc_auc']}\n"
        f"- Recall teste: {metrics['recall']}\n"
    )
    (NOTEBOOKS_DIR / "analysis_notes.md").write_text(notes, encoding="utf-8")

    print(f"Threshold selecionado: {metrics['selected_threshold']}")
    print(f"ROC-AUC teste: {metrics['roc_auc']}")
    print(f"F1 teste: {metrics['f1']}")


if __name__ == "__main__":
    main()
