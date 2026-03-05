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


def make_employee_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dept = rng.choice(["sales", "engineering", "operations", "hr", "finance"], size=n, p=[0.24, 0.32, 0.22, 0.1, 0.12])
    seniority = rng.choice(["junior", "mid", "senior", "lead"], size=n, p=[0.25, 0.45, 0.23, 0.07])
    work_mode = rng.choice(["remote", "hybrid", "onsite"], size=n, p=[0.28, 0.50, 0.22])

    tenure_months = rng.integers(1, 160, size=n)
    salary_market_ratio = np.clip(rng.normal(0.95, 0.15, size=n), 0.6, 1.5)
    engagement_score = np.clip(rng.normal(0.66, 0.2, size=n), 0.0, 1.0)
    overtime_hours_month = np.clip(rng.normal(14, 9, size=n), 0, 75)
    manager_changes_24m = rng.poisson(0.8, size=n)
    absences_last_12m = rng.poisson(4.5, size=n)
    performance_score = np.clip(rng.normal(3.4, 0.8, size=n), 1.0, 5.0)
    years_since_promotion = np.clip(tenure_months / 12 - rng.normal(1.8, 1.2, size=n), 0, 15)

    linear = (
        -1.45
        + 0.025 * overtime_hours_month
        + 0.22 * manager_changes_24m
        + 0.07 * absences_last_12m
        - 2.0 * engagement_score
        - 0.95 * (salary_market_ratio - 1.0)
        - 0.22 * performance_score
        + 0.12 * years_since_promotion
        - 0.006 * tenure_months
        + np.where(work_mode == "onsite", 0.18, 0.0)
        + np.where(dept == "sales", 0.12, 0.0)
    )
    attrition_prob = sigmoid(linear)
    attrition = (rng.uniform(size=n) < attrition_prob).astype(int)

    return pd.DataFrame(
        {
            "department": dept,
            "seniority": seniority,
            "work_mode": work_mode,
            "tenure_months": tenure_months,
            "salary_market_ratio": salary_market_ratio,
            "engagement_score": engagement_score,
            "overtime_hours_month": overtime_hours_month,
            "manager_changes_24m": manager_changes_24m,
            "absences_last_12m": absences_last_12m,
            "performance_score": performance_score,
            "years_since_promotion": years_since_promotion,
            "attrition": attrition,
        }
    )


def fit_logreg(x: np.ndarray, y: np.ndarray, lr: float = 0.06, epochs: int = 1600, l2: float = 0.0005):
    w = np.zeros(x.shape[1])
    b = 0.0
    pos_weight = (len(y) - y.sum()) / max(1.0, y.sum())
    for _ in range(epochs):
        p = sigmoid(x @ w + b)
        sample_weight = np.where(y == 1, pos_weight, 1.0)
        err = (p - y) * sample_weight
        w -= lr * ((x.T @ err) / len(y) + l2 * w)
        b -= lr * err.mean()
    return w, b


def build_features(df: pd.DataFrame):
    y = df["attrition"].astype(int).to_numpy()
    x = df.drop(columns=["attrition"]).copy()
    x = pd.get_dummies(x, columns=["department", "seniority", "work_mode"], drop_first=False)
    return x, y


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
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
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    df = make_employee_dataset(seed=RANDOM_STATE)
    x, y = build_features(df)

    idx = np.arange(len(df))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(idx)

    n_train = int(len(idx) * 0.7)
    n_val = int(len(idx) * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    x_train = x.iloc[train_idx].copy()
    x_val = x.iloc[val_idx].copy()
    x_test = x.iloc[test_idx].copy()

    means = x_train.mean()
    stds = x_train.std().replace(0, 1.0)
    x_train = (x_train - means) / stds
    x_val = (x_val - means) / stds
    x_test = (x_test - means) / stds

    w, b = fit_logreg(x_train.to_numpy(dtype=float), y[train_idx])

    val_score = sigmoid(x_val.to_numpy(dtype=float) @ w + b)
    thresholds = np.linspace(0.1, 0.8, 71)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        pred = (val_score >= t).astype(int)
        tp = ((y[val_idx] == 1) & (pred == 1)).sum()
        fp = ((y[val_idx] == 0) & (pred == 1)).sum()
        fn = ((y[val_idx] == 1) & (pred == 0)).sum()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-12, precision + recall)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    test_score = sigmoid(x_test.to_numpy(dtype=float) @ w + b)
    test_pred = (test_score >= best_t).astype(int)
    m = metrics(y[test_idx], test_pred, test_score)
    m["selected_threshold"] = round(best_t, 4)

    scored = df.iloc[test_idx].copy()
    scored["attrition_score"] = test_score
    scored["attrition_pred"] = test_pred

    df.to_csv(DATA_DIR / "employees_synthetic.csv", index=False)
    scored.to_csv(DATA_DIR / "employees_test_scored.csv", index=False)

    with (MODELS_DIR / "model.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "model_type": "logistic_regression_numpy",
                "columns": x.columns.tolist(),
                "weights": w.tolist(),
                "bias": b,
                "threshold": best_t,
            },
            fp,
            indent=2,
        )
    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(m, fp, indent=2)

    notes = (
        "# People Analytics Turnover - Analysis Notes\n\n"
        f"- ROC-AUC teste: {m['roc_auc']}\n"
        f"- F1 teste: {m['f1']}\n"
        f"- Threshold: {m['selected_threshold']}\n"
    )
    (NOTEBOOKS_DIR / "analysis_notes.md").write_text(notes, encoding="utf-8")

    print(f"ROC-AUC teste: {m['roc_auc']}")
    print(f"F1 teste: {m['f1']}")


if __name__ == "__main__":
    main()
