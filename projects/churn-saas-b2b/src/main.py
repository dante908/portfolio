from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn_data import make_churn_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
NOTEBOOKS_DIR = ROOT / "notebooks"
RANDOM_STATE = 42


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = (2 * precision * recall) / max(1e-12, (precision + recall))

    return {
        "roc_auc": round(roc_auc_score_np(y_true, y_score), 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def stratified_split(y: np.ndarray, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    train_idx, val_idx, test_idx = [], [], []

    for cls in [0, 1]:
        cls_idx = indices[y == cls]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train : n_train + n_val])
        test_idx.append(cls_idx[n_train + n_val :])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def prepare_features(df: pd.DataFrame):
    target_col = "churn"
    numeric_cols = [
        "monthly_fee",
        "num_seats",
        "tenure_months",
        "support_tickets_last_90d",
        "days_since_last_login",
        "feature_adoption_score",
        "nps_score",
        "late_payments_last_12m",
    ]
    categorical_cols = ["plan_type", "billing_cycle", "industry", "region"]

    x_num = df[numeric_cols].copy()
    x_cat = pd.get_dummies(df[categorical_cols], drop_first=False)
    x_full = pd.concat([x_num, x_cat], axis=1)
    y = df[target_col].astype(int).to_numpy()

    return x_full, y, numeric_cols


def scale_train_val_test(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame, numeric_cols: list[str]):
    means = x_train[numeric_cols].mean()
    stds = x_train[numeric_cols].std().replace(0, 1.0)

    for x in [x_train, x_val, x_test]:
        x[numeric_cols] = (x[numeric_cols] - means) / stds

    return x_train, x_val, x_test, means.to_dict(), stds.to_dict()


def fit_logistic_regression(x: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 1200, l2: float = 0.001):
    n_samples, n_features = x.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        logits = x @ w + b
        p = sigmoid(logits)
        err = p - y

        grad_w = (x.T @ err) / n_samples + l2 * w
        grad_b = err.mean()

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def score_rule_based(df: pd.DataFrame) -> np.ndarray:
    linear = (
        -0.8
        + 0.05 * df["support_tickets_last_90d"].to_numpy()
        + 0.06 * df["days_since_last_login"].to_numpy()
        - 1.4 * df["feature_adoption_score"].to_numpy()
        - 0.015 * df["tenure_months"].to_numpy()
        - 0.001 * df["nps_score"].to_numpy()
        + 0.22 * df["late_payments_last_12m"].to_numpy()
    )
    return sigmoid(linear)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    df = make_churn_dataset(n_samples=5000, random_state=RANDOM_STATE)
    x_full, y, numeric_cols = prepare_features(df)

    train_idx, val_idx, test_idx = stratified_split(y, seed=RANDOM_STATE)
    x_train = x_full.iloc[train_idx].copy()
    x_val = x_full.iloc[val_idx].copy()
    x_test = x_full.iloc[test_idx].copy()
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    x_train, x_val, x_test, means, stds = scale_train_val_test(x_train, x_val, x_test, numeric_cols)

    cols = x_train.columns.tolist()
    w, b = fit_logistic_regression(x_train.to_numpy(dtype=float), y_train)

    val_proba_lr = sigmoid(x_val.to_numpy(dtype=float) @ w + b)
    val_auc_lr = roc_auc_score_np(y_val, val_proba_lr)

    val_proba_rule = score_rule_based(df.iloc[val_idx])
    val_auc_rule = roc_auc_score_np(y_val, val_proba_rule)

    if val_auc_lr >= val_auc_rule:
        selected_model = "logistic_regression"
        test_proba = sigmoid(x_test.to_numpy(dtype=float) @ w + b)
        model_artifact = {
            "model_type": selected_model,
            "weights": w.tolist(),
            "bias": b,
            "columns": cols,
            "numeric_scaler_mean": means,
            "numeric_scaler_std": stds,
        }
    else:
        selected_model = "rule_based"
        test_proba = score_rule_based(df.iloc[test_idx])
        model_artifact = {
            "model_type": selected_model,
            "description": "Rule-based baseline using business risk drivers",
        }

    test_pred = (test_proba >= 0.5).astype(int)
    metrics = classification_metrics(y_test, test_pred, test_proba)
    metrics["model_selected"] = selected_model
    metrics["val_auc_logistic"] = round(float(val_auc_lr), 4)
    metrics["val_auc_rule"] = round(float(val_auc_rule), 4)

    dataset_path = DATA_DIR / "churn_saas_synthetic.csv"
    scored_path = DATA_DIR / "churn_test_scored.csv"
    model_path = MODELS_DIR / "churn_model.json"
    metrics_path = MODELS_DIR / "metrics.json"
    notebook_path = NOTEBOOKS_DIR / "analysis_notes.md"

    df.to_csv(dataset_path, index=False)
    scored_df = df.iloc[test_idx].copy()
    scored_df["churn_score"] = test_proba
    scored_df["churn_pred"] = test_pred
    scored_df.to_csv(scored_path, index=False)

    with model_path.open("w", encoding="utf-8") as fp:
        json.dump(model_artifact, fp, indent=2)
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    notebook_text = (
        "# Churn SaaS B2B - Analysis Notes\n\n"
        f"- Selected model: {selected_model}\n"
        f"- ROC-AUC (test): {metrics['roc_auc']}\n"
        f"- F1-score (test): {metrics['f1']}\n"
        "- Next step: calibrar threshold por custo de retention campaign.\n"
    )
    notebook_path.write_text(notebook_text, encoding="utf-8")

    print(f"Modelo selecionado: {selected_model}")
    print(f"ROC-AUC teste: {metrics['roc_auc']}")
    print(f"F1 teste: {metrics['f1']}")
    print(f"Arquivos gerados em {DATA_DIR}, {MODELS_DIR} e {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    main()
