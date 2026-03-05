from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn_data import make_churn_dataset

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache-churn")

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


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
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


def fit_logistic_regression(x: np.ndarray, y: np.ndarray, lr: float = 0.04, epochs: int = 1700, l2: float = 0.001):
    n_samples, n_features = x.shape
    w = np.zeros(n_features)
    b = 0.0

    pos_weight = (len(y) - y.sum()) / max(1.0, y.sum())
    for _ in range(epochs):
        logits = x @ w + b
        p = sigmoid(logits)

        sample_weight = np.where(y == 1, pos_weight, 1.0)
        err = (p - y) * sample_weight

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
        + np.where(df["billing_cycle"].to_numpy() == "monthly", 0.35, -0.25)
        + np.where(df["plan_type"].to_numpy() == "starter", 0.28, 0.0)
    )
    return sigmoid(linear)


def tune_xgboost(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int):
    if xgb is None:
        return None, None, None

    if optuna is None:
        params = {
            "n_estimators": 450,
            "max_depth": 6,
            "learning_rate": 0.06,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": 1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_val)[:, 1]
        return model, params, roc_auc_score_np(y_val, proba)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": 1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_val)[:, 1]
        return roc_auc_score_np(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    best_params = study.best_params | {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": 1,
    }
    model = xgb.XGBClassifier(**best_params)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_val)[:, 1]
    return model, best_params, roc_auc_score_np(y_val, proba)


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.10, 0.85, 76)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-12, precision + recall)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def make_feature_importance_df(selected_model: str, cols: list[str], model_obj: Any, w: np.ndarray | None) -> pd.DataFrame:
    if selected_model == "xgboost" and model_obj is not None:
        importance = model_obj.feature_importances_
        imp = pd.DataFrame({"feature": cols, "importance": importance})
        return imp.sort_values("importance", ascending=False).head(15)

    if selected_model == "logistic_regression" and w is not None:
        imp = pd.DataFrame({"feature": cols, "importance": np.abs(w)})
        return imp.sort_values("importance", ascending=False).head(15)

    rule_importance = pd.DataFrame(
        {
            "feature": [
                "feature_adoption_score",
                "days_since_last_login",
                "support_tickets_last_90d",
                "late_payments_last_12m",
                "tenure_months",
                "nps_score",
            ],
            "importance": [1.4, 0.06, 0.05, 0.22, 0.015, 0.001],
        }
    )
    return rule_importance.sort_values("importance", ascending=False)


def generate_reports(test_df: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray, importance_df: pd.DataFrame) -> list[str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if plt is None:
        return []

    generated: list[str] = []

    # score distribution by class
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(test_df.loc[test_df["churn"] == 0, "churn_score"], bins=30, alpha=0.75, label="No churn")
    ax.hist(test_df.loc[test_df["churn"] == 1, "churn_score"], bins=30, alpha=0.75, label="Churn")
    ax.set_title("Distribuicao de Score de Churn (Teste)")
    ax.set_xlabel("Churn score")
    ax.set_ylabel("Frequencia")
    ax.legend()
    ax.grid(alpha=0.2)
    p1 = REPORTS_DIR / "score_distribution.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    generated.append(p1.name)

    # confusion matrix
    cm = np.array(
        [
            [int(((y_test == 0) & (y_pred == 0)).sum()), int(((y_test == 0) & (y_pred == 1)).sum())],
            [int(((y_test == 1) & (y_pred == 0)).sum()), int(((y_test == 1) & (y_pred == 1)).sum())],
        ]
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["Real 0", "Real 1"])
    ax.set_title("Matriz de Confusao (Teste)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    p2 = REPORTS_DIR / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    generated.append(p2.name)

    # feature importance
    imp = importance_df.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp["feature"], imp["importance"])
    ax.set_title("Top Features - Importancia")
    ax.set_xlabel("Importancia")
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

    df = make_churn_dataset(n_samples=6000, random_state=RANDOM_STATE)
    x_full, y, numeric_cols = prepare_features(df)

    train_idx, val_idx, test_idx = stratified_split(y, seed=RANDOM_STATE)
    x_train = x_full.iloc[train_idx].copy()
    x_val = x_full.iloc[val_idx].copy()
    x_test = x_full.iloc[test_idx].copy()
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    x_train, x_val, x_test, means, stds = scale_train_val_test(x_train, x_val, x_test, numeric_cols)
    cols = x_train.columns.tolist()

    # Logistic baseline
    w, b = fit_logistic_regression(x_train.to_numpy(dtype=float), y_train)
    val_proba_lr = sigmoid(x_val.to_numpy(dtype=float) @ w + b)
    val_auc_lr = roc_auc_score_np(y_val, val_proba_lr)

    # Rule baseline
    val_proba_rule = score_rule_based(df.iloc[val_idx])
    val_auc_rule = roc_auc_score_np(y_val, val_proba_rule)

    # XGBoost optional
    xgb_model, xgb_params, val_auc_xgb = tune_xgboost(
        x_train.to_numpy(dtype=float),
        y_train,
        x_val.to_numpy(dtype=float),
        y_val,
        RANDOM_STATE,
    )

    candidates = [
        ("logistic_regression", float(val_auc_lr)),
        ("rule_based", float(val_auc_rule)),
    ]
    if val_auc_xgb is not None:
        candidates.append(("xgboost", float(val_auc_xgb)))

    selected_model = max(candidates, key=lambda t: t[1])[0]

    model_artifact: dict[str, Any]
    if selected_model == "xgboost" and xgb_model is not None:
        test_proba = xgb_model.predict_proba(x_test.to_numpy(dtype=float))[:, 1]
        val_proba_selected = xgb_model.predict_proba(x_val.to_numpy(dtype=float))[:, 1]
        model_artifact = {
            "model_type": selected_model,
            "optuna_used": optuna is not None,
            "xgboost_available": True,
            "best_params": xgb_params,
            "columns": cols,
            "numeric_scaler_mean": means,
            "numeric_scaler_std": stds,
        }
    elif selected_model == "logistic_regression":
        test_proba = sigmoid(x_test.to_numpy(dtype=float) @ w + b)
        val_proba_selected = val_proba_lr
        model_artifact = {
            "model_type": selected_model,
            "weights": w.tolist(),
            "bias": b,
            "columns": cols,
            "numeric_scaler_mean": means,
            "numeric_scaler_std": stds,
        }
    else:
        test_proba = score_rule_based(df.iloc[test_idx])
        val_proba_selected = val_proba_rule
        model_artifact = {
            "model_type": selected_model,
            "description": "Rule-based baseline using business risk drivers",
        }

    best_threshold, val_best_f1 = tune_threshold(y_val, val_proba_selected)
    test_pred = (test_proba >= best_threshold).astype(int)

    metrics = classification_metrics(y_test, test_pred, test_proba)
    metrics["model_selected"] = selected_model
    metrics["threshold_selected"] = round(float(best_threshold), 4)
    metrics["validation_best_f1"] = round(float(val_best_f1), 4)
    metrics["val_auc_logistic"] = round(float(val_auc_lr), 4)
    metrics["val_auc_rule"] = round(float(val_auc_rule), 4)
    metrics["val_auc_xgboost"] = round(float(val_auc_xgb), 4) if val_auc_xgb is not None else None
    metrics["xgboost_available"] = xgb is not None
    metrics["optuna_used"] = optuna is not None and selected_model == "xgboost"

    test_df = df.iloc[test_idx].copy()
    test_df["churn_score"] = test_proba
    test_df["churn_pred"] = test_pred

    importance_df = make_feature_importance_df(selected_model, cols, xgb_model, w if selected_model == "logistic_regression" else None)
    report_files = generate_reports(test_df, y_test, test_pred, importance_df)
    metrics["reports_generated"] = report_files
    metrics["matplotlib_available"] = plt is not None

    dataset_path = DATA_DIR / "churn_saas_synthetic.csv"
    scored_path = DATA_DIR / "churn_test_scored.csv"
    model_path = MODELS_DIR / "churn_model.json"
    metrics_path = MODELS_DIR / "metrics.json"
    notebook_path = NOTEBOOKS_DIR / "analysis_notes.md"

    df.to_csv(dataset_path, index=False)
    test_df.to_csv(scored_path, index=False)

    with model_path.open("w", encoding="utf-8") as fp:
        json.dump(model_artifact, fp, indent=2)
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    notebook_text = (
        "# Churn SaaS B2B - Analysis Notes\n\n"
        f"- Selected model: {selected_model}\n"
        f"- ROC-AUC (test): {metrics['roc_auc']}\n"
        f"- F1-score (test): {metrics['f1']}\n"
        f"- Threshold selected: {metrics['threshold_selected']}\n"
        f"- Reports: {', '.join(report_files) if report_files else 'not generated'}\n"
        "- Next step: calibrar threshold por custo de retencao e capacidade operacional do time de CS.\n"
    )
    notebook_path.write_text(notebook_text, encoding="utf-8")

    print(f"Modelo selecionado: {selected_model}")
    print(f"ROC-AUC teste: {metrics['roc_auc']}")
    print(f"F1 teste: {metrics['f1']}")
    print(f"Threshold: {metrics['threshold_selected']}")


if __name__ == "__main__":
    main()
