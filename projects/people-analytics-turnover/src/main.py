from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache-people")

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


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
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
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


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


def build_features(df: pd.DataFrame):
    y = df["attrition"].astype(int).to_numpy()
    x = df.drop(columns=["attrition"]).copy()
    x = pd.get_dummies(x, columns=["department", "seniority", "work_mode"], drop_first=False)
    return x, y


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


def tune_xgboost(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int):
    if xgb is None:
        return None, None, None

    if optuna is None:
        params = {
            "n_estimators": 420,
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
            "n_estimators": trial.suggest_int("n_estimators", 180, 700),
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
        return roc_auc_score_np(y_val, model.predict_proba(x_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=16, show_progress_bar=False)

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
    thresholds = np.linspace(0.1, 0.8, 71)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((y_true == 1) & (pred == 1)).sum()
        fp = ((y_true == 0) & (pred == 1)).sum()
        fn = ((y_true == 1) & (pred == 0)).sum()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-12, precision + recall)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def generate_reports(test_df: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray, importance_df: pd.DataFrame) -> list[str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if plt is None:
        return []

    generated: list[str] = []

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(test_df.loc[test_df["attrition"] == 0, "attrition_score"], bins=30, alpha=0.75, label="Stay")
    ax.hist(test_df.loc[test_df["attrition"] == 1, "attrition_score"], bins=30, alpha=0.75, label="Attrition")
    ax.set_title("Distribuicao de Score de Turnover (Teste)")
    ax.set_xlabel("Attrition score")
    ax.set_ylabel("Frequencia")
    ax.legend()
    ax.grid(alpha=0.2)
    p1 = REPORTS_DIR / "score_distribution.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    generated.append(p1.name)

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

    # baseline logreg
    w, b = fit_logreg(x_train.to_numpy(dtype=float), y[train_idx])
    val_logreg = sigmoid(x_val.to_numpy(dtype=float) @ w + b)
    auc_logreg = roc_auc_score_np(y[val_idx], val_logreg)

    # optional advanced model
    xgb_model, xgb_params, auc_xgb = tune_xgboost(
        x_train.to_numpy(dtype=float),
        y[train_idx],
        x_val.to_numpy(dtype=float),
        y[val_idx],
        RANDOM_STATE,
    )

    selected_model = "logistic_regression"
    y_val_score = val_logreg
    if auc_xgb is not None and auc_xgb >= auc_logreg:
        selected_model = "xgboost"
        y_val_score = xgb_model.predict_proba(x_val.to_numpy(dtype=float))[:, 1]

    best_t, best_f1_val = tune_threshold(y[val_idx], y_val_score)

    if selected_model == "xgboost":
        test_score = xgb_model.predict_proba(x_test.to_numpy(dtype=float))[:, 1]
        importance = pd.DataFrame(
            {"feature": x.columns.tolist(), "importance": xgb_model.feature_importances_}
        ).sort_values("importance", ascending=False).head(15)
    else:
        test_score = sigmoid(x_test.to_numpy(dtype=float) @ w + b)
        importance = pd.DataFrame(
            {"feature": x.columns.tolist(), "importance": np.abs(w)}
        ).sort_values("importance", ascending=False).head(15)

    test_pred = (test_score >= best_t).astype(int)
    m = metrics(y[test_idx], test_pred, test_score)
    m["selected_threshold"] = round(best_t, 4)
    m["val_f1_best"] = round(best_f1_val, 4)
    m["selected_model"] = selected_model
    m["val_auc_logreg"] = round(float(auc_logreg), 4)
    m["val_auc_xgboost"] = round(float(auc_xgb), 4) if auc_xgb is not None else None
    m["used_optuna"] = bool(optuna is not None and xgb is not None)
    m["xgboost_available"] = bool(xgb is not None)

    scored = df.iloc[test_idx].copy()
    scored["attrition_score"] = test_score
    scored["attrition_pred"] = test_pred

    report_files = generate_reports(scored, y[test_idx], test_pred, importance)
    m["reports_generated"] = len(report_files)

    df.to_csv(DATA_DIR / "employees_synthetic.csv", index=False)
    scored.to_csv(DATA_DIR / "employees_test_scored.csv", index=False)

    with (MODELS_DIR / "model.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "model_type": selected_model,
                "columns": x.columns.tolist(),
                "weights": w.tolist(),
                "bias": b,
                "xgboost_params": xgb_params,
                "threshold": best_t,
            },
            fp,
            indent=2,
        )
    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(m, fp, indent=2)

    notes = (
        "# People Analytics Turnover - Analysis Notes\n\n"
        f"- Modelo selecionado: {selected_model}\n"
        f"- ROC-AUC teste: {m['roc_auc']}\n"
        f"- F1 teste: {m['f1']}\n"
        f"- Recall teste: {m['recall']}\n"
        f"- Threshold selecionado: {m['selected_threshold']}\n"
        f"- ROC-AUC validacao (logreg): {m['val_auc_logreg']}\n"
        f"- ROC-AUC validacao (xgboost): {m['val_auc_xgboost']}\n"
        f"- Graficos em reports/: {', '.join(report_files) if report_files else 'nao gerado'}\n"
    )
    (NOTEBOOKS_DIR / "analysis_notes.md").write_text(notes, encoding="utf-8")

    print(f"Modelo selecionado: {selected_model}")
    print(f"ROC-AUC teste: {m['roc_auc']}")
    print(f"F1 teste: {m['f1']}")


if __name__ == "__main__":
    main()
