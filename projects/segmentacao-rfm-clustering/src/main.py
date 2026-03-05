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


def make_transactions(n_customers: int = 1600, n_tx: int = 38000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customer_ids = [f"C{i:05d}" for i in range(1, n_customers + 1)]

    customer_value = rng.gamma(shape=2.0, scale=1.0, size=n_customers)
    chosen_customers = rng.choice(n_customers, size=n_tx, p=customer_value / customer_value.sum())

    start = pd.Timestamp("2025-01-01")
    days = rng.integers(0, 365, size=n_tx)
    dates = start + pd.to_timedelta(days, unit="D")

    amounts = np.round(np.exp(rng.normal(4.2, 0.9, size=n_tx)), 2)

    return pd.DataFrame(
        {
            "customer_id": [customer_ids[i] for i in chosen_customers],
            "order_date": dates,
            "order_amount": amounts,
        }
    ).sort_values("order_date")


def kmeans(x: np.ndarray, k: int = 5, max_iter: int = 60, seed: int = 42):
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), size=k, replace=False)]

    for _ in range(max_iter):
        dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists, axis=1)

        new_centroids = np.vstack([
            x[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < 1e-6:
            break

    return labels, centroids


def assign_personas(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary.copy()
    s["score"] = (
        -s["recency_days"].rank(pct=True)
        + s["frequency"].rank(pct=True)
        + s["monetary"].rank(pct=True)
    )
    s = s.sort_values("score", ascending=False).reset_index(drop=True)

    names = ["Champions", "Loyal", "Potential", "At Risk", "Hibernating"]
    s["persona"] = [names[i] if i < len(names) else f"Segment_{i+1}" for i in range(len(s))]
    return s[["cluster", "persona"]]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    tx = make_transactions(seed=RANDOM_STATE)

    snapshot_date = tx["order_date"].max() + pd.Timedelta(days=1)
    rfm = (
        tx.groupby("customer_id")
        .agg(
            recency_days=("order_date", lambda s: (snapshot_date - s.max()).days),
            frequency=("order_date", "count"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    x = rfm[["recency_days", "frequency", "monetary"]].copy()
    x["monetary"] = np.log1p(x["monetary"])
    x_scaled = (x - x.mean()) / x.std().replace(0, 1.0)

    labels, centroids = kmeans(x_scaled.to_numpy(dtype=float), k=5, seed=RANDOM_STATE)
    rfm["cluster"] = labels.astype(int)

    summary = (
        rfm.groupby("cluster")
        .agg(
            customers=("customer_id", "count"),
            recency_days=("recency_days", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
        .reset_index()
    )

    personas = assign_personas(summary)
    rfm = rfm.merge(personas, on="cluster", how="left")
    summary = summary.merge(personas, on="cluster", how="left")

    tx.to_csv(DATA_DIR / "transactions_synthetic.csv", index=False)
    rfm.to_csv(DATA_DIR / "rfm_clusters.csv", index=False)
    summary.to_csv(MODELS_DIR / "cluster_summary.csv", index=False)

    with (MODELS_DIR / "model_info.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "algorithm": "kmeans_numpy",
                "k": 5,
                "features": ["recency_days", "frequency", "log1p_monetary"],
                "centroids": centroids.tolist(),
            },
            fp,
            indent=2,
        )

    notes = "# Segmentacao RFM - Analysis Notes\n\n" + "\n".join(
        [
            f"- {row.persona}: {int(row.customers)} clientes"
            for row in summary.sort_values("customers", ascending=False).itertuples(index=False)
        ]
    )
    (NOTEBOOKS_DIR / "analysis_notes.md").write_text(notes + "\n", encoding="utf-8")

    print("Clusters gerados com sucesso.")
    print(summary[["cluster", "persona", "customers"]].to_string(index=False))


if __name__ == "__main__":
    main()
