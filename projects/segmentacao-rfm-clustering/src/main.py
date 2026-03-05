from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache-seg")

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


def kmeans(x: np.ndarray, k: int = 5, max_iter: int = 70, seed: int = 42):
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), size=k, replace=False)]

    inertia_prev = None
    for _ in range(max_iter):
        dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists, axis=1)

        new_centroids = np.vstack(
            [x[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)]
        )

        inertia = float(np.sum((x - new_centroids[labels]) ** 2))
        centroids = new_centroids
        if inertia_prev is not None and abs(inertia_prev - inertia) < 1e-7:
            break
        inertia_prev = inertia

    return labels, centroids, float(inertia_prev if inertia_prev is not None else 0.0)


def silhouette_score_np(x: np.ndarray, labels: np.ndarray) -> float:
    n = len(x)
    unique = np.unique(labels)
    if len(unique) < 2 or n < 3:
        return 0.0

    # pairwise euclidean distance matrix
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    s_vals = []

    for i in range(n):
        own = labels[i]
        own_mask = labels == own
        own_mask[i] = False

        a = float(d[i, own_mask].mean()) if np.any(own_mask) else 0.0

        b_candidates = []
        for other in unique:
            if other == own:
                continue
            other_mask = labels == other
            if np.any(other_mask):
                b_candidates.append(float(d[i, other_mask].mean()))
        b = min(b_candidates) if b_candidates else 0.0

        denom = max(a, b)
        s_vals.append((b - a) / denom if denom > 0 else 0.0)

    return float(np.mean(s_vals))


def assign_personas(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary.copy()
    s["score"] = (
        -s["recency_days"].rank(pct=True)
        + s["frequency"].rank(pct=True)
        + s["monetary"].rank(pct=True)
    )
    s = s.sort_values("score", ascending=False).reset_index(drop=True)

    names = ["Champions", "Loyal", "Potential", "At Risk", "Hibernating", "Dormant"]
    s["persona"] = [names[i] if i < len(names) else f"Segment_{i+1}" for i in range(len(s))]
    return s[["cluster", "persona"]]


def find_best_k(x_scaled: np.ndarray, candidate_ks: list[int], seed: int = 42):
    best = None
    all_results = []
    for k in candidate_ks:
        labels, centroids, inertia = kmeans(x_scaled, k=k, seed=seed)
        sil = silhouette_score_np(x_scaled, labels)
        row = {
            "k": int(k),
            "silhouette": round(float(sil), 4),
            "inertia": round(float(inertia), 4),
            "labels": labels,
            "centroids": centroids,
        }
        all_results.append(row)
        if best is None or row["silhouette"] > best["silhouette"]:
            best = row
    return best, all_results


def generate_reports(rfm: pd.DataFrame, summary: pd.DataFrame, k_eval: list[dict]) -> list[str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if plt is None:
        return []

    generated: list[str] = []

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(summary["persona"], summary["customers"])
    ax.set_title("Tamanho por Segmento")
    ax.set_xlabel("Persona")
    ax.set_ylabel("Clientes")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    p1 = REPORTS_DIR / "cluster_sizes.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    generated.append(p1.name)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for persona, g in rfm.groupby("persona"):
        ax.scatter(g["frequency"], g["monetary"], s=10, alpha=0.25, label=persona)
    ax.set_title("Frequencia x Monetary por Segmento")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Monetary")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    p2 = REPORTS_DIR / "frequency_monetary_scatter.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    generated.append(p2.name)

    eval_df = pd.DataFrame(k_eval)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(eval_df["k"], eval_df["silhouette"], marker="o", label="Silhouette")
    ax.set_title("Selecao de K por Silhouette")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.grid(alpha=0.25)
    ax.legend()
    p3 = REPORTS_DIR / "k_selection_silhouette.png"
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
    x_np = x_scaled.to_numpy(dtype=float)

    best_k, k_eval = find_best_k(x_np, candidate_ks=[4, 5, 6], seed=RANDOM_STATE)
    labels = best_k["labels"].astype(int)
    centroids = best_k["centroids"]

    rfm["cluster"] = labels

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

    report_files = generate_reports(rfm, summary, [{"k": r["k"], "silhouette": r["silhouette"], "inertia": r["inertia"]} for r in k_eval])

    with (MODELS_DIR / "model_info.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "algorithm": "kmeans_numpy",
                "selected_k": int(best_k["k"]),
                "features": ["recency_days", "frequency", "log1p_monetary"],
                "centroids": centroids.tolist(),
            },
            fp,
            indent=2,
        )

    largest = summary.sort_values("customers", ascending=False).iloc[0]
    champions_count = int(summary.loc[summary["persona"] == "Champions", "customers"].sum())

    metrics = {
        "selected_k": int(best_k["k"]),
        "silhouette": round(float(best_k["silhouette"]), 4),
        "inertia": round(float(best_k["inertia"]), 4),
        "customers_segmented": int(len(rfm)),
        "largest_segment": {
            "persona": str(largest["persona"]),
            "customers": int(largest["customers"]),
        },
        "champions_customers": champions_count,
        "k_candidates": [{"k": int(r["k"]), "silhouette": float(r["silhouette"]), "inertia": float(r["inertia"])} for r in k_eval],
        "reports_generated": len(report_files),
    }

    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    notes = (
        "# Segmentacao RFM - Analysis Notes\n\n"
        f"- K selecionado: {metrics['selected_k']}\n"
        f"- Silhouette: {metrics['silhouette']}\n"
        f"- Clientes segmentados: {metrics['customers_segmented']}\n"
        f"- Maior segmento: {metrics['largest_segment']['persona']} ({metrics['largest_segment']['customers']} clientes)\n"
        f"- Champions: {metrics['champions_customers']} clientes\n"
        f"- Graficos em reports/: {', '.join(report_files) if report_files else 'nao gerado'}\n"
    )
    (NOTEBOOKS_DIR / "analysis_notes.md").write_text(notes, encoding="utf-8")

    print(f"K selecionado: {metrics['selected_k']}")
    print(f"Silhouette: {metrics['silhouette']}")
    print(f"Clientes segmentados: {metrics['customers_segmented']}")


if __name__ == "__main__":
    main()
