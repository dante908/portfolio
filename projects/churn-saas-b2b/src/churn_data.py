from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_churn_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    plan_type = rng.choice(["starter", "growth", "enterprise"], size=n_samples, p=[0.45, 0.40, 0.15])
    billing_cycle = rng.choice(["monthly", "annual"], size=n_samples, p=[0.62, 0.38])
    industry = rng.choice(
        ["fintech", "healthcare", "retail", "edtech", "logistics"],
        size=n_samples,
        p=[0.18, 0.17, 0.27, 0.20, 0.18],
    )
    region = rng.choice(["north_america", "latam", "europe"], size=n_samples, p=[0.45, 0.30, 0.25])

    num_seats = rng.integers(3, 450, size=n_samples)
    tenure_months = rng.integers(1, 84, size=n_samples)
    support_tickets_last_90d = rng.poisson(lam=3.0, size=n_samples)
    days_since_last_login = rng.integers(0, 45, size=n_samples)
    feature_adoption_score = np.clip(rng.normal(loc=0.62, scale=0.18, size=n_samples), 0.0, 1.0)
    nps_score = np.clip(rng.normal(loc=28, scale=24, size=n_samples), -100, 100)
    late_payments_last_12m = rng.poisson(lam=1.2, size=n_samples)

    base_monthly_fee = np.where(plan_type == "starter", 99, np.where(plan_type == "growth", 349, 1200))
    monthly_fee = base_monthly_fee + (num_seats * rng.uniform(1.5, 5.5, size=n_samples))
    monthly_fee = np.round(monthly_fee, 2)

    logits = (
        -0.8
        + 0.05 * support_tickets_last_90d
        + 0.06 * days_since_last_login
        - 1.4 * feature_adoption_score
        - 0.015 * tenure_months
        - 0.01 * (nps_score / 10.0)
        + 0.22 * late_payments_last_12m
        + np.where(billing_cycle == "monthly", 0.35, -0.25)
        + np.where(plan_type == "starter", 0.28, np.where(plan_type == "enterprise", -0.12, 0.0))
        + np.where(region == "latam", 0.10, 0.0)
    )

    churn_probability = _sigmoid(logits)
    churn = (rng.uniform(0, 1, size=n_samples) < churn_probability).astype(int)

    df = pd.DataFrame(
        {
            "plan_type": plan_type,
            "billing_cycle": billing_cycle,
            "industry": industry,
            "region": region,
            "monthly_fee": monthly_fee,
            "num_seats": num_seats,
            "tenure_months": tenure_months,
            "support_tickets_last_90d": support_tickets_last_90d,
            "days_since_last_login": days_since_last_login,
            "feature_adoption_score": feature_adoption_score,
            "nps_score": nps_score,
            "late_payments_last_12m": late_payments_last_12m,
            "churn": churn,
        }
    )
    return df
