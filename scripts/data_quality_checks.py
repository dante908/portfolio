from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import pandera.pandas as pa
    from pandera import Check
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "pandera nao instalado. Rode: python3 -m pip install pandera"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"


def validate_churn() -> None:
    p = PROJECTS / "churn-saas-b2b" / "data" / "churn_saas_synthetic.csv"
    df = pd.read_csv(p)
    schema = pa.DataFrameSchema(
        {
            "monthly_fee": pa.Column(float, Check.ge(0)),
            "num_seats": pa.Column(int, Check.ge(1)),
            "tenure_months": pa.Column(int, Check.ge(1)),
            "feature_adoption_score": pa.Column(float, [Check.ge(0), Check.le(1)]),
            "churn": pa.Column(int, Check.isin([0, 1])),
        },
        coerce=True,
    )
    schema.validate(df)


def validate_forecast() -> None:
    p = PROJECTS / "forecast-demanda-omnichannel" / "data" / "demand_history_synthetic.csv"
    df = pd.read_csv(p)
    schema = pa.DataFrameSchema(
        {
            "unit_price": pa.Column(float, Check.ge(0)),
            "is_promo": pa.Column(int, Check.isin([0, 1])),
            "is_holiday": pa.Column(int, Check.isin([0, 1])),
            "avg_temp_c": pa.Column(float),
            "rainfall_mm": pa.Column(float, Check.ge(0)),
            "demand_units": pa.Column(float, Check.ge(1)),
        },
        coerce=True,
    )
    schema.validate(df)


def validate_fraud() -> None:
    p = PROJECTS / "fraude-pagamentos-rtr" / "data" / "transactions_synthetic.csv"
    df = pd.read_csv(p)
    schema = pa.DataFrameSchema(
        {
            "amount": pa.Column(float, Check.ge(0)),
            "hour": pa.Column(int, [Check.ge(0), Check.le(23)]),
            "device_risk": pa.Column(float, [Check.ge(0), Check.le(1)]),
            "merchant_risk": pa.Column(float, [Check.ge(0), Check.le(1)]),
            "is_fraud": pa.Column(int, Check.isin([0, 1])),
        },
        coerce=True,
    )
    schema.validate(df)


def validate_people() -> None:
    p = PROJECTS / "people-analytics-turnover" / "data" / "employees_synthetic.csv"
    df = pd.read_csv(p)
    schema = pa.DataFrameSchema(
        {
            "tenure_months": pa.Column(int, Check.ge(1)),
            "engagement_score": pa.Column(float, [Check.ge(0), Check.le(1)]),
            "salary_market_ratio": pa.Column(float, Check.ge(0)),
            "attrition": pa.Column(int, Check.isin([0, 1])),
        },
        coerce=True,
    )
    schema.validate(df)


def validate_recommendation() -> None:
    p = PROJECTS / "recomendacao-ecommerce" / "data" / "interactions_synthetic.csv"
    df = pd.read_csv(p)
    schema = pa.DataFrameSchema(
        {
            "user_id": pa.Column(str),
            "item_id": pa.Column(str),
            "event_type": pa.Column(str, Check.isin(["view", "cart", "purchase"])),
            "event_weight": pa.Column(float, Check.ge(0)),
        },
        coerce=True,
    )
    schema.validate(df)


def validate_segmentation() -> None:
    p = PROJECTS / "segmentacao-rfm-clustering" / "data" / "rfm_clusters.csv"
    df = pd.read_csv(p)
    schema = pa.DataFrameSchema(
        {
            "customer_id": pa.Column(str),
            "recency_days": pa.Column(int, Check.ge(0)),
            "frequency": pa.Column(int, Check.ge(1)),
            "monetary": pa.Column(float, Check.ge(0)),
            "cluster": pa.Column(int),
        },
        coerce=True,
    )
    schema.validate(df)


def main() -> None:
    validate_churn()
    validate_forecast()
    validate_fraud()
    validate_people()
    validate_recommendation()
    validate_segmentation()
    print("Data quality checks passed for all projects.")


if __name__ == "__main__":
    main()
