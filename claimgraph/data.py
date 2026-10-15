"""Reproducible synthetic warranty-claim data for development and demos."""
from __future__ import annotations
import numpy as np
import pandas as pd


def generate_claims(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("rows must be at least 100 for a useful demo dataset")
    rng = np.random.default_rng(seed)
    n_customers = max(80, rows // 3)
    n_devices = max(100, int(rows * 0.78))
    n_retailers = max(12, rows // 100)
    n_centres = max(10, rows // 120)
    n_payments = max(70, int(rows * 0.60))
    customer = rng.integers(0, n_customers, rows)
    device = rng.integers(0, n_devices, rows)
    retailer = rng.integers(0, n_retailers, rows)
    repair = rng.integers(0, n_centres, rows)
    payment = rng.integers(0, n_payments, rows)
    amount = np.clip(rng.lognormal(5.6, 0.65, rows), 35, 5000).round(2)
    device_age = rng.integers(0, 72, rows)
    days_since_purchase = rng.integers(0, 1500, rows)
    claims_90 = rng.poisson(0.7, rows)
    base_logit = -3.0 + 0.018 * claims_90 + 0.00025 * amount + 0.75 * (days_since_purchase < 45) + 0.55 * (device_age < 6)
    # Plant repeat-entity risk patterns so graph methods have a learnable signal.
    hot_payment = rng.choice(n_payments, size=max(3, n_payments // 30), replace=False)
    hot_centres = rng.choice(n_centres, size=max(2, n_centres // 12), replace=False)
    suspicious = np.isin(payment, hot_payment) | np.isin(repair, hot_centres)
    base_logit += suspicious * 1.7
    probability = 1 / (1 + np.exp(-base_logit))
    fraud = rng.binomial(1, probability)
    claim_date = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, rows), unit="D")
    return pd.DataFrame({
        "claim_id": [f"CLM{i:07d}" for i in range(rows)],
        "claim_date": claim_date,
        "customer_id": [f"C{v:05d}" for v in customer],
        "device_id": [f"D{v:05d}" for v in device],
        "retailer_id": [f"R{v:04d}" for v in retailer],
        "repair_center_id": [f"RC{v:04d}" for v in repair],
        "payment_account_id": [f"P{v:05d}" for v in payment],
        "claim_amount": amount,
        "device_age_months": device_age,
        "days_since_purchase": days_since_purchase,
        "claims_last_90d": claims_90,
        "fraud_label": fraud.astype(int),
    })
