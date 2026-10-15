"""Leakage-aware tabular and graph feature preparation."""
from __future__ import annotations
import pandas as pd
from .graph import graph_features
TARGET = "fraud_label"
ID_COLUMNS = ["claim_id", "customer_id", "device_id", "retailer_id", "repair_center_id", "payment_account_id", "claim_date"]
NUMERIC_BASE = ["claim_amount", "device_age_months", "days_since_purchase", "claims_last_90d"]

def build_features(df: pd.DataFrame, include_graph: bool = True) -> pd.DataFrame:
    required = set(ID_COLUMNS + NUMERIC_BASE + [TARGET])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = df[["claim_id"] + NUMERIC_BASE].copy()
    result["amount_per_device_month"] = result["claim_amount"] / (result["device_age_months"] + 1)
    result["recent_purchase_claim"] = (df["days_since_purchase"] <= 45).astype(int)
    result["new_device_claim"] = (df["device_age_months"] <= 6).astype(int)
    result["claim_amount_log"] = __import__("numpy").log1p(result["claim_amount"].clip(lower=0))
    if include_graph:
        result = result.merge(graph_features(df), on="claim_id", how="left", validate="one_to_one")
    return result.fillna(0)

def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c != "claim_id"]
