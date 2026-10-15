"""Train comparable baseline and graph-enhanced fraud classifiers."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import feature_columns

def _model(seed: int = 42):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=180, max_depth=4, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=seed, n_jobs=2)
    except ImportError:
        return RandomForestClassifier(n_estimators=220, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)

def train_compare(df: pd.DataFrame, features: pd.DataFrame, output_dir: str = "artifacts", seed: int = 42) -> dict:
    merged = features.merge(df[["claim_id", "fraud_label"]], on="claim_id", validate="one_to_one")
    y = merged["fraud_label"].astype(int)
    baseline_cols = [c for c in feature_columns(features) if c not in {"graph_degree", "component_id", "component_size", "shared_payment_claims", "repair_center_claims_30d", "retailer_claims_30d", "customer_claims_180d", "device_claims_180d"}]
    all_cols = feature_columns(features)
    train_idx, test_idx = train_test_split(np.arange(len(merged)), test_size=0.25, random_state=seed, stratify=y)
    reports = {}
    models = {}
    for name, cols in [("baseline", baseline_cols), ("graph_enhanced", all_cols)]:
        model = _model(seed)
        model.fit(merged.iloc[train_idx][cols], y.iloc[train_idx])
        probabilities = model.predict_proba(merged.iloc[test_idx][cols])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        reports[name] = {
            "pr_auc": round(float(average_precision_score(y.iloc[test_idx], probabilities)), 4),
            "roc_auc": round(float(roc_auc_score(y.iloc[test_idx], probabilities)), 4),
            "precision_at_0_5": round(float(precision_score(y.iloc[test_idx], predictions, zero_division=0)), 4),
            "recall_at_0_5": round(float(recall_score(y.iloc[test_idx], predictions, zero_division=0)), 4),
            "f1_at_0_5": round(float(f1_score(y.iloc[test_idx], predictions, zero_division=0)), 4),
            "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y.iloc[test_idx], predictions, labels=[0, 1]).ravel().tolist(),
            "features": cols,
        }
        models[name] = (model, cols)
    # Save graph-enhanced model used by the API.
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    model, cols = models["graph_enhanced"]
    joblib.dump({"model": model, "features": cols, "threshold": 0.5}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports
