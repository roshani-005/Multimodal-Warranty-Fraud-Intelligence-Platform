"""Feature explanations using SHAP when available and a transparent fallback otherwise."""
from __future__ import annotations
import numpy as np

def explain_one(model, feature_names: list[str], values: list[float], top_k: int = 5) -> list[dict]:
    row = np.asarray(values, dtype=float).reshape(1, -1)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        explanation = explainer.shap_values(row)
        if isinstance(explanation, list): explanation = explanation[-1]
        impacts = np.asarray(explanation).reshape(-1)
        return [{"feature": feature_names[i], "impact": round(float(impacts[i]), 5)} for i in np.argsort(np.abs(impacts))[::-1][:top_k]]
    except Exception:
        # Fallback is intentionally described as feature magnitude, not SHAP attribution.
        order = np.argsort(np.abs(row[0]))[::-1][:top_k]
        return [{"feature": feature_names[i], "impact": round(float(row[0][i]), 5), "method": "absolute_feature_magnitude_fallback"} for i in order]
