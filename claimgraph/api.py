"""FastAPI service for graph-enhanced warranty claim risk scoring."""
from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .explain import explain_one

app = FastAPI(title="ClaimGraph API", version="0.1.0", description="Synthetic-demo warranty claim risk scoring. Scores support human review; they are not proof of fraud.")
class ClaimInput(BaseModel):
    claim_amount: float = Field(ge=0, le=100000)
    device_age_months: int = Field(ge=0, le=240)
    claims_last_90d: int = Field(ge=0, le=1000)
    days_since_purchase: int = Field(ge=0, le=10000)
    repair_center_claims_30d: int = Field(default=0, ge=0)
    shared_payment_claims: int = Field(default=0, ge=0)
    customer_claims_180d: int = Field(default=0, ge=0)
    device_claims_180d: int = Field(default=0, ge=0)
    retailer_claims_30d: int = Field(default=0, ge=0)
    graph_degree: int = Field(default=5, ge=0)
    component_size: int = Field(default=1, ge=1)
    community_claim_rate: float = Field(default=0.0, ge=0, le=1)

@app.get("/health")
def health(): return {"status": "ok", "service": "claimgraph"}

@app.post("/score")
def score(payload: ClaimInput):
    artifact = Path("artifacts/model.joblib")
    if not artifact.exists():
        raise HTTPException(status_code=503, detail="Model artifact missing. Run python -m claimgraph.pipeline first.")
    bundle = joblib.load(artifact)
    raw = payload.model_dump()
    raw["amount_per_device_month"] = raw["claim_amount"] / (raw["device_age_months"] + 1)
    raw["recent_purchase_claim"] = int(raw["days_since_purchase"] <= 45)
    raw["new_device_claim"] = int(raw["device_age_months"] <= 6)
    import numpy as np
    raw["claim_amount_log"] = float(np.log1p(raw["claim_amount"]))
    # Align to the model's training columns; IDs/component identifiers are not required by the API model.
    row = pd.DataFrame([{c: raw.get(c, 0) for c in bundle["features"]}])
    probability = float(bundle["model"].predict_proba(row)[0, 1])
    threshold = float(bundle.get("threshold", 0.5))
    explanation = explain_one(bundle["model"], bundle["features"], row.iloc[0].astype(float).tolist())
    return {"risk_score": round(probability, 4), "review_recommended": probability >= threshold, "threshold": threshold, "explanation": explanation, "disclaimer": "Synthetic-demo score for prioritizing human review; not proof of fraud."}
