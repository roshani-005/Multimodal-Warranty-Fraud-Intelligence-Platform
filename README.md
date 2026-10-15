# ClaimGraph: Multimodal Warranty Fraud Intelligence Platform

ClaimGraph is an end-to-end portfolio project for identifying suspicious warranty claims by combining tabular claim signals with a Neo4j-style entity graph. It models relationships among claims, customers, devices, retailers, repair centres and payment accounts.

> **Data notice:** The repository starts with reproducible synthetic data. It contains no real customer or insurer data. Metrics produced from synthetic data are demonstration results, not evidence of real-world fraud performance.

## Why this project

Warranty fraud is rarely visible in one row alone. A claim might look ordinary in isolation but become suspicious when its device, customer, payment account or repair centre is repeatedly linked to claims flagged for review. ClaimGraph combines:
- Synthetic claim-level data with realistic risk patterns.
- Entity resolution and graph features such as shared-account claim counts, degree, connected components and community identifiers.
- A baseline XGBoost-compatible classifier and graph-enhanced model comparison.
- PR-AUC, recall, precision, F1 and confusion-matrix reporting for imbalanced labels.
- SHAP-based explanations when SHAP is installed, with a model-agnostic fallback.
- A FastAPI scoring service, Docker setup, tests and a reproducible CLI pipeline.

## Architecture

```text
Synthetic data generator
        |
        v
Data validation + entity normalization
        |
        +----------------------+
        |                      |
        v                      v
Tabular features         Entity relationship graph
(claim, customer,        (customer, device, retailer,
repair, payment)          repair centre, payment)
        |                      |
        +-----------+----------+
                    v
       Baseline and graph-enhanced models
                    |
                    v
       Evaluation + explanation + API
```

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m claimgraph.pipeline --rows 5000 --output data/generated
pytest -q
```

Run the API after generating a model:

```bash
uvicorn claimgraph.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Pipeline outputs

The CLI writes generated claims, engineered features, model metrics and a serialized model under `data/generated/` and `artifacts/`. Generated files are ignored by Git to keep the repository small and reproducible.

## API example

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "claim_amount": 850,
    "device_age_months": 4,
    "claims_last_90d": 3,
    "days_since_purchase": 25,
    "repair_center_claims_30d": 18,
    "shared_payment_claims": 4,
    "customer_claims_180d": 3,
    "device_claims_180d": 2,
    "retailer_claims_30d": 12,
    "graph_degree": 7,
    "component_size": 5,
    "community_claim_rate": 0.45
  }'
```

The API expects a trained artifact at `artifacts/model.joblib`. Run the pipeline first.

## Project structure

```text
claimgraph/
  __init__.py
  data.py          # synthetic data generation
  features.py      # tabular and relationship feature engineering
  graph.py         # NetworkX entity graph and graph features
  modeling.py      # model training, comparison and metrics
  explain.py       # SHAP/fallback feature explanations
  pipeline.py      # reproducible end-to-end CLI
  api.py           # FastAPI scoring endpoint
tests/
  test_features.py
  test_graph.py
  test_api.py
Dockerfile
requirements.txt
```

## Evaluation design

- Use a stratified train/test split and keep the target out of features.
- Compare a tabular baseline against the graph-enhanced model on the same split.
- Report PR-AUC alongside recall and precision because fraud labels are imbalanced.
- Choose a review threshold based on investigation capacity and the cost of missed fraud.
- Avoid claiming graph features improve performance unless the measured holdout results support it.
- For a production deployment, split by time and monitor data drift, alert volume, recall on matured labels and subgroup performance.

## Neo4j extension

The local demo uses NetworkX so it runs without a database server. `cypher/schema.cypher` includes example constraints and relationship patterns for loading entities into Neo4j. The same entity model supports a Neo4j-backed deployment.

## Limitations and responsible use

This is a learning and portfolio system, not an automated fraud verdict engine. A high score indicates a claim deserves human review; it is not proof of wrongdoing. Synthetic labels encode intentionally designed patterns, so results must not be interpreted as real-world accuracy. Before real deployment, use authorized data, temporal validation, calibration, fairness testing, privacy controls, audit logging and human review.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, NetworkX, Neo4j/Cypher extension, SHAP, FastAPI, Pydantic, Docker, pytest.

## Resume-ready project description

**ClaimGraph: Multimodal Warranty Fraud Intelligence Platform** — Built a reproducible fraud analytics pipeline combining warranty-claim features with customer, device, retailer, repair-centre and payment-account relationships; engineered graph-risk signals, compared baseline and graph-enhanced classifiers using PR-AUC and recall, and exposed claim risk scoring with FastAPI, Docker and explainability.

Only add specific performance numbers to a resume after running the pipeline and confirming the generated evaluation report.
