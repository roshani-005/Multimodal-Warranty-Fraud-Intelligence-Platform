"""Run the full synthetic data -> features -> model evaluation workflow."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .data import generate_claims
from .features import build_features
from .modeling import train_compare

def run(rows: int = 5000, seed: int = 42, output: str = "data/generated") -> dict:
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    claims = generate_claims(rows, seed)
    features = build_features(claims, include_graph=True)
    claims.to_csv(out / "claims.csv", index=False)
    features.to_csv(out / "features.csv", index=False)
    metrics = train_compare(claims, features, output_dir="artifacts", seed=seed)
    print(f"Generated {len(claims):,} synthetic claims; label rate={claims.fraud_label.mean():.3%}")
    print(pd.DataFrame(metrics).T[["pr_auc", "roc_auc", "precision_at_0_5", "recall_at_0_5", "f1_at_0_5"]].to_string())
    print("Metrics saved to artifacts/metrics.json; treat these as synthetic-demo results.")
    return metrics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/generated")
    args = parser.parse_args()
    run(args.rows, args.seed, args.output)
if __name__ == "__main__": main()
