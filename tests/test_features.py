import pandas as pd
from claimgraph.data import generate_claims
from claimgraph.features import build_features

def test_generator_is_reproducible():
    a = generate_claims(200, 7)
    b = generate_claims(200, 7)
    pd.testing.assert_frame_equal(a, b)

def test_features_exclude_target_and_preserve_rows():
    claims = generate_claims(200, 7)
    features = build_features(claims)
    assert len(features) == len(claims)
    assert "fraud_label" not in features.columns
    assert features["claim_id"].is_unique
    assert features.select_dtypes(include="number").notna().all().all()
