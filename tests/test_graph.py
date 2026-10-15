from claimgraph.data import generate_claims
from claimgraph.graph import build_entity_graph, graph_features

def test_graph_contains_claim_and_entity_nodes():
    claims = generate_claims(200, 2)
    graph = build_entity_graph(claims)
    assert graph.number_of_nodes() > len(claims)
    assert graph.number_of_edges() >= len(claims) * 5

def test_graph_features_are_one_row_per_claim():
    claims = generate_claims(200, 2)
    features = graph_features(claims)
    assert len(features) == len(claims)
    assert features.claim_id.is_unique
    assert (features.component_size >= 1).all()
