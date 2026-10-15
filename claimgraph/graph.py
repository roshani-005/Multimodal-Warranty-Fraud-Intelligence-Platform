"""Entity graph construction and per-claim graph features."""
from __future__ import annotations
import networkx as nx
import pandas as pd
ENTITY_COLUMNS = ["customer_id", "device_id", "retailer_id", "repair_center_id", "payment_account_id"]

def build_entity_graph(df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in df.itertuples(index=False):
        claim = f"claim:{row.claim_id}"
        graph.add_node(claim, node_type="claim")
        for col in ENTITY_COLUMNS:
            value = getattr(row, col)
            if pd.notna(value):
                entity = f"{col.removesuffix('_id')}:{value}"
                graph.add_node(entity, node_type=col.removesuffix("_id"))
                graph.add_edge(claim, entity, relation=f"linked_{col.removesuffix('_id')}")
    return graph

def graph_features(df: pd.DataFrame) -> pd.DataFrame:
    graph = build_entity_graph(df)
    degrees = dict(graph.degree())
    components = {}
    for component_id, nodes in enumerate(nx.connected_components(graph)):
        for node in nodes:
            components[node] = (component_id, len(nodes))
    rows = []
    for row in df.itertuples(index=False):
        claim = f"claim:{row.claim_id}"
        neighbours = list(graph.neighbors(claim))
        rows.append({
            "claim_id": row.claim_id,
            "graph_degree": int(degrees.get(claim, 0)),
            "component_id": int(components.get(claim, (-1, 1))[0]),
            "component_size": int(components.get(claim, (-1, 1))[1]),
            "shared_payment_claims": max(0, degrees.get(f"payment_account:{row.payment_account_id}", 1) - 1),
            "repair_center_claims_30d": max(0, degrees.get(f"repair_center:{row.repair_center_id}", 1) - 1),
            "retailer_claims_30d": max(0, degrees.get(f"retailer:{row.retailer_id}", 1) - 1),
            "customer_claims_180d": max(0, degrees.get(f"customer:{row.customer_id}", 1) - 1),
            "device_claims_180d": max(0, degrees.get(f"device:{row.device_id}", 1) - 1),
        })
    return pd.DataFrame(rows)
