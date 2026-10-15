// Optional Neo4j schema for the ClaimGraph entity model.
CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (n:Claim) REQUIRE n.claim_id IS UNIQUE;
CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (n:Customer) REQUIRE n.customer_id IS UNIQUE;
CREATE CONSTRAINT device_id IF NOT EXISTS FOR (n:Device) REQUIRE n.device_id IS UNIQUE;
CREATE CONSTRAINT retailer_id IF NOT EXISTS FOR (n:Retailer) REQUIRE n.retailer_id IS UNIQUE;
CREATE CONSTRAINT repair_center_id IF NOT EXISTS FOR (n:RepairCenter) REQUIRE n.repair_center_id IS UNIQUE;
CREATE CONSTRAINT payment_account_id IF NOT EXISTS FOR (n:PaymentAccount) REQUIRE n.payment_account_id IS UNIQUE;

// Example relationship pattern after loading entity nodes and claims:
// MATCH (c:Claim {claim_id: $claim_id}), (p:PaymentAccount {payment_account_id: $payment_account_id})
// MERGE (c)-[:PAID_FROM]->(p);

// Example investigation query: shared payment accounts linked to multiple claims.
MATCH (p:PaymentAccount)<-[:PAID_FROM]-(c:Claim)
WITH p, collect(c) AS claims
WHERE size(claims) >= 3
RETURN p.payment_account_id AS payment_account, size(claims) AS linked_claims
ORDER BY linked_claims DESC;
