[orchestrator] DAG 'capstone_support_platform' execution order: ['generate_data', 'ingest', 'quality_gate', 'lakehouse', 'rag_demo']

[orchestrator] ---- running task 'generate_data' (depends_on=[]) ----
[orchestrator] task 'generate_data' SUCCESS (0.004s)

[orchestrator] ---- running task 'ingest' (depends_on=['generate_data']) ----
[ingestion] producer publishing 200 events to topic 'support-tickets'
[ingestion] consumer accepted 200 events, rejected 0 at schema validation
[orchestrator] task 'ingest' SUCCESS (0.504s)

[orchestrator] ---- running task 'quality_gate' (depends_on=['ingest']) ----
[quality_gate] PASS_WITH_WARNINGS — 180/200 rows passed (90.0%). Failures by dimension: {'completeness': 8, 'validity': 8, 'accuracy': 0, 'uniqueness': 7}
[orchestrator] task 'quality_gate' SUCCESS (0.005s)

[orchestrator] ---- running task 'lakehouse' (depends_on=['quality_gate']) ----
[lakehouse] gold table written: 6 product rows (6 upserted this run) -> data/gold/product_stats_gold.json
[orchestrator] task 'lakehouse' SUCCESS (0.000s)

[orchestrator] ---- running task 'rag_demo' (depends_on=['generate_data', 'lakehouse']) ----
[rag] indexed 6 chunks from 6 KB articles
[rag] query='My Laptop Pro 14 shipment is late, what should I do?' -> top match KB-002 (score=0.2112)
[rag] query='Customer wants a refund for a defective product' -> top match KB-003 (score=0.5182)
[rag] query='I can't login to my account, how do I fix it?' -> top match KB-004 (score=0.1959)
[orchestrator] task 'rag_demo' SUCCESS (0.007s)
