# Real-Time Customer Support Intelligence Platform
**Capstone Project — Modern Data Engineering for AI Systems**

This project integrates the five things the course covered — ingestion,
a bronze/silver/gold lakehouse, data quality gating, RAG, and pipeline
orchestration — into one working, run-it-yourself Python pipeline.

It simulates a support desk system: customer tickets stream in, get
validated, get cleaned, get rolled up into business metrics, and a RAG
system answers support questions using a knowledge base + those metrics.

## Why it's built this way

The course slides recommend an enterprise stack (Kafka, Delta Lake,
Airflow, Qdrant, Great Expectations, an LLM API). Those need paid
accounts, servers, or internet access to install/run. This project
implements the **same architecture and the same algorithms** those tools
provide, in plain Python, so it runs anywhere with just
`pip install -r requirements.txt` — no cloud accounts, no Docker, no API
keys. Every module's docstring explains exactly which real tool it stands
in for and how to swap it in. That trade-off is intentional and is called
out below and in the code — I'd rather submit something that actually
runs and that I can explain, than something that name-drops tools it
never actually executes.

## Architecture

```
generate_data → ingest (producer/consumer + schema check)
             → quality_gate (completeness/validity/uniqueness/accuracy)
             → lakehouse (bronze → silver → gold, MERGE upsert)
             → rag_demo (chunk → embed → hybrid search → grounded answer)
```

This is a real DAG (Directed Acyclic Graph) — `orchestrator.py` sorts and
runs these five tasks in dependency order, exactly like Airflow's
scheduler does. `airflow_dag.py` is the same DAG written as an actual
Airflow `DAG` object, for production reference.

| Layer | What it does | Production tool it represents | What's actually running here |
|---|---|---|---|
| Ingestion | Producer publishes ticket events, consumer reads them, schema is checked at the door | Apache Kafka | Python `queue.Queue` + a background thread |
| Quality Gate | Checks Completeness, Validity, Uniqueness, Accuracy; quarantines bad rows | Great Expectations | Hand-written check functions (same 4 dimensions) |
| Lineage | START/COMPLETE/FAIL events logged per run | OpenLineage → Marquez | Same JSON event shape, appended to `logs/lineage_events.jsonl` |
| Lakehouse | bronze (raw) → silver (validated) → gold (aggregated), MERGE upsert, schema enforcement | Delta Lake on S3 | JSON files in `data/bronze` `/silver` `/gold` + a Python `merge_upsert()` function |
| RAG | Chunk KB articles → vectorize → hybrid search → rerank → grounded, cited answer | Sentence-Transformers + Qdrant/Chroma + an LLM | scikit-learn `TfidfVectorizer` + cosine similarity + a template generator |
| Orchestration | Task dependency graph, topological execution order | Apache Airflow | `orchestrator.py`'s `SimpleDAG` class (same topological-sort algorithm Airflow uses) + `airflow_dag.py` (the real Airflow DAG file) |

## How to run it

```bash
pip install -r requirements.txt
python orchestrator.py
```

That single command runs the entire pipeline end-to-end: generates 200
synthetic support tickets (with some intentionally broken ones mixed in
so the quality gate has real problems to catch), ingests them, quality
checks them, builds the gold aggregate table, and answers 3 sample support
questions using RAG. Everything is deterministic (seeded), so you'll get
the same numbers every run.

Each module can also be run on its own to inspect one stage at a time:

```bash
python data_generator.py   # writes data/raw_tickets.json, data/kb_articles.json
python ingestion.py        # reads raw tickets, writes data/bronze/
python quality_gate.py     # reads bronze, writes data/silver/, data/quarantine/
python lakehouse.py        # reads silver, writes data/gold/
python rag_pipeline.py     # reads kb_articles + gold, writes logs/rag_demo_output.json
```

## Output (actual run, captured in `logs/full_run_output.txt`)

```
[orchestrator] DAG 'capstone_support_platform' execution order: ['generate_data', 'ingest', 'quality_gate', 'lakehouse', 'rag_demo']

[orchestrator] ---- running task 'generate_data' (depends_on=[]) ----
[orchestrator] task 'generate_data' SUCCESS (0.004s)

[orchestrator] ---- running task 'ingest' (depends_on=['generate_data']) ----
[ingestion] producer publishing 200 events to topic 'support-tickets'
[ingestion] consumer accepted 200 events, rejected 0 at schema validation
[orchestrator] task 'ingest' SUCCESS (0.504s)

[orchestrator] ---- running task 'quality_gate' (depends_on=['ingest']) ----
[quality_gate] PASS_WITH_WARNINGS — 180/200 rows passed (90.0%). Failures by dimension: {'completeness': 8, 'validity': 8, 'accuracy': 0, 'uniqueness': 7}
[orchestrator] task 'quality_gate' SUCCESS (0.003s)

[orchestrator] ---- running task 'lakehouse' (depends_on=['quality_gate']) ----
[lakehouse] gold table written: 6 product rows (6 upserted this run) -> data/gold/product_stats_gold.json
[orchestrator] task 'lakehouse' SUCCESS (0.000s)

[orchestrator] ---- running task 'rag_demo' (depends_on=['generate_data', 'lakehouse']) ----
[rag] indexed 6 chunks from 6 KB articles
[rag] query='My Laptop Pro 14 shipment is late, what should I do?' -> top match KB-002 (score=0.2112)
[rag] query='Customer wants a refund for a defective product' -> top match KB-003 (score=0.5182)
[rag] query='I can't login to my account, how do I fix it?' -> top match KB-004 (score=0.1959)
[orchestrator] task 'rag_demo' SUCCESS (0.008s)
```

Sample generated (grounded, cited) RAG answer, from `logs/rag_demo_output.json`:

> **Q:** Customer wants a refund for a defective product
> **A:** Based on Handling Defective Products (KB-003): For defective
> product reports (dead pixels, broken buttons, DOA units), agents should
> first offer a free replacement before a refund. Escalate to the
> hardware team if the same product model has 3+ defect reports in a
> week.
> **Citations:** KB-003 (Handling Defective Products), KB-002 (Refund
> Eligibility), KB-001 (Shipping Delay Policy)

Full quality report, from `logs/quality_report.json`:

```json
{
  "total_rows": 200,
  "passed": 180,
  "quarantined": 20,
  "pass_rate_pct": 90.0,
  "dimension_failure_counts": {
    "completeness": 8,
    "validity": 8,
    "accuracy": 0,
    "uniqueness": 7
  },
  "status": "PASS_WITH_WARNINGS"
}
```

Every quarantined row is kept in `data/quarantine/quality_rejects.json`
with the exact reason it failed (e.g.
`"validity: priority 'URGENT!!' not in ['high', 'low', 'medium']"`), so
you can trace precisely why the gate rejected it — that traceability is
the entire point of a quality gate.

## Repo layout

```
data_generator.py     synthetic ticket + knowledge-base data (with intentional bad rows)
ingestion.py           Deliverable 1: producer/consumer + schema validation
quality_gate.py        Deliverable 5: quality checks + OpenLineage-style events
lakehouse.py            Deliverable 2: bronze/silver/gold + MERGE + schema enforcement
rag_pipeline.py         Deliverable 3: chunking, embedding, hybrid search, grounded answers
orchestrator.py          Deliverable 4: DAG engine that runs everything end-to-end
airflow_dag.py            the same DAG as a real Apache Airflow DAG (reference only)
data/                       bronze / silver / gold / quarantine outputs land here
logs/                        quality report, lineage events, RAG output, full run log
requirements.txt
```

## Mapping to the grading rubric

- **Deliverable 1 — Ingestion (20 pts):** `ingestion.py` — producer/consumer
  pattern + schema validation at the boundary.
- **Deliverable 2 — Lakehouse (25 pts):** `lakehouse.py` — bronze/silver/gold
  zones, `merge_upsert()` implementing MERGE semantics, `enforce_schema()`
  rejecting malformed gold rows.
- **Deliverable 3 — RAG pipeline (25 pts):** `rag_pipeline.py` — chunking,
  vectorization, hybrid (vector + lexical) search, reranking, grounded/cited
  generation.
- **Deliverable 4 — Orchestration (15 pts):** `orchestrator.py` (executable
  DAG engine) + `airflow_dag.py` (real Airflow DAG definition).
- **Deliverable 5 — Quality gate (15 pts):** `quality_gate.py` — four DAMA
  quality dimensions + OpenLineage-style START/COMPLETE/FAIL events in
  `logs/lineage_events.jsonl`.

## What I'd change for a real production deployment

- Swap `queue.Queue` in `ingestion.py` for `kafka-python`'s
  `KafkaProducer`/`KafkaConsumer` — the `produce()`/`consume_all()` method
  shapes are already written to match.
- Swap the JSON files in `lakehouse.py` for real Delta Lake tables via the
  `deltalake` package (`write_deltalake(path, df, mode="merge", ...)`).
- Swap `TfidfVectorizer` in `rag_pipeline.py` for a real neural embedding
  model (`sentence-transformers`) and store vectors in Qdrant/ChromaDB
  instead of an in-memory list.
- Swap `generate_answer()`'s template for an actual LLM API call (pass it
  the retrieved chunks as context and the user's question).
- Point `emit_lineage_event()` at a running Marquez server instead of a
  local file.
- Deploy `airflow_dag.py` to a real Airflow instance for scheduling,
  retries, and monitoring instead of running `orchestrator.py` by hand.
