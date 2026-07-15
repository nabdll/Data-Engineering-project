"""
lakehouse.py
------------
DELIVERABLE 2: Lakehouse (bronze / silver / gold zones + MERGE + schema
enforcement) — real Delta Lake, via the `deltalake` package.

Zones:
  bronze -> raw accepted events, no transformation (written by ingestion.py)
  silver -> cleaned + quality-passed rows (written by quality_gate.py)
  gold   -> business-ready aggregates, stored as a real Delta table at
            data/gold/product_stats_gold (a directory: Delta stores data
            as Parquet files + a _delta_log/ transaction log, not a
            single JSON file)

Schema enforcement is Delta's native behavior: write_deltalake() with
mode="merge" (via DeltaTable.merge()) validates the incoming Arrow/pandas
schema against the table's declared schema and raises on mismatch — we no
longer need a hand-written isinstance() check, we let the table itself
refuse malformed writes. We still run enforce_schema() first as a fast,
readable pre-check with clear error messages, but Delta's own schema
enforcement is the real gate.

Upsert semantics: DeltaTable.merge() with when_matched_update_all() /
when_not_matched_insert_all() is Delta's real `MERGE INTO ... WHEN MATCHED
THEN UPDATE ... WHEN NOT MATCHED THEN INSERT` — this is exactly what a
production Delta Lake deployment on S3/ADLS would run.
"""

import os
from collections import Counter

import pandas as pd
from deltalake import DeltaTable, write_deltalake

GOLD_SCHEMA = {
    "product": str,
    "ticket_count": int,
    "top_topic": str,
    "high_priority_count": int,
}

GOLD_TABLE_PATH = "data/gold/product_stats_gold"


def enforce_schema(row: dict, schema: dict) -> list[str]:
    """Fast pre-check with human-readable errors before we even build the
    DataFrame. Delta's own schema enforcement (triggered inside
    run_lakehouse below) is what actually protects the table."""
    errors = []
    for field, expected_type in schema.items():
        if field not in row:
            errors.append(f"gold row missing '{field}'")
        elif not isinstance(row[field], expected_type):
            errors.append(f"gold row '{field}' wrong type: expected {expected_type.__name__}")
    return errors


def build_gold_aggregates(silver_rows: list[dict]) -> list[dict]:
    """Business-ready rollup: per-product ticket volume, dominant topic,
    and high-priority count. This is the table the RAG layer reads
    operational numbers from."""
    by_product = {}
    for row in silver_rows:
        p = row["product"]
        by_product.setdefault(p, {"topics": Counter(), "count": 0, "high": 0})
        by_product[p]["topics"][row["topic"]] += 1
        by_product[p]["count"] += 1
        if row["priority"] == "high":
            by_product[p]["high"] += 1

    gold_rows = []
    for product, stats in by_product.items():
        top_topic = stats["topics"].most_common(1)[0][0] if stats["topics"] else "n/a"
        row = {
            "product": product,
            "ticket_count": stats["count"],
            "top_topic": top_topic,
            "high_priority_count": stats["high"],
        }
        errors = enforce_schema(row, GOLD_SCHEMA)
        if errors:
            raise ValueError(f"Schema enforcement failed for gold row {row}: {errors}")
        gold_rows.append(row)
    return gold_rows


def run_lakehouse(silver_rows: list[dict], gold_path=GOLD_TABLE_PATH, log_fn=print):
    new_gold_rows = build_gold_aggregates(silver_rows)
    new_df = pd.DataFrame(new_gold_rows)

    table_exists = os.path.exists(os.path.join(gold_path, "_delta_log"))

    if not table_exists:
        # First run: no table yet, just create it. Delta enforces the
        # schema of this initial write for every write that follows.
        write_deltalake(gold_path, new_df, mode="overwrite")
        merged_count = len(new_df)
    else:
        dt = DeltaTable(gold_path)
        (
            dt.merge(
                source=new_df,
                predicate="target.product = source.product",
                source_alias="source",
                target_alias="target",
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )
        merged_count = len(new_df)

    # Read the table back so callers (rag_pipeline.py, orchestrator.py)
    # keep getting the same list[dict] shape they always have.
    final_df = DeltaTable(gold_path).to_pandas()
    merged = final_df.to_dict(orient="records")

    log_fn(f"[lakehouse] gold Delta table written: {len(merged)} product rows "
           f"({merged_count} upserted this run) -> {gold_path}")
    return merged


if __name__ == "__main__":
    import json

    with open("data/silver/tickets_silver.json") as f:
        silver = json.load(f)
    result = run_lakehouse(silver)
    print(json.dumps(result, indent=2))
