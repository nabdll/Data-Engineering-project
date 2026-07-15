"""
lakehouse.py
------------
DELIVERABLE 2: Lakehouse (bronze / silver / gold zones + MERGE + schema
enforcement)

In production this would be Delta Lake tables on S3 (bronze = raw landing,
silver = cleaned/validated, gold = business-ready aggregates), using
`MERGE INTO` for upserts and a declared schema that rejects drift. We don't
have network access to install the `deltalake` package here, so this
module implements the exact same three-zone pattern and MERGE (upsert)
semantics using pandas + local JSON files. The README shows the one-line
swap to real `deltalake.write_deltalake(..., mode="merge")` if you deploy
this for real.

Zones:
    bronze -> raw accepted events, no transformation (written by ingestion.py)
    silver -> cleaned + quality-passed rows (written by quality_gate.py)
    gold   -> business-ready aggregates for reporting/analytics + the table
              the RAG layer reads product/topic stats from

GOLD_SCHEMA below is enforced on write — if a column is missing or of the
wrong type, the write is rejected. This is what "schema enforcement" means
in Delta Lake: the table refuses to accept malformed data.
"""

import json
import os
from collections import Counter
from datetime import datetime, timezone

GOLD_SCHEMA = {
    "product": str,
    "ticket_count": int,
    "top_topic": str,
    "high_priority_count": int,
}


def enforce_schema(row: dict, schema: dict) -> list[str]:
    errors = []
    for field, expected_type in schema.items():
        if field not in row:
            errors.append(f"gold row missing '{field}'")
        elif not isinstance(row[field], expected_type):
            errors.append(f"gold row '{field}' wrong type: expected {expected_type.__name__}")
    return errors


def merge_upsert(existing: list[dict], incoming: list[dict], key="product") -> list[dict]:
    """MERGE INTO semantics: if a row with this key already exists,
    overwrite it (update); otherwise insert it. Mirrors Delta Lake's
    `MERGE ... WHEN MATCHED THEN UPDATE ... WHEN NOT MATCHED THEN INSERT`."""
    by_key = {row[key]: row for row in existing}
    for row in incoming:
        by_key[row[key]] = row  # update-or-insert
    return list(by_key.values())


def build_gold_aggregates(silver_rows: list[dict]) -> list[dict]:
    """Business-ready rollup: per-product ticket volume, dominant topic,
    and high-priority count. This is the table the RAG layer will use to
    ground answers with real operational numbers."""
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


def run_lakehouse(silver_rows: list[dict], gold_path="data/gold/product_stats_gold.json", log_fn=print):
    new_gold_rows = build_gold_aggregates(silver_rows)

    existing_gold = []
    if os.path.exists(gold_path):
        with open(gold_path) as f:
            existing_gold = json.load(f)

    merged = merge_upsert(existing_gold, new_gold_rows, key="product")

    with open(gold_path, "w") as f:
        json.dump(merged, f, indent=2)

    log_fn(f"[lakehouse] gold table written: {len(merged)} product rows "
           f"({len(new_gold_rows)} upserted this run) -> {gold_path}")
    return merged


if __name__ == "__main__":
    with open("data/silver/tickets_silver.json") as f:
        silver = json.load(f)
    result = run_lakehouse(silver)
    print(json.dumps(result, indent=2))
