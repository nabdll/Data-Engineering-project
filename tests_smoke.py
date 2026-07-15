"""
tests_smoke.py
---------------
Quick sanity checks you can run after `python orchestrator.py` to prove
the pipeline actually produced correct, non-empty output — this is what
"the code has to work and the output has to be shown" means in practice:
don't just trust that it ran, check what it produced.

Run with:  python tests_smoke.py
"""

import json
import os


def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    return condition


def main():
    all_ok = True

    all_ok &= check(os.path.exists("data/bronze/tickets_bronze.json"), "bronze layer exists")
    all_ok &= check(os.path.exists("data/silver/tickets_silver.json"), "silver layer exists")
    all_ok &= check(os.path.exists("data/gold/product_stats_gold.json"), "gold layer exists")
    all_ok &= check(os.path.exists("logs/quality_report.json"), "quality report exists")
    all_ok &= check(os.path.exists("logs/lineage_events.jsonl"), "lineage events exist")
    all_ok &= check(os.path.exists("logs/rag_demo_output.json"), "RAG demo output exists")

    with open("logs/quality_report.json") as f:
        report = json.load(f)
    all_ok &= check(report["total_rows"] > 0, "quality report checked > 0 rows")
    all_ok &= check(report["quarantined"] > 0, "quality gate actually caught bad rows "
                     "(proves the checks work, not just pass everything)")
    all_ok &= check(report["status"] in ("PASS", "PASS_WITH_WARNINGS", "FAIL"),
                     "quality report has a valid status")

    with open("data/gold/product_stats_gold.json") as f:
        gold = json.load(f)
    all_ok &= check(len(gold) > 0, "gold table has rows")
    all_ok &= check(all("ticket_count" in row for row in gold), "gold rows have ticket_count")

    with open("logs/rag_demo_output.json") as f:
        rag_results = json.load(f)
    all_ok &= check(len(rag_results) > 0, "RAG produced results for demo queries")
    all_ok &= check(all(r["generated"]["citations"] for r in rag_results),
                     "every RAG answer has at least one citation (grounded, not hallucinated)")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
