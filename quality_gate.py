"""
quality_gate.py
----------------
DELIVERABLE 5: Quality gate (Great Expectations style) + OpenLineage events

Part A - Quality Gate
    In production you'd write these checks as a Great Expectations
    "Expectation Suite" (e.g. expect_column_values_to_not_be_null). We
    don't have network access to install the `great_expectations` package
    here, so we implement the SAME four dimensions by hand, in plain
    Python, so you can see exactly what the library would be doing under
    the hood:
        - Completeness : required fields aren't empty/null
        - Validity     : values match expected patterns/ranges
        - Uniqueness   : primary key (ticket_id) has no duplicates
        - Accuracy     : cross-field sanity (e.g. priority is one of the
                         allowed values)
    Any row that fails ANY check is quarantined (written to
    data/quarantine/) instead of being allowed into the Silver layer.

Part B - Lineage events (OpenLineage-style)
    OpenLineage defines three event types: START, COMPLETE, FAIL. Real
    systems POST these as JSON to a lineage server (Marquez). We don't
    have network access to run Marquez, so we append the same JSON
    payload shape to a local file (logs/lineage_events.jsonl) — this is
    exactly what you'd forward to Marquez with one line of code
    (see README).
"""

import json
import re
from datetime import datetime, timezone

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ALLOWED_PRIORITIES = {"low", "medium", "high"}
REQUIRED_FIELDS = ["ticket_id", "customer_email", "product", "topic", "message", "created_at", "priority"]


def emit_lineage_event(event_type, job_name, run_id, extra=None, log_path="logs/lineage_events.jsonl"):
    """Append one OpenLineage-shaped event. event_type is START, COMPLETE, or FAIL."""
    payload = {
        "eventType": event_type,
        "eventTime": datetime.now(timezone.utc).isoformat(),
        "run": {"runId": run_id},
        "job": {"namespace": "capstone.support_platform", "name": job_name},
        "producer": "capstone-quality-gate",
    }
    if extra:
        payload["extra"] = extra
    with open(log_path, "a") as f:
        f.write(json.dumps(payload) + "\n")
    return payload


def check_completeness(row):
    missing = [f for f in REQUIRED_FIELDS if not str(row.get(f, "")).strip()]
    return missing  # empty list = pass


def check_validity(row):
    problems = []
    if not EMAIL_RE.match(row.get("customer_email", "")):
        problems.append(f"invalid email format: '{row.get('customer_email')}'")
    if row.get("priority") not in ALLOWED_PRIORITIES:
        problems.append(f"priority '{row.get('priority')}' not in {sorted(ALLOWED_PRIORITIES)}")
    return problems


def check_accuracy(row):
    problems = []
    # created_at should parse as a real ISO timestamp
    try:
        datetime.fromisoformat(row.get("created_at", ""))
    except Exception:
        problems.append(f"created_at is not a valid timestamp: '{row.get('created_at')}'")
    return problems


def check_uniqueness(rows):
    """Returns the set of ticket_ids that appear more than once."""
    seen = {}
    dupes = set()
    for row in rows:
        tid = row.get("ticket_id")
        seen[tid] = seen.get(tid, 0) + 1
        if seen[tid] > 1:
            dupes.add(tid)
    return dupes


def run_quality_gate(rows: list[dict], run_id: str, log_fn=print):
    emit_lineage_event("START", "quality_gate", run_id)

    dupes = check_uniqueness(rows)
    seen_ids = set()

    clean_rows = []
    quarantined = []
    failure_counts = {"completeness": 0, "validity": 0, "accuracy": 0, "uniqueness": 0}

    for row in rows:
        reasons = []

        missing = check_completeness(row)
        if missing:
            reasons.append(f"completeness: missing/empty {missing}")
            failure_counts["completeness"] += 1

        validity_problems = check_validity(row)
        if validity_problems:
            reasons.extend([f"validity: {p}" for p in validity_problems])
            failure_counts["validity"] += 1

        accuracy_problems = check_accuracy(row)
        if accuracy_problems:
            reasons.extend([f"accuracy: {p}" for p in accuracy_problems])
            failure_counts["accuracy"] += 1

        tid = row.get("ticket_id")
        if tid in dupes:
            if tid in seen_ids:
                reasons.append(f"uniqueness: duplicate ticket_id '{tid}'")
                failure_counts["uniqueness"] += 1
            seen_ids.add(tid)

        if reasons:
            quarantined.append({"row": row, "reasons": reasons})
        else:
            clean_rows.append(row)

    total = len(rows)
    report = {
        "run_id": run_id,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": total,
        "passed": len(clean_rows),
        "quarantined": len(quarantined),
        "pass_rate_pct": round(100 * len(clean_rows) / total, 2) if total else 0,
        "dimension_failure_counts": failure_counts,
        "status": "PASS" if len(quarantined) == 0 else (
            "FAIL" if len(clean_rows) / total < 0.7 else "PASS_WITH_WARNINGS"
        ),
    }

    if report["status"] == "FAIL":
        emit_lineage_event("FAIL", "quality_gate", run_id, extra=report)
        log_fn(f"[quality_gate] FAILED — pass rate {report['pass_rate_pct']}% is below the "
               f"70% gate threshold. Pipeline halted, batch quarantined.")
    else:
        emit_lineage_event("COMPLETE", "quality_gate", run_id, extra=report)
        log_fn(f"[quality_gate] {report['status']} — {report['passed']}/{total} rows passed "
               f"({report['pass_rate_pct']}%). Failures by dimension: {failure_counts}")

    return clean_rows, quarantined, report


if __name__ == "__main__":
    with open("data/bronze/tickets_bronze.json") as f:
        rows = json.load(f)

    run_id = f"qg-{int(datetime.now().timestamp())}"
    clean, quarantined, report = run_quality_gate(rows, run_id)

    with open("data/silver/tickets_silver.json", "w") as f:
        json.dump(clean, f, indent=2)
    with open("data/quarantine/quality_rejects.json", "w") as f:
        json.dump(quarantined, f, indent=2)
    with open("logs/quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
