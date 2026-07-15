"""
quality_gate.py
----------------
DELIVERABLE 5: Quality gate (real Great Expectations 1.x) + real
OpenLineage lineage events.

Part A — Quality Gate
The four DAMA dimensions (Completeness, Validity, Uniqueness, Accuracy)
are still checked by hand, row by row, because that's what gives us
per-row quarantine reasons like "validity: priority 'URGENT!!' not in
[...]" — that granular traceability is the whole point of the gate and
Great Expectations' Checkpoint result doesn't hand you that per-row
string for free.

Alongside those hand-rolled checks, we now ALSO run a real GX 1.x
ExpectationSuite + Checkpoint against the same batch, as an independent,
industry-standard validation pass. Its aggregate pass/fail result is
folded into the quality report. If your installed great_expectations
version has drifted from the fluent 1.x API used below (GX's API moved a
few times across 1.x minor releases), the checkpoint step logs a warning
and the hand-rolled checks still run the show — see
https://docs.greatexpectations.io for the exact API of your version.

Part B — Lineage events (real OpenLineage)
Real openlineage-python OpenLineageClient, emitting START/COMPLETE/FAIL
RunEvents through a local FileTransport (logs/lineage_events.jsonl) —
this is the exact client you'd point at a running Marquez server instead,
by swapping FileTransport for HttpTransport.
"""

import json
import re
import uuid
from datetime import datetime, timezone

import pandas as pd

from openlineage.client import OpenLineageClient
from openlineage.client.transport.file import FileConfig, FileTransport
from openlineage.client.run import Job, Run, RunEvent, RunState

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ALLOWED_PRIORITIES = {"low", "medium", "high"}
REQUIRED_FIELDS = ["ticket_id", "customer_email", "product", "topic", "message", "created_at", "priority"]

LINEAGE_LOG_PATH = "logs/lineage_events.jsonl"
JOB_NAMESPACE = "capstone.support_platform"


def _lineage_client() -> OpenLineageClient:
    transport = FileTransport(FileConfig(log_file_path=LINEAGE_LOG_PATH))
    return OpenLineageClient(transport=transport)


def emit_lineage_event(event_type: str, job_name: str, run_id: str, extra: dict | None = None):
    """Emit one real OpenLineage RunEvent. event_type is 'START',
    'COMPLETE', or 'FAIL' (matches openlineage.client.run.RunState)."""
    client = _lineage_client()
    event = RunEvent(
        eventType=getattr(RunState, event_type),
        eventTime=datetime.now(timezone.utc).isoformat(),
        run=Run(runId=run_id, facets=extra or {}),
        job=Job(namespace=JOB_NAMESPACE, name=job_name),
        producer="capstone-quality-gate",
        inputs=[],
        outputs=[],
    )
    client.emit(event)
    return event


# ---------------------------------------------------------------------
# Part A: hand-rolled per-row checks (kept — this is where the
# quarantine-with-reasons traceability comes from)
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Part B: real Great Expectations 1.x ExpectationSuite + Checkpoint
# ---------------------------------------------------------------------

def run_gx_checkpoint(rows: list[dict], run_id: str, log_fn=print) -> dict | None:
    """Runs a real GX 1.x Checkpoint over the batch as an independent
    validation pass. Returns the checkpoint's summary dict, or None if GX
    couldn't run (missing/incompatible install) — the hand-rolled checks
    above are the pipeline's actual gate either way, so this is additive
    evidence, not a hard dependency."""
    try:
        import great_expectations as gx
        from great_expectations.expectations import (
            ExpectColumnValuesToNotBeNull,
            ExpectColumnValuesToBeInSet,
            ExpectColumnValuesToMatchRegex,
            ExpectColumnValuesToBeUnique,
        )

        df = pd.DataFrame(rows)

        context = gx.get_context(mode="ephemeral")

        suite = context.suites.add(gx.ExpectationSuite(name=f"ticket_quality_suite_{run_id}"))
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column="ticket_id"))
        suite.add_expectation(ExpectColumnValuesToBeUnique(column="ticket_id"))
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column="customer_email"))
        suite.add_expectation(
            ExpectColumnValuesToMatchRegex(column="customer_email", regex=EMAIL_RE.pattern)
        )
        suite.add_expectation(
            ExpectColumnValuesToBeInSet(column="priority", value_set=sorted(ALLOWED_PRIORITIES))
        )
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column="message"))

        data_source = context.data_sources.add_pandas(f"pandas_{run_id}")
        data_asset = data_source.add_dataframe_asset(name="tickets")
        batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")

        validation_definition = context.validation_definitions.add(
            gx.ValidationDefinition(name=f"ticket_validation_{run_id}", data=batch_definition, suite=suite)
        )
        checkpoint = context.checkpoints.add(
            gx.Checkpoint(name=f"ticket_checkpoint_{run_id}", validation_definitions=[validation_definition])
        )
        result = checkpoint.run(batch_parameters={"dataframe": df})

        summary = {
            "success": bool(result.success),
            "statistics": getattr(result, "statistics", None),
        }
        log_fn(f"[quality_gate] GX checkpoint success={summary['success']}")
        return summary

    except Exception as e:  # GX 1.x API surface has moved between minor
        # releases — don't let a version mismatch take down the whole
        # pipeline; the hand-rolled checks are still the real gate.
        log_fn(f"[quality_gate] WARNING: GX checkpoint could not run ({e}); "
               f"continuing with hand-rolled checks only")
        return None


# ---------------------------------------------------------------------
# Orchestration of both
# ---------------------------------------------------------------------

def run_quality_gate(rows: list[dict], run_id: str, log_fn=print):
    emit_lineage_event("START", "quality_gate", run_id)

    gx_summary = run_gx_checkpoint(rows, run_id, log_fn=log_fn)

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
        "great_expectations_checkpoint": gx_summary,
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

    run_id = f"qg-{uuid.uuid4().hex[:8]}"
    clean, quarantined, report = run_quality_gate(rows, run_id)

    with open("data/silver/tickets_silver.json", "w") as f:
        json.dump(clean, f, indent=2)
    with open("data/quarantine/quality_rejects.json", "w") as f:
        json.dump(quarantined, f, indent=2)
    with open("logs/quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
