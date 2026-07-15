"""
orchestrator.py
----------------
DELIVERABLE 4: Orchestration (Airflow DAG connecting all modules end-to-end)

A real deployment would run this as an Apache Airflow DAG (see
`airflow_dag.py` in this repo for the actual Airflow DAG definition — task
dependencies, retries, schedule_interval, etc.). Airflow needs its own
scheduler/webserver/metadata database running as a service, which isn't
available in this sandbox, so THIS file is a minimal, dependency-free DAG
engine: it defines tasks as nodes, dependencies as edges, sorts them into a
valid execution order (topological sort — the exact algorithm Airflow uses
internally), and runs each task only after its dependencies succeed.

Both files describe the SAME DAG:
    generate_data -> ingest -> quality_gate -> lakehouse -> rag_demo

Run this file directly (`python orchestrator.py`) to execute the full
pipeline end-to-end and see every stage's output. Run `airflow_dag.py`'s
definitions inside a real Airflow install to schedule it in production.
"""

import json
import time
from datetime import datetime

import data_generator
import ingestion
import quality_gate
import lakehouse
import rag_pipeline


class Task:
    def __init__(self, task_id, fn, depends_on=None):
        self.task_id = task_id
        self.fn = fn
        self.depends_on = depends_on or []


class SimpleDAG:
    """Topological-sort task runner — same core algorithm Airflow's
    scheduler uses to decide execution order from a dependency graph."""

    def __init__(self, dag_id):
        self.dag_id = dag_id
        self.tasks = {}

    def add_task(self, task: Task):
        self.tasks[task.task_id] = task

    def _topological_order(self):
        visited, order = set(), []

        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            for dep in self.tasks[task_id].depends_on:
                visit(dep)
            order.append(task_id)

        for tid in self.tasks:
            visit(tid)
        return order

    def run(self, log_fn=print):
        order = self._topological_order()
        log_fn(f"[orchestrator] DAG '{self.dag_id}' execution order: {order}")
        context = {}
        for task_id in order:
            task = self.tasks[task_id]
            log_fn(f"\n[orchestrator] ---- running task '{task_id}' "
                   f"(depends_on={task.depends_on}) ----")
            t0 = time.time()
            try:
                context[task_id] = task.fn(context)
                log_fn(f"[orchestrator] task '{task_id}' SUCCESS "
                       f"({time.time() - t0:.3f}s)")
            except Exception as e:
                log_fn(f"[orchestrator] task '{task_id}' FAILED: {e}")
                raise
        return context


# ---- task functions: each one wraps a deliverable module ----

def task_generate_data(ctx):
    tickets = data_generator.generate_tickets()
    kb = data_generator.generate_kb_articles()
    with open("data/raw_tickets.json", "w") as f:
        json.dump(tickets, f, indent=2)
    with open("data/kb_articles.json", "w") as f:
        json.dump(kb, f, indent=2)
    return {"tickets": tickets, "kb": kb}


def task_ingest(ctx):
    tickets = ctx["generate_data"]["tickets"]
    result = ingestion.run_ingestion(tickets)
    with open("data/bronze/tickets_bronze.json", "w") as f:
        json.dump(result["accepted"], f, indent=2)
    with open("data/quarantine/schema_rejects.json", "w") as f:
        json.dump(result["rejected"], f, indent=2)
    return result


def task_quality_gate(ctx):
    bronze_rows = ctx["ingest"]["accepted"]
    run_id = f"orchestrated-{int(datetime.now().timestamp())}"
    clean, quarantined, report = quality_gate.run_quality_gate(bronze_rows, run_id)
    with open("data/silver/tickets_silver.json", "w") as f:
        json.dump(clean, f, indent=2)
    with open("data/quarantine/quality_rejects.json", "w") as f:
        json.dump(quarantined, f, indent=2)
    with open("logs/quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    if report["status"] == "FAIL":
        raise RuntimeError("Quality gate failed — halting pipeline before lakehouse write")
    return {"clean": clean, "quarantined": quarantined, "report": report}


def task_lakehouse(ctx):
    clean_rows = ctx["quality_gate"]["clean"]
    gold = lakehouse.run_lakehouse(clean_rows)
    return {"gold": gold}


def task_rag_demo(ctx):
    kb = ctx["generate_data"]["kb"]
    gold = ctx["lakehouse"]["gold"]
    demo_queries = [
        "My Laptop Pro 14 shipment is late, what should I do?",
        "Customer wants a refund for a defective product",
        "I can't login to my account, how do I fix it?",
    ]
    results = rag_pipeline.run_rag_demo(kb, gold, demo_queries)
    with open("logs/rag_demo_output.json", "w") as f:
        json.dump(results, f, indent=2)
    return {"results": results}


def build_dag():
    dag = SimpleDAG("capstone_support_platform")
    dag.add_task(Task("generate_data", task_generate_data))
    dag.add_task(Task("ingest", task_ingest, depends_on=["generate_data"]))
    dag.add_task(Task("quality_gate", task_quality_gate, depends_on=["ingest"]))
    dag.add_task(Task("lakehouse", task_lakehouse, depends_on=["quality_gate"]))
    dag.add_task(Task("rag_demo", task_rag_demo, depends_on=["generate_data", "lakehouse"]))
    return dag


if __name__ == "__main__":
    dag = build_dag()
    dag.run()
