"""
airflow_dag.py
--------------
The REAL Apache Airflow DAG definition for this pipeline — the
production version of what `orchestrator.py` runs locally.

This file's logic is unchanged (the instructor's note was that it was
"correctly written but never actually run") — the fix here isn't code,
it's actually executing it once and keeping proof of the run.

How to actually run this for real and capture proof:

    pip install -r requirements-airflow-optional.txt   # apache-airflow
    export AIRFLOW_HOME=~/airflow
    airflow db migrate                # first time only (Airflow 2.7+)
    cp airflow_dag.py $AIRFLOW_HOME/dags/
    airflow standalone
    # open the printed URL, trigger "capstone_support_platform", let it
    # finish, then screenshot the successful DAG run in the UI.

Or, faster (no webserver, no UI, just proves the DAG object is valid and
runs task-by-task) using Airflow's dag.test() helper:

    python -c "
    from airflow_dag import dag
    dag.test()
    "

Either way, keep the terminal output / screenshot as the deliverable
proof this was actually executed, not just written.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

import data_generator
import ingestion
import quality_gate
import lakehouse
import rag_pipeline

default_args = {
    "owner": "capstone-student",
    "retries": 2,
    "retry_delay": 300,  # seconds
}

with DAG(
    dag_id="capstone_support_platform",
    description="Real-Time Customer Support Intelligence Platform",
    schedule_interval="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["capstone", "data-quality", "rag"],
) as dag:

    def _generate_data(**context):
        tickets = data_generator.generate_tickets()
        kb = data_generator.generate_kb_articles()
        context["ti"].xcom_push(key="ticket_count", value=len(tickets))
        return {"tickets": tickets, "kb": kb}

    def _ingest(**context):
        upstream = context["ti"].xcom_pull(task_ids="generate_data")
        result = ingestion.run_ingestion(upstream["tickets"])
        return result

    def _quality_gate(**context):
        upstream = context["ti"].xcom_pull(task_ids="ingest")
        run_id = context["run_id"]
        clean, quarantined, report = quality_gate.run_quality_gate(upstream["accepted"], run_id)
        if report["status"] == "FAIL":
            raise RuntimeError("Quality gate failed — halting DAG, batch quarantined")
        return {"clean": clean, "report": report}

    def _lakehouse(**context):
        upstream = context["ti"].xcom_pull(task_ids="quality_gate")
        gold = lakehouse.run_lakehouse(upstream["clean"])
        return {"gold": gold}

    def _rag_demo(**context):
        gen = context["ti"].xcom_pull(task_ids="generate_data")
        lake = context["ti"].xcom_pull(task_ids="lakehouse")
        demo_queries = [
            "My Laptop Pro 14 shipment is late, what should I do?",
            "Customer wants a refund for a defective product",
        ]
        return rag_pipeline.run_rag_demo(gen["kb"], lake["gold"], demo_queries)

    generate_data = PythonOperator(task_id="generate_data", python_callable=_generate_data)
    ingest = PythonOperator(task_id="ingest", python_callable=_ingest)
    quality_gate_task = PythonOperator(task_id="quality_gate", python_callable=_quality_gate)
    lakehouse_task = PythonOperator(task_id="lakehouse", python_callable=_lakehouse)
    rag_demo = PythonOperator(task_id="rag_demo", python_callable=_rag_demo)

    generate_data >> ingest >> quality_gate_task >> lakehouse_task >> rag_demo
