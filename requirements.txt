"""
ingestion.py
------------
DELIVERABLE 1: Ingestion layer (producer + consumer + schema validation)

In a real enterprise system this would be an Apache Kafka topic: a
"producer" publishes events, and one or more "consumers" read them off the
topic. We don't have a Kafka cluster available in this environment, so we
build the same producer/consumer PATTERN using Python's built-in `queue`
module and a background thread — the architecture (decoupled
producer -> topic -> consumer, async processing, schema validation at the
boundary) is identical to Kafka; only the transport is swapped out.

The README explains exactly how to swap this for real Kafka
(kafka-python's KafkaProducer/KafkaConsumer have almost the same
send()/poll() shape).

Schema validation: every event is checked against a small hand-written
schema (types + required fields) before it's allowed onto the "topic".
This is the same idea as a Pydantic model or a Kafka Schema Registry
entry — reject bad shape at the door instead of downstream.
"""

import json
import queue
import threading
import time
from datetime import datetime

TICKET_SCHEMA = {
    "ticket_id": str,
    "customer_email": str,
    "product": str,
    "topic": str,
    "message": str,
    "created_at": str,
    "priority": str,
}


def validate_schema(event: dict) -> tuple[bool, list[str]]:
    """Structural check only: right fields, right types. Business-rule
    checks (like 'is this email actually valid') happen later in the
    Quality Gate (deliverable 5) — that separation mirrors real systems,
    where schema validation and data quality are two different layers."""
    errors = []
    for field, expected_type in TICKET_SCHEMA.items():
        if field not in event:
            errors.append(f"missing field '{field}'")
        elif not isinstance(event[field], expected_type):
            errors.append(f"field '{field}' should be {expected_type.__name__}")
    return (len(errors) == 0, errors)


class SupportTicketTopic:
    """Stands in for a Kafka topic. Thread-safe queue, producer publishes,
    consumer polls."""

    def __init__(self, name="support-tickets"):
        self.name = name
        self._queue = queue.Queue()
        self.rejected = []  # events that failed schema validation

    def produce(self, event: dict):
        ok, errors = validate_schema(event)
        if ok:
            self._queue.put(event)
        else:
            self.rejected.append({"event": event, "errors": errors})
        return ok, errors

    def consume_all(self, timeout=1.0):
        """Drain everything currently on the topic (a real consumer would
        run forever with .poll(); we drain once since this is a batch demo
        run, not a long-lived service)."""
        events = []
        try:
            while True:
                events.append(self._queue.get(timeout=timeout))
        except queue.Empty:
            pass
        return events


def run_ingestion(raw_events: list[dict], log_fn=print):
    """Producer publishes every raw ticket onto the topic; a consumer
    thread reads them and writes accepted ones to the Bronze layer
    (raw landing zone) in the lakehouse."""
    topic = SupportTicketTopic()

    accepted = []

    def consumer_worker():
        for event in topic.consume_all(timeout=0.5):
            accepted.append(event)

    # Producer publishes all events
    log_fn(f"[ingestion] producer publishing {len(raw_events)} events to topic '{topic.name}'")
    for event in raw_events:
        topic.produce(event)

    # Consumer drains the topic (in a background thread, like a real
    # long-running consumer service would)
    t = threading.Thread(target=consumer_worker)
    t.start()
    t.join()

    log_fn(f"[ingestion] consumer accepted {len(accepted)} events, "
           f"rejected {len(topic.rejected)} at schema validation")

    return {
        "accepted": accepted,
        "rejected": topic.rejected,
        "topic_name": topic.name,
        "ingested_at": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    with open("data/raw_tickets.json") as f:
        raw = json.load(f)
    result = run_ingestion(raw)
    with open("data/bronze/tickets_bronze.json", "w") as f:
        json.dump(result["accepted"], f, indent=2)
    with open("data/quarantine/schema_rejects.json", "w") as f:
        json.dump(result["rejected"], f, indent=2)
    print(f"Bronze layer written: {len(result['accepted'])} records")
    print(f"Schema-rejected (quarantined): {len(result['rejected'])} records")
