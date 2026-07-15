"""
ingestion.py
------------
DELIVERABLE 1: Ingestion layer (producer + consumer + schema validation)

Real Apache Kafka, via kafka-python. A KafkaProducer publishes ticket
events onto the 'support-tickets' topic; a KafkaConsumer reads them back.
Schema validation happens BEFORE anything is sent to Kafka (a Pydantic
model plays the role a Kafka Schema Registry / Avro schema would play in
production) — bad events never touch the topic, they go straight to the
rejected/quarantine list.

Requires a running broker. Easiest local option (matches the version the
instructor suggested):

    docker run -d --name kafka -p 9092:9092 apache/kafka:3.7.0

Override the broker address with the KAFKA_BOOTSTRAP_SERVERS env var if
you're not running it on localhost:9092.
"""

import json
import os
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable
from pydantic import BaseModel, EmailStr, ValidationError, field_validator

TOPIC_NAME = "support-tickets"
BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


class TicketEvent(BaseModel):
    """Structural schema for a ticket event — the Pydantic-model
    upgrade of the old hand-written TICKET_SCHEMA dict. This is the same
    job a Kafka Schema Registry / Avro schema does in production: reject
    malformed events at the boundary, before they're allowed onto the
    topic. Business-rule checks (is this email *actually* valid for our
    CRM, is the priority one we care about, etc.) still happen downstream
    in the Quality Gate (deliverable 5) — that separation mirrors real
    systems, where schema validation and data quality are two different
    layers.
    """

    ticket_id: str
    customer_email: str  # kept as str (not EmailStr) — malformed emails
    # are intentionally allowed through here so the Quality Gate's
    # validity check has real problems to catch; EmailStr would reject
    # them at this earlier layer instead.
    product: str
    topic: str
    message: str
    created_at: str
    priority: str

    @field_validator("ticket_id", "product", "topic", "created_at", "priority")
    @classmethod
    def _not_empty_type_check(cls, v):
        # Pydantic already enforces "is this a str"; this just documents
        # that structural (not business-rule) emptiness is still a
        # schema-shape concern, not a quality-gate concern.
        return v


def validate_schema(event: dict) -> tuple[bool, list[str]]:
    """Kept as the public rejection gate ingestion.py has always exposed,
    now backed by the Pydantic model above instead of hand-rolled
    isinstance() checks."""
    try:
        TicketEvent(**event)
        return True, []
    except ValidationError as e:
        errors = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
        return False, errors


def _make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
    )


def _make_consumer(group_id: str) -> KafkaConsumer:
    return KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=8000,  # stop iterating after 8s of silence —
        # this is a batch demo run, not a long-lived service; a real
        # consumer would just run forever with .poll()/for-loop.
    )


def run_ingestion(raw_events: list[dict], log_fn=print):
    """Producer publishes every raw ticket that passes schema validation
    onto the Kafka topic; a consumer (fresh, unique group each run so we
    only see this run's messages and don't replay history) reads them
    back off the topic and hands them to the Bronze layer."""
    rejected = []
    valid_events = []
    for event in raw_events:
        ok, errors = validate_schema(event)
        if ok:
            valid_events.append(event)
        else:
            rejected.append({"event": event, "errors": errors})

    try:
        producer = _make_producer()
    except NoBrokersAvailable as e:
        raise RuntimeError(
            f"Could not reach a Kafka broker at {BOOTSTRAP_SERVERS}. "
            f"Start one first, e.g.:\n"
            f"  docker run -d --name kafka -p 9092:9092 apache/kafka:3.7.0\n"
            f"or set KAFKA_BOOTSTRAP_SERVERS to point at yours."
        ) from e

    # Fresh consumer group per run + auto_offset_reset='latest' means this
    # consumer only picks up messages produced *after* it subscribes, so
    # it only sees this run's batch even though the topic persists across
    # runs. Subscribe and force a partition assignment before producing.
    group_id = f"support-tickets-consumer-{uuid.uuid4().hex[:8]}"
    consumer = _make_consumer(group_id)
    consumer.poll(timeout_ms=0)  # triggers group join / partition assignment

    log_fn(f"[ingestion] producer publishing {len(valid_events)} events to topic '{TOPIC_NAME}'")
    for event in valid_events:
        producer.send(TOPIC_NAME, value=event)
    producer.flush()
    producer.close()

    accepted = [msg.value for msg in consumer]
    consumer.close()

    log_fn(f"[ingestion] consumer accepted {len(accepted)} events, "
           f"rejected {len(rejected)} at schema validation")

    return {
        "accepted": accepted,
        "rejected": rejected,
        "topic_name": TOPIC_NAME,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
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
