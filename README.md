"""
data_generator.py
------------------
Creates fake (synthetic) customer support data so the whole pipeline has
something to chew on, without needing a real company's data.

Two kinds of data come out of this:
1. "tickets"   -> raw customer support messages (like what would arrive on
                  a Kafka topic in real life)
2. "kb_articles" -> knowledge base articles support agents use to answer
                  tickets (this is what our RAG system will search over)

We deliberately inject some BAD rows (missing fields, bad emails, duplicate
ids, out-of-range values) so the Data Quality Gate later has real problems
to catch. That's the whole point of the assignment: prove the quality gate
actually works, not just that it runs.
"""

import json
import random
from datetime import datetime, timedelta

random.seed(42)  # makes the "random" data reproducible every run

PRODUCTS = ["Laptop Pro 14", "Wireless Mouse X1", "4K Monitor 27in",
            "Mechanical Keyboard K2", "USB-C Dock", "Noise Cancelling Headset"]

TOPICS = ["shipping delay", "refund request", "product defect",
          "login issue", "billing question", "warranty claim"]

GOOD_MESSAGE_TEMPLATES = [
    "My order for {product} is delayed, can you check the status?",
    "I want a refund for my {product}, it stopped working after 2 days.",
    "The {product} I received is defective, the screen has dead pixels.",
    "I can't log into my account to track my {product} order.",
    "I was charged twice for my {product}, please fix my billing.",
    "Is my {product} still under warranty? It broke after 3 months.",
]


def _random_timestamp(days_back=30):
    start = datetime(2026, 6, 1)
    offset = timedelta(days=random.randint(0, days_back),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59))
    return (start + offset).isoformat()


def generate_tickets(n_good=180, n_bad=20):
    """Generate a mix of clean and intentionally-broken support tickets."""
    tickets = []

    # ---- good, valid tickets ----
    for i in range(1, n_good + 1):
        product = random.choice(PRODUCTS)
        template = random.choice(GOOD_MESSAGE_TEMPLATES)
        tickets.append({
            "ticket_id": f"TCK-{i:05d}",
            "customer_email": f"customer{i}@example.com",
            "product": product,
            "topic": random.choice(TOPICS),
            "message": template.format(product=product),
            "created_at": _random_timestamp(),
            "priority": random.choice(["low", "medium", "high"]),
        })

    # ---- intentionally broken tickets (for the quality gate to catch) ----
    bad_makers = [
        # missing ticket_id
        lambda i: {"ticket_id": "", "customer_email": f"bad{i}@example.com",
                   "product": random.choice(PRODUCTS), "topic": "billing question",
                   "message": "Missing id test", "created_at": _random_timestamp(),
                   "priority": "low"},
        # malformed email
        lambda i: {"ticket_id": f"TCK-BAD{i:03d}", "customer_email": "not-an-email",
                   "product": random.choice(PRODUCTS), "topic": "login issue",
                   "message": "Bad email test", "created_at": _random_timestamp(),
                   "priority": "medium"},
        # missing message (empty)
        lambda i: {"ticket_id": f"TCK-BAD{i:03d}", "customer_email": f"empty{i}@example.com",
                   "product": random.choice(PRODUCTS), "topic": "refund request",
                   "message": "", "created_at": _random_timestamp(), "priority": "high"},
        # invalid priority value (out of allowed set)
        lambda i: {"ticket_id": f"TCK-BAD{i:03d}", "customer_email": f"pri{i}@example.com",
                   "product": random.choice(PRODUCTS), "topic": "warranty claim",
                   "message": "Bad priority test", "created_at": _random_timestamp(),
                   "priority": "URGENT!!"},
        # duplicate ticket_id (re-uses TCK-00001)
        lambda i: {"ticket_id": "TCK-00001", "customer_email": f"dup{i}@example.com",
                   "product": random.choice(PRODUCTS), "topic": "shipping delay",
                   "message": "Duplicate id test", "created_at": _random_timestamp(),
                   "priority": "low"},
    ]
    for i in range(1, n_bad + 1):
        maker = bad_makers[i % len(bad_makers)]
        tickets.append(maker(i))

    random.shuffle(tickets)
    return tickets


def generate_kb_articles():
    """Small knowledge base the RAG system will retrieve answers from."""
    return [
        {
            "doc_id": "KB-001",
            "title": "Shipping Delay Policy",
            "text": ("If an order is delayed by more than 5 business days, customers are "
                      "eligible for a shipping refund or expedited replacement shipping at "
                      "no extra cost. Always check the carrier tracking link first before "
                      "issuing a refund."),
        },
        {
            "doc_id": "KB-002",
            "title": "Refund Eligibility",
            "text": ("Products are eligible for a full refund within 30 days of delivery if "
                      "unused, or within 14 days if opened but defective. Defective items "
                      "must have photo evidence attached to the ticket before a refund is "
                      "approved."),
        },
        {
            "doc_id": "KB-003",
            "title": "Handling Defective Products",
            "text": ("For defective product reports (dead pixels, broken buttons, DOA units), "
                      "agents should first offer a free replacement before a refund. Escalate "
                      "to the hardware team if the same product model has 3+ defect reports in "
                      "a week."),
        },
        {
            "doc_id": "KB-004",
            "title": "Account Login Troubleshooting",
            "text": ("Login issues are usually caused by expired sessions or password resets "
                      "sent to an old email. Direct customers to the 'Forgot Password' flow, "
                      "and check the account_email field matches their current email before "
                      "resetting."),
        },
        {
            "doc_id": "KB-005",
            "title": "Duplicate Billing Charges",
            "text": ("Duplicate charges usually come from a failed payment retry. Agents should "
                      "check the billing_transactions table for two charges within 60 seconds "
                      "of each other before refunding the duplicate automatically."),
        },
        {
            "doc_id": "KB-006",
            "title": "Warranty Claim Process",
            "text": ("Standard warranty covers manufacturing defects for 12 months from "
                      "purchase date. Accidental damage is not covered. Ask for the order "
                      "confirmation email to verify the purchase date before approving a claim."),
        },
    ]


if __name__ == "__main__":
    tickets = generate_tickets()
    kb = generate_kb_articles()
    with open("data/raw_tickets.json", "w") as f:
        json.dump(tickets, f, indent=2)
    with open("data/kb_articles.json", "w") as f:
        json.dump(kb, f, indent=2)
    print(f"Generated {len(tickets)} tickets and {len(kb)} KB articles.")
