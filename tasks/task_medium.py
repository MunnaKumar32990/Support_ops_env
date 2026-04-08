"""
Task 2 (Medium): Ticket Prioritization
Given a support ticket with urgency signals, output a priority level:
  - low
  - medium
  - high

Penalty applies for underestimating high-priority issues.
"""

from typing import Optional
from pydantic import BaseModel


# ─────────────────────────────────────────────
# Mock Dataset
# ─────────────────────────────────────────────
MOCK_TICKETS = [
    {
        "id": "ticket_001",
        "text": (
            "Our entire production system is down. All users are affected. "
            "We are losing $10,000 per hour. Need immediate attention!"
        ),
        "priority": "high",
        "urgency_signals": ["production down", "all users affected", "revenue loss"],
    },
    {
        "id": "ticket_002",
        "text": (
            "The report export feature is slow — takes about 3 minutes instead of 30 seconds. "
            "Annoying but we can still work."
        ),
        "priority": "medium",
        "urgency_signals": ["performance degradation", "non-blocking"],
    },
    {
        "id": "ticket_003",
        "text": (
            "Could you update the color theme of the dashboard button? "
            "I'd prefer blue over green."
        ),
        "priority": "low",
        "urgency_signals": ["cosmetic change", "no business impact"],
    },
    {
        "id": "ticket_004",
        "text": (
            "Security vulnerability detected: SQL injection possible in the search API. "
            "Our security team confirmed this. Immediate patch required."
        ),
        "priority": "high",
        "urgency_signals": ["security vulnerability", "confirmed exploit", "immediate patch"],
    },
    {
        "id": "ticket_005",
        "text": (
            "About 20% of our users can't complete checkout. We're missing significant sales. "
            "Error: 'Payment gateway timeout'."
        ),
        "priority": "high",
        "urgency_signals": ["partial outage", "revenue impact", "checkout failure"],
    },
    {
        "id": "ticket_006",
        "text": (
            "The CSV import feature fails for files with more than 5000 rows. "
            "We have a workaround (split the file) but it's tedious."
        ),
        "priority": "medium",
        "urgency_signals": ["feature limitation", "workaround available"],
    },
    {
        "id": "ticket_007",
        "text": (
            "Could you add a dark mode? Several colleagues have asked about this. "
            "Not urgent at all — just a nice-to-have."
        ),
        "priority": "low",
        "urgency_signals": ["feature request", "cosmetic", "nice-to-have"],
    },
    {
        "id": "ticket_008",
        "text": (
            "Emails sent from your platform are landing in spam. "
            "Our marketing campaigns are completely blocked. This is urgent."
        ),
        "priority": "high",
        "urgency_signals": ["email deliverability", "campaigns blocked", "urgent"],
    },
    {
        "id": "ticket_009",
        "text": (
            "The tooltip text on the settings page has a typo: 'Sav' instead of 'Save'. "
            "Minor issue."
        ),
        "priority": "low",
        "urgency_signals": ["typo", "cosmetic", "minor"],
    },
    {
        "id": "ticket_010",
        "text": (
            "Webhook delivery is delayed by 15-20 minutes. "
            "Our downstream integrations are affected but not completely broken."
        ),
        "priority": "medium",
        "urgency_signals": ["delay", "integration impact", "not critical"],
    },
]

VALID_PRIORITIES = {"low", "medium", "high"}

# Numeric mapping for scoring
PRIORITY_SCORE = {"low": 0, "medium": 1, "high": 2}


class MediumObservation(BaseModel):
    """Observation for the ticket prioritization task."""
    task_id: str = "task_medium"
    ticket_id: str
    ticket_text: str
    urgency_signals: list
    valid_priorities: list = ["low", "medium", "high"]
    instruction: str = (
        "Analyze the support ticket and its urgency signals. "
        "Assign a priority level: 'low', 'medium', or 'high'. "
        "Return only the priority string."
    )


class MediumAction(BaseModel):
    """Action for the ticket prioritization task."""
    priority: str  # must be one of "low", "medium", "high"


def get_task_data() -> list[dict]:
    """Return all mock ticket samples."""
    return MOCK_TICKETS


def get_observation(sample: dict) -> MediumObservation:
    """Build an Observation from a sample dict."""
    return MediumObservation(
        ticket_id=sample["id"],
        ticket_text=sample["text"],
        urgency_signals=sample["urgency_signals"],
    )
