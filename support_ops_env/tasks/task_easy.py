"""
Task 1 (Easy): Email Classification
Given a customer email, classify it into one of three categories:
  - billing
  - technical
  - general
"""

from typing import Optional
from pydantic import BaseModel


# ─────────────────────────────────────────────
# Mock Dataset
# ─────────────────────────────────────────────
MOCK_EMAILS = [
    {
        "id": "email_001",
        "text": (
            "Hi, I was charged twice for my subscription this month. "
            "Can you please refund one of those payments? My order ID is #445566."
        ),
        "label": "billing",
    },
    {
        "id": "email_002",
        "text": (
            "The mobile app keeps crashing whenever I try to open the dashboard. "
            "I'm on iOS 17.2 and version 3.4.1 of your app. Please fix this urgently."
        ),
        "label": "technical",
    },
    {
        "id": "email_003",
        "text": (
            "Hello, I'd like to know your office hours and whether you offer "
            "live chat support on weekends."
        ),
        "label": "general",
    },
    {
        "id": "email_004",
        "text": (
            "My invoice shows a charge of $49.99 but my plan is supposed to be $29.99. "
            "Please explain this discrepancy and issue a credit."
        ),
        "label": "billing",
    },
    {
        "id": "email_005",
        "text": (
            "I cannot log in to my account. It says 'invalid credentials' even though "
            "I reset my password twice today. Very frustrated."
        ),
        "label": "technical",
    },
    {
        "id": "email_006",
        "text": (
            "I'm interested in your enterprise plan. Can someone from sales contact me "
            "to discuss pricing for 200 seats?"
        ),
        "label": "general",
    },
    {
        "id": "email_007",
        "text": (
            "The API is returning 500 errors intermittently when calling /v2/reports. "
            "This is breaking our production pipeline."
        ),
        "label": "technical",
    },
    {
        "id": "email_008",
        "text": (
            "I cancelled my subscription but was still charged this month. "
            "I need a full refund immediately."
        ),
        "label": "billing",
    },
    {
        "id": "email_009",
        "text": (
            "Can you send me documentation on how to integrate your product "
            "with Salesforce?"
        ),
        "label": "general",
    },
    {
        "id": "email_010",
        "text": (
            "Data export is failing with a timeout error for datasets larger than 10k rows. "
            "This started after your last update."
        ),
        "label": "technical",
    },
]

VALID_LABELS = {"billing", "technical", "general"}

# Semantic similarity groups: labels that are "close" to each other
SEMANTIC_GROUPS = {
    "billing": {"billing", "general"},      # billing ↔ general is partially close
    "technical": {"technical", "general"},  # tech ↔ general is partially close
    "general": {"general", "billing", "technical"},
}


class EasyObservation(BaseModel):
    """Observation for the email classification task."""
    task_id: str = "task_easy"
    email_id: str
    email_text: str
    valid_labels: list = ["billing", "technical", "general"]
    instruction: str = (
        "Classify the following customer email into one of the valid_labels. "
        "Return only the label string."
    )


class EasyAction(BaseModel):
    """Action for the email classification task."""
    label: str  # must be one of "billing", "technical", "general"


def get_task_data() -> list[dict]:
    """Return all mock email samples."""
    return MOCK_EMAILS


def get_observation(sample: dict) -> EasyObservation:
    """Build an Observation from a sample dict."""
    return EasyObservation(
        email_id=sample["id"],
        email_text=sample["text"],
    )
