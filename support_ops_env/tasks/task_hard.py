"""
Task 3 (Hard): Response Generation
Given a customer complaint, generate a structured JSON response:
  {
    "tone": "...",              # empathetic / formal / apologetic / assertive
    "resolution_steps": "...", # string with actionable steps
    "escalation": true/false   # whether to escalate to a human agent
  }

Reward factors:
  - Structure correctness
  - Semantic similarity to reference answer
  - Penalty for unsafe/irrelevant responses
"""

from typing import Optional
from pydantic import BaseModel


# ─────────────────────────────────────────────
# Mock Dataset
# ─────────────────────────────────────────────
MOCK_COMPLAINTS = [
    {
        "id": "complaint_001",
        "text": (
            "I've been charged three times for the same order and nobody is helping me! "
            "This is absolutely unacceptable. I want my money back NOW."
        ),
        "reference_response": {
            "tone": "apologetic",
            "resolution_steps": (
                "1. Sincerely apologize for the triple charge. "
                "2. Initiate immediate refund for the two duplicate charges. "
                "3. Send confirmation email with refund timeline (3-5 business days). "
                "4. Offer a discount code as goodwill gesture."
            ),
            "escalation": True,
        },
        "valid_tones": ["empathetic", "formal", "apologetic", "assertive"],
        "requires_escalation": True,
    },
    {
        "id": "complaint_002",
        "text": (
            "Your app has been crashing for two days straight and I can't do my work. "
            "This is costing my business money. Fix it or I'm cancelling."
        ),
        "reference_response": {
            "tone": "empathetic",
            "resolution_steps": (
                "1. Acknowledge the disruption to their business. "
                "2. Ask for device model, OS version, and app version. "
                "3. Provide immediate workaround (use web version). "
                "4. Escalate to engineering team with priority flag. "
                "5. Commit to status updates every 4 hours."
            ),
            "escalation": True,
        },
        "valid_tones": ["empathetic", "formal", "apologetic", "assertive"],
        "requires_escalation": True,
    },
    {
        "id": "complaint_003",
        "text": (
            "I cancelled my account last month but I'm still getting charged. "
            "I have the cancellation confirmation email. This needs to stop."
        ),
        "reference_response": {
            "tone": "apologetic",
            "resolution_steps": (
                "1. Apologize for the billing error post-cancellation. "
                "2. Request the cancellation confirmation email for verification. "
                "3. Process full refund for all charges post-cancellation. "
                "4. Confirm account closure and remove payment details."
            ),
            "escalation": False,
        },
        "valid_tones": ["empathetic", "formal", "apologetic", "assertive"],
        "requires_escalation": False,
    },
    {
        "id": "complaint_004",
        "text": (
            "Your customer service rep was rude to me on the phone yesterday. "
            "She hung up on me while I was still talking. I want to complain formally."
        ),
        "reference_response": {
            "tone": "formal",
            "resolution_steps": (
                "1. Take the complaint seriously and formally acknowledge it. "
                "2. Apologize unconditionally for the agent's behavior. "
                "3. Log a formal complaint and provide a complaint reference number. "
                "4. Inform the customer the matter will be reviewed within 48 hours. "
                "5. Follow up with outcome of the internal review."
            ),
            "escalation": True,
        },
        "valid_tones": ["empathetic", "formal", "apologetic", "assertive"],
        "requires_escalation": True,
    },
    {
        "id": "complaint_005",
        "text": (
            "Your system lost all my saved data after the latest update. "
            "6 months of work is gone. This is a disaster."
        ),
        "reference_response": {
            "tone": "apologetic",
            "resolution_steps": (
                "1. Express deep apology for the data loss. "
                "2. Immediately escalate to data recovery team. "
                "3. Check backup systems for the customer's last known good state. "
                "4. Provide recovery timeline and keep customer updated. "
                "5. Offer service credit for the inconvenience."
            ),
            "escalation": True,
        },
        "valid_tones": ["empathetic", "formal", "apologetic", "assertive"],
        "requires_escalation": True,
    },
    {
        "id": "complaint_006",
        "text": (
            "I've been waiting 3 weeks for a response to my support ticket. "
            "This level of service is unacceptable for a paid customer."
        ),
        "reference_response": {
            "tone": "empathetic",
            "resolution_steps": (
                "1. Apologize for the unacceptably long wait time. "
                "2. Take immediate ownership of the original ticket. "
                "3. Provide a direct resolution or clear timeline within 24 hours. "
                "4. Explain the steps being taken to prevent this in future."
            ),
            "escalation": False,
        },
        "valid_tones": ["empathetic", "formal", "apologetic", "assertive"],
        "requires_escalation": False,
    },
    {
        "id": "complaint_007",
        "text": (
            "The integration with our CRM broke after your update and we never consented "
            "to breaking changes without notice. We need compensation."
        ),
        "reference_response": {
            "tone": "formal",
            "resolution_steps": (
                "1. Acknowledge the breaking change and apologize for lack of prior notice. "
                "2. Provide technical documentation for the new API endpoints. "
                "3. Offer a free migration assistance call with an engineer. "
                "4. Review compensation options and respond within 2 business days."
            ),
            "escalation": True,
        },
        "valid_tones": ["empathetic", "formal", "apologetic", "assertive"],
        "requires_escalation": True,
    },
    {
        "id": "complaint_008",
        "text": (
            "Why can't I download my invoice? The button is broken on your website. "
            "I need this for my taxes, it's due tomorrow."
        ),
        "reference_response": {
            "tone": "empathetic",
            "resolution_steps": (
                "1. Empathize with the urgency given the tax deadline. "
                "2. Manually generate and email the invoice within 30 minutes. "
                "3. Log the bug with the website team for immediate fix. "
                "4. Provide a direct download link as backup."
            ),
            "escalation": False,
        },
        "valid_tones": ["empathetic", "formal", "apologetic", "assertive"],
        "requires_escalation": False,
    },
]

VALID_TONES = {"empathetic", "formal", "apologetic", "assertive"}

# Unsafe keywords that trigger a safety penalty
UNSAFE_KEYWORDS = [
    "kill", "die", "threat", "lawsuit", "illegal", "fraud", "sue",
    "scam", "fake", "lie", "stupid", "idiot", "hate",
]


class HardObservation(BaseModel):
    """Observation for the response generation task."""
    task_id: str = "task_hard"
    complaint_id: str
    complaint_text: str
    valid_tones: list = ["empathetic", "formal", "apologetic", "assertive"]
    instruction: str = (
        "Generate a structured customer support response for the given complaint. "
        "Return a JSON object with keys: 'tone' (one of valid_tones), "
        "'resolution_steps' (string with numbered steps), "
        "and 'escalation' (boolean, true if human agent needed)."
    )


class HardAction(BaseModel):
    """Action for the response generation task."""
    tone: str                   # one of VALID_TONES
    resolution_steps: str       # actionable resolution string
    escalation: bool            # whether to escalate


def get_task_data() -> list[dict]:
    """Return all mock complaint samples."""
    return MOCK_COMPLAINTS


def get_observation(sample: dict) -> HardObservation:
    """Build an Observation from a sample dict."""
    return HardObservation(
        complaint_id=sample["id"],
        complaint_text=sample["text"],
    )
