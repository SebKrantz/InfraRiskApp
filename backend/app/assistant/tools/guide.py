"""The read_guide tool — detailed recipes the system prompt only points at."""

from __future__ import annotations

from typing import Any

from .. import guides
from ..conversations import Conversation
from . import tool


@tool(
    "read_guide",
    "Detailed how-to guides. ALWAYS read the matching guide before starting the "
    "task: 'exposure_analysis' before your first analysis of a dataset (the "
    "exact affected-rule, points vs lines, how to choose and justify a "
    "threshold), 'vulnerability_analysis' before any damage-cost work (curve "
    "formats, what replacement_value means, the two caveats that must reach the "
    "write-up), 'multi_hazard' before comparing return periods or climate "
    "scenarios, 'reports' before writing any Word/Excel/CSV deliverable, and "
    "'figures' when choosing what to plot.",
    {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "enum": [
                    "exposure_analysis",
                    "vulnerability_analysis",
                    "multi_hazard",
                    "reports",
                    "figures",
                ],
            },
        },
        "required": ["topic"],
    },
)
def read_guide(conv: Conversation, topic: str) -> dict[str, Any]:
    text = guides.TOPICS.get(topic)
    if text is None:
        raise ValueError(f"unknown topic {topic!r}; one of: {sorted(guides.TOPICS)}")
    return {"topic": topic, "guide": text}
