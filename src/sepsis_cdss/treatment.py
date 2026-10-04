"""Treatment Recommendation Agent: interface only.

The review identifies Jeon et al. (ADT2R, an offline-RL decision transformer on
MIMIC-III) as the strongest basis for this component. Training it needs
credentialed MIMIC-III data, a validated action space and clinical oversight,
none of which exist in this repository, so no treatment logic is shipped. The
Protocol below is the slot where a validated policy would plug in; whatever it
proposes still has to pass the approval gate.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class TreatmentAgent(Protocol):
    def propose(self, patient_history: pd.DataFrame) -> dict | None:
        """Return a proposal dict (action, rationale, confidence) or None."""


class NullTreatmentAgent:
    """Default: proposes nothing and says so."""

    note = ("No treatment model is configured. Treatment decisions remain entirely with "
            "the treating clinician.")

    def propose(self, patient_history: pd.DataFrame) -> dict | None:
        return None
