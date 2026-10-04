"""Structured Reasoning Agent.

Deterministic, template-based assembly of an auditable reasoning trace (no LLM).
Each step states what was observed, which component produced it, and its
limits. Inspired by the auditable multi-step thought process in Pouplin et al.,
but without a generative model so it cannot hallucinate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .features import describe_feature


@dataclass
class CaseReport:
    case_id: str
    sepsis_risk: float
    threshold: float
    alert_level: str
    recommended_action: str
    mortality_risk: float | None
    phenotype: int | None
    drivers: list[dict]
    mortality_drivers: list[dict]
    evidence: list[dict]
    verification: dict
    reasoning: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        L = [f"# Case {self.case_id}", "",
             f"**Alert level:** {self.alert_level}  ",
             f"**Sepsis risk:** {self.sepsis_risk:.3f} (alert threshold {self.threshold:.3f})  ",
             f"**Proposed action (needs clinician decision):** {self.recommended_action}", ""]
        if self.mortality_risk is not None:
            L += [f"**Mortality risk (septic-cohort model):** {self.mortality_risk:.3f}  ",
                  f"**Data-driven phenotype cluster:** {self.phenotype}", ""]
        L += ["## Top drivers of the sepsis score", ""]
        for d in self.drivers:
            L.append(f"- {d['label']} = {d['value']:.3g} ({d['direction']}, contribution {d['contribution']:+.3f})")
        L += ["", "## Reasoning trace", ""]
        for i, s in enumerate(self.reasoning, 1):
            L.append(f"{i}. **{s['step']}** ({s['source']}): {s['content']}")
        L += ["", "## Supporting literature (retrieved, verbatim KB entries)", ""]
        for e in self.evidence:
            L.append(f"- [{e['source']}] (match {e['score']:.2f}) {e['text']}")
        v = self.verification
        L += ["", "## Explanation check", "",
              f"Removing the top-{v['k']} features changed the score {v['ratio']:.1f}x more than removing "
              f"{v['k']} random features -> {'faithful' if v['faithful'] else 'NOT faithful'}.", "",
              "## Caveats", ""]
        L += [f"- {c}" for c in self.caveats]
        return "\n".join(L) + "\n"


def top_drivers(row, contribs, columns, k=5) -> list[dict]:
    order = sorted(range(len(columns)), key=lambda i: -abs(contribs[i]))[:k]
    out = []
    for i in order:
        c = float(contribs[i])
        out.append(dict(feature=columns[i], label=describe_feature(columns[i]),
                        value=float(row[i]), contribution=c,
                        direction="raises risk" if c > 0 else "lowers risk"))
    return out


def evidence_query(drivers: list[dict], mortality_drivers: list[dict]) -> str:
    return " ".join(d["label"] for d in drivers + mortality_drivers)


def build_reasoning(risk, threshold, flagged, drivers, mortality_risk, phenotype,
                    mortality_drivers, evidence, verification) -> list[dict]:
    steps = [dict(
        step="Score", source="SepsisPredictionAgent",
        content=(f"Model risk {risk:.3f} is {'at or above' if flagged else 'below'} the validation-tuned "
                 f"threshold {threshold:.3f}."))]
    steps.append(dict(
        step="Drivers", source="TreeSHAP",
        content="Largest contributions: " + "; ".join(
            f"{d['label']} {d['direction']}" for d in drivers[:3]) + "."))
    if mortality_risk is not None:
        steps.append(dict(
            step="Outcome and phenotype", source="DEKD",
            content=(f"Mortality model (trained on septic patients only) gives {mortality_risk:.3f}; patient "
                     f"falls in SHAP-space cluster {phenotype}. Leading mortality drivers: " + ", ".join(
                         d["label"] for d in mortality_drivers[:3]) + ".")))
    steps.append(dict(
        step="Evidence", source="EvidenceRetriever",
        content=(f"Retrieved {len(evidence)} knowledge-base entries: " + ", ".join(e["source"] for e in evidence)
                 if evidence else "No relevant knowledge-base entry found.")))
    steps.append(dict(
        step="Verification", source="deletion test",
        content=(f"Explanation {'passed' if verification['faithful'] else 'failed'} the deletion test "
                 f"(ratio {verification['ratio']:.1f}).")))
    steps.append(dict(
        step="Gate", source="ApprovalGate",
        content="Proposal is held for clinician review; nothing is actioned automatically."))
    return steps
