"""Evidence Retrieval Agent.

A small, fully local TF-IDF retriever (no LLM, no network, so no private data
leaves the machine and nothing can be hallucinated: every returned passage is a
verbatim entry from the knowledge base with its source).

The built-in knowledge base holds short summaries, written for this project, of
the findings in the reviewed papers. It is evidence about *methods and
predictors*, not a clinical guideline. To ground the system in real guidelines,
put one ``.md`` / ``.txt`` file per guideline excerpt in a folder and load it
with ``EvidenceRetriever.from_directory``; they are searched the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Doc:
    id: str
    source: str
    text: str


BUILTIN_KB = [
    Doc("choudhury2025", "Choudhury et al., ICETCI 2025",
        "XGBoost with a focal weighted binary cross-entropy loss predicts sepsis from ICU "
        "vitals and labs on PhysioNet 2019 data. SHAP showed the important features spanned "
        "cardiovascular, respiratory, coagulation, liver, renal and biochemical groups. "
        "Temperature, ICU length of stay and the missingness of FiO2 were influential."),
    Doc("liu2022", "Liu et al., IEEE JBHI 2022",
        "Time-dependent features (previous values, differences between time points, counts) "
        "improve XGBoost sepsis prediction. Top features included lactate, temperature, "
        "neutrophils, potassium, glasgow coma score, age and weight."),
    Doc("wang2022", "Wang & Yao, IEEE JBHI 2022",
        "A multi-branching temporal convolutional network with a missing-value mask handles "
        "missing and imbalanced EHR data. Missingness is informative, and high prediction "
        "uncertainty was associated with more missing vital signs."),
    Doc("vanwyk2019", "van Wyk et al., IEEE JBHI 2019",
        "Hierarchical analysis of continuous bedside physiological data predicted sepsis about "
        "205 minutes earlier than SIRS criteria. Gains in sensitivity came with lower "
        "specificity, so alert thresholds need clinical tuning."),
    Doc("jazayeri2021", "Jazayeri et al., IEEE JBHI 2021",
        "Failures of organ systems that occur close together in time carry predictive "
        "information about sepsis outcome beyond individual failures. A 90 minute aggregation "
        "window worked best."),
    Doc("jin2026", "Jin et al., IEEE JBHI 2026",
        "A masked-autoencoder pretrained teacher-student multitask model on 48 hours of "
        "vasoactive-inotropic dosing predicted sepsis ICU mortality (AUROC 0.829). SOFA score "
        "had the largest SHAP impact, followed by LODS. Marital status and insurance type also "
        "shifted risk, and the authors treat them as non-causal proxies for access to care."),
    Doc("he2025", "He, Liu & Guo, IEEE JBHI 2025",
        "Clustering patients in SHAP space reveals sepsis phenotypes with different risk "
        "mappings. Mechanical ventilation, age, blood urea nitrogen, glasgow coma score and "
        "urine output were leading mortality predictors. Distilling the dynamic ensemble into "
        "one student model restores explainability."),
    Doc("peng2023", "Peng et al., IEEE JBHI 2023",
        "In preterm infants, heart rate variability, respiration and motion features from "
        "bedside monitors predicted late-onset sepsis (AUC 0.88). Motion changes appeared "
        "earlier than cardiorespiratory changes."),
    Doc("jeon2025", "Jeon, Choi & Suk, IEEE TNNLS 2025",
        "An adaptive decision transformer trained offline on MIMIC-III recommends sepsis "
        "treatments while accounting for patient heterogeneity. Recommendations are research "
        "outputs and were not clinically validated."),
    Doc("li2025", "Li et al., IEEE JBHI 2025",
        "Large language models used for evidence-based medicine tasks show hallucination and "
        "factual inconsistency. Knowledge-guided prompting helps but human evaluation remains "
        "essential before clinical use."),
    Doc("pouplin2025", "Pouplin et al., IEEE JBHI 2025",
        "Retrieval-augmented thought processes let language models reason over private "
        "clinical records with an auditable, multi-step thought trace, improving question "
        "answering accuracy over plain retrieval-augmented generation."),
]


class EvidenceRetriever:
    def __init__(self, docs: list[Doc] | None = None):
        self.docs = list(docs if docs is not None else BUILTIN_KB)
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vec.fit_transform([d.text for d in self.docs])

    @classmethod
    def from_directory(cls, directory: str | Path, include_builtin: bool = True) -> "EvidenceRetriever":
        docs = list(BUILTIN_KB) if include_builtin else []
        for f in sorted(Path(directory).glob("*")):
            if f.suffix.lower() in {".md", ".txt"}:
                docs.append(Doc(f.stem, f.name, f.read_text(encoding="utf-8")))
        return cls(docs)

    def search(self, query: str, k: int = 3, min_score: float = 0.02) -> list[dict]:
        if not query.strip():
            return []
        sims = cosine_similarity(self._vec.transform([query]), self._matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [dict(id=self.docs[i].id, source=self.docs[i].source,
                     score=round(float(sims[i]), 4), text=self.docs[i].text)
                for i in order if sims[i] >= min_score]
