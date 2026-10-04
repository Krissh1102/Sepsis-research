from __future__ import annotations

import sys
from pathlib import Path

try:
    import sepsis_cdss  # noqa: F401
except ModuleNotFoundError:  # running as a plain script without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from sepsis_cdss.approval import Status
from sepsis_cdss.data import LABEL, generate_cohort, split_patients
from sepsis_cdss.features import build_features
from sepsis_cdss.pipeline import SepsisCDSS

st.set_page_config(page_title="Sepsis decision support", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Training models on a synthetic cohort (about 10 s)...")
def load_bundle(n_patients: int, seed: int) -> dict:
    hourly, patients = generate_cohort(n_patients, seed)
    train_ids, val_ids, test_ids = split_patients(patients, seed)
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    audit_path = out / f"ui_audit_{n_patients}_{seed}.jsonl"
    if audit_path.exists():
        audit_path.unlink()
    system = SepsisCDSS(audit_path=audit_path)
    fit_ids = train_ids + val_ids
    system.fit(hourly[hourly["patient_id"].isin(fit_ids)],
               patients[patients["patient_id"].isin(fit_ids)], val_ids)

    test = hourly[hourly["patient_id"].isin(test_ids)].sort_values(["patient_id", "ICULOS"])
    X, meta = build_features(test, system.medians)
    meta = meta.assign(risk=system.prediction.predict_proba(X))
    return dict(system=system, test=test, meta=meta, patients=patients.set_index("patient_id"),
                metrics=system.prediction.evaluate(X, meta[LABEL].to_numpy()), reports={})


def queue_table(b: dict) -> pd.DataFrame:
    thr = b["system"].prediction.threshold
    g = b["meta"].groupby("patient_id")
    q = pd.DataFrame({
        "stay_hours": g["ICULOS"].max(),
        "peak_risk": g["risk"].max(),
        "peak_hour": g.apply(lambda d: int(d.loc[d["risk"].idxmax(), "ICULOS"]), include_groups=False),
        "first_alert_hour": g.apply(
            lambda d: int(d.loc[d["risk"] >= thr, "ICULOS"].min()) if (d["risk"] >= thr).any() else 0,
            include_groups=False),
    })
    q["alert"] = q["peak_risk"] >= thr
    return q.sort_values("peak_risk", ascending=False)


STATUS_LABEL = {Status.PENDING_REVIEW: "Awaiting clinician", Status.APPROVED: "Approved",
                Status.OVERRIDDEN: "Overridden", Status.DEFERRED: "Deferred"}


# ------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Session")
    n_patients = st.select_slider("Synthetic cohort size", [300, 600, 1200], value=600)
    seed = st.number_input("Random seed", 0, 999, 0)
    clinician = st.text_input("Your clinician ID", placeholder="e.g. dr.rao",
                              help="Required to record a decision. It is written to the audit trail.")

bundle = load_bundle(int(n_patients), int(seed))
system: SepsisCDSS = bundle["system"]
thr = system.prediction.threshold

with st.sidebar:
    m = bundle["metrics"]
    st.divider()
    st.caption("Model on held-out synthetic patients")
    c1, c2 = st.columns(2)
    c1.metric("AUROC", f"{m['auroc']:.3f}")
    c2.metric("Alert threshold", f"{thr:.2f}")
    c1.metric("Sensitivity", f"{m['sensitivity']:.2f}")
    c2.metric("Specificity", f"{m['specificity']:.2f}")
    st.caption("Synthetic data is easy by construction, so these numbers say nothing about real-world accuracy.")

st.title("Sepsis decision support")
st.caption("AI proposes, a clinician decides. Every step is recorded in a tamper-evident audit trail.")

tab_review, tab_queue, tab_audit = st.tabs(["Case review", "Case queue", "Audit trail"])

# ------------------------------------------------------------------------- queue
q = queue_table(bundle)
with tab_queue:
    st.subheader("Patients ranked by peak risk")
    show = q.rename(columns={"stay_hours": "Hours in ICU", "peak_risk": "Peak risk", "peak_hour": "Peak at hour",
                             "first_alert_hour": "First alert at hour (0 = never)", "alert": "Ever alerted"})
    st.dataframe(show.head(60), width="stretch",
                 column_config={"Peak risk": st.column_config.ProgressColumn("Peak risk", min_value=0, max_value=1,
                                                                             format="%.2f")})
    st.caption(f"{int(q['alert'].sum())} of {len(q)} patients crossed the threshold at some point. "
               "Pick a patient on the Case review tab.")

# ------------------------------------------------------------------------- review
with tab_review:
    left, right = st.columns([1, 2])
    with left:
        pid = st.selectbox("Patient", list(q.index),
                           format_func=lambda p: f"{p}  (peak risk {q.loc[p, 'peak_risk']:.2f})")
        n_hours = int(q.loc[pid, "stay_hours"])
        default_hour = int(q.loc[pid, "first_alert_hour"]) or n_hours
        hour = st.slider("Assess at ICU hour", 1, n_hours, min(default_hour, n_hours),
                         help="The system only sees data up to this hour.")
        truth = bundle["patients"].loc[pid]
        with st.expander("Synthetic ground truth (not available in real life)"):
            st.write(f"Sepsis: **{'yes' if truth['sepsis'] else 'no'}**"
                     + (f", onset at hour {int(truth['onset']) + 1}, phenotype {truth['phenotype']}"
                        if truth["sepsis"] else ""))
            st.write(f"Died: **{'yes' if truth['died'] else 'no'}**")

    with right:
        tl = bundle["meta"][bundle["meta"]["patient_id"] == pid].set_index("ICULOS")["risk"]
        chart = pd.DataFrame({"sepsis risk": tl, "alert threshold": thr})
        st.caption("Risk over the stay (each point uses only data up to that hour)")
        st.line_chart(chart, height=220)

    case_id = f"{pid}-h{hour}"
    st.divider()
    if case_id not in bundle["reports"]:
        if st.button("Generate AI proposal for this patient and hour", type="primary"):
            stay = bundle["test"][(bundle["test"]["patient_id"] == pid) & (bundle["test"]["ICULOS"] <= hour)]
            bundle["reports"][case_id] = system.assess(stay, case_id)
            st.rerun()
        st.info("No proposal yet. Generating one scores the patient, explains the score, retrieves "
                "literature and runs the explanation check. It is then held for your review.")
    else:
        r = bundle["reports"][case_id]
        review = system.gate.get(case_id)

        banner = (st.error if r.alert_level == "HIGH" else st.success)
        banner(f"**{r.alert_level} alert.** Sepsis risk {r.sepsis_risk:.2f} against a threshold of {r.threshold:.2f}. "
               f"Proposed action: {r.recommended_action}.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Sepsis risk", f"{r.sepsis_risk:.2f}")
        k2.metric("Mortality risk (if septic)", f"{r.mortality_risk:.2f}")
        k3.metric("Phenotype cluster", r.phenotype)
        k4.metric("Explanation check", "Faithful" if r.verification["faithful"] else "Not faithful",
                  help=f"Removing the top {r.verification['k']} features moved the score "
                       f"{r.verification['ratio']:.1f}x more than removing random ones.")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Why this score** (positive pushes risk up)")
            drv = pd.DataFrame(r.drivers)
            st.bar_chart(drv.set_index("label")["contribution"], horizontal=True, height=240)
            st.dataframe(drv[["label", "value", "direction"]].rename(
                columns={"label": "Feature", "value": "Value", "direction": "Effect"}),
                hide_index=True, width="stretch")
        with d2:
            st.markdown("**Reasoning trace**")
            for i, s in enumerate(r.reasoning, 1):
                st.markdown(f"{i}. **{s['step']}** · _{s['source']}_  \n{s['content']}")

        with st.expander(f"Supporting literature ({len(r.evidence)})"):
            for e in r.evidence:
                st.markdown(f"**{e['source']}** · match {e['score']:.2f}")
                st.write(e["text"])
            st.caption("Entries are summaries of research papers, not clinical guidelines.")
        with st.expander("Caveats", expanded=True):
            for c in r.caveats:
                st.markdown(f"- {c}")

        st.divider()
        st.subheader("Clinician decision")
        if review.status == Status.PENDING_REVIEW:
            with st.form(f"decision-{case_id}"):
                decision = st.radio("Decision", ["Approve", "Override", "Defer"], horizontal=True,
                                    help="Approve accepts the proposed action. Override replaces it with "
                                         "yours and needs a reason. Defer leaves the case unresolved.")
                final_action = st.text_input("Your action (required for override)")
                rationale = st.text_area("Rationale (required for override)")
                submitted = st.form_submit_button("Record decision", type="primary")
            if submitted:
                try:
                    system.review(case_id, clinician, decision.lower(), rationale or None, final_action or None)
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))
        else:
            st.success(f"**{STATUS_LABEL[review.status]}** by {review.clinician_id} at {review.decision_ts[:19]} UTC")
            if review.rationale:
                st.write(f"Rationale: {review.rationale}")
            if review.status in (Status.APPROVED, Status.OVERRIDDEN):
                if st.button("Release final action"):
                    out = system.release(case_id)
                    st.info(f"Released: **{out['final_action']}** (authorised by {out['clinician_id']})")
            else:
                st.caption("Deferred cases cannot be released. Generate a new proposal at a later hour.")

        st.download_button("Download case report (Markdown)", r.to_markdown(), file_name=f"{case_id}.md",
                           mime="text/markdown")

# --------------------------------------------------------------------------- audit
with tab_audit:
    ok, bad = system.audit.verify()
    (st.success if ok else st.error)(
        f"Hash chain intact across {len(system.audit.records)} records." if ok
        else f"Chain broken at record {bad}: the log has been altered.")
    rows = [dict(seq=r["seq"], time=r["ts"][:19], actor=r["actor"], event=r["event"],
                 case=r["payload"].get("case_id", ""), hash=r["hash"][:12]) for r in system.audit.records]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Each record includes the hash of the one before it, so editing or deleting history is detectable.")
