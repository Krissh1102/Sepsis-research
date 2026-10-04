"""Headless smoke test of the Streamlit UI (skipped if streamlit is not installed)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "sepsis_cdss" / "ui.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_app_loads_without_exception(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Case review", "Case queue", "Audit trail"]


def test_decision_flow_requires_clinician_and_records_audit(app):
    # generate proposal for the default (highest-risk) patient
    gen = [b for b in app.button if b.label.startswith("Generate AI proposal")]
    assert gen, "expected the generate button"
    gen[0].click().run()
    assert not app.exception
    assert any("alert" in e.value.lower() for e in list(app.error) + list(app.success))

    # submit without a clinician ID -> blocked
    app.radio[0].set_value("Approve")
    [b for b in app.button if b.label == "Record decision"][0].click().run()
    assert not app.exception
    assert any("clinician_id" in e.value for e in app.error)

    # with an ID -> recorded and chain still valid
    [t for t in app.text_input if t.label == "Your clinician ID"][0].set_value("dr.test").run()
    [b for b in app.button if b.label == "Record decision"][0].click().run()
    assert not app.exception
    assert any("Approved" in s.value for s in app.success)
    assert any("Hash chain intact" in s.value for s in app.success)
