"""The optional LLM layer's tools run offline against the twin; no API call is made here."""

import json

import pytest

from twin.paths import ARTIFACTS

pytestmark = pytest.mark.skipif(not (ARTIFACTS / "demo" / "index.json").exists(), reason="demo cohort not built")
anthropic = pytest.importorskip("anthropic")


@pytest.fixture(scope="module")
def patient():
    from twin.service.store import Store

    store = Store(ARTIFACTS / "demo", ARTIFACTS / "results")
    return store.get(store.order[0])


def test_tool_schemas_are_well_formed():
    from twin.service.llm import TOOLS

    names = [t["name"] for t in TOOLS]
    assert len(names) == len(set(names))
    for t in TOOLS:
        assert t["input_schema"]["type"] == "object"
        assert t["input_schema"].get("additionalProperties") is False
        assert len(t["description"]) > 40


@pytest.mark.parametrize("name,args", [
    ("get_current_forecast", {}),
    ("get_glucose_summary", {"days": 7}),
    ("get_patient_record", {}),
    ("get_twin_physiology", {}),
    ("rank_meals", {}),
    ("simulate_scenario", {"food": "idli_sambar", "portion": 1.0, "walk_minutes": 15}),
    ("get_insights", {}),
])
def test_tools_return_serialisable_results(patient, name, args):
    from twin.service.llm import _run_tool

    out = _run_tool(patient, 780, name, args)
    assert isinstance(out, dict) and out
    json.dumps(out, default=float)


def test_llm_disabled_by_default(monkeypatch):
    from twin.service import llm

    monkeypatch.delenv("MADHUTWIN_LLM", raising=False)
    assert llm.available() is False
