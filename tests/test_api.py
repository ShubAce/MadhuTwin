import pytest

from twin.paths import ARTIFACTS

pytestmark = pytest.mark.skipif(not (ARTIFACTS / "demo" / "index.json").exists(), reason="demo cohort not built")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from twin.service.app import app
    return TestClient(app)


def test_panel_and_patient(client):
    rows = client.get("/api/patients?clock=500").json()
    assert len(rows) >= 3
    pid = rows[0]["id"]
    assert client.get(f"/api/patients/{pid}").status_code == 200
    st = client.get(f"/api/patients/{pid}/state?clock=500").json()
    assert st["forecast"] is None or len(st["forecast"]["q50"]) == 24
    agp = client.get(f"/api/patients/{pid}/agp?clock=500").json()
    assert abs(agp["tir"] + agp["tar1"] + agp["tar2"] + agp["tbr1"] + agp["tbr2"] - 100) < 0.5


def test_whatif_walk_lowers_peak(client):
    pid = client.get("/api/patients").json()[0]["id"]
    body = {"clock": 780, "meal": {"food": "rice_sambar", "portion": 1.0, "in_min": 0}}
    no_walk = client.post(f"/api/patients/{pid}/whatif", json=body).json()
    walk = client.post(f"/api/patients/{pid}/whatif", json=body | {"walk": {"minutes": 30, "delay_min": 20}}).json()
    assert walk["scenario_peak"] <= no_walk["scenario_peak"]


def test_therapy_simulator_dose_response(client):
    idx = [p["id"] for p in client.get("/api/patients").json()]
    plans = {pid: client.get(f"/api/patients/{pid}/therapy?clock=1830").json() for pid in idx}
    pid = next(p for p, pl in plans.items() if any(d["unit"] == "IU" for d in pl["doses"]))
    usual = plans[pid]["doses"]
    r = client.post(f"/api/patients/{pid}/therapy", json={"clock": 1830, "doses": usual}).json()
    assert r["usual"] == r["scenario"]  # an unchanged plan reproduces the usual day
    less = client.post(f"/api/patients/{pid}/therapy", json={"clock": 1830, "doses": [d | {"amount": d["amount"] * 0.5} for d in usual]}).json()
    assert less["scenario_stats"]["mean"] >= less["usual_stats"]["mean"]  # less insulin, higher glucose
    rows = r["dose_response"]["rows"]
    assert rows[0]["mean"] >= rows[-1]["mean"]  # monotone dose response
    skip = client.post(f"/api/patients/{pid}/therapy", json={"clock": 1830, "pattern": "skip_lunch"}).json()
    assert skip["usual_stats"]["mean"] <= r["usual_stats"]["mean"]
    assert client.post(f"/api/patients/{pid}/therapy", json={"pattern": "bogus"}).status_code == 422


def test_fhir_has_risk_assessment(client):
    pid = client.get("/api/patients").json()[0]["id"]
    b = client.get(f"/api/patients/{pid}/fhir?clock=500").json()
    assert any(e["resource"]["resourceType"] == "RiskAssessment" for e in b["entry"])


def test_ask_is_grounded_and_audited(client):
    pid = client.get("/api/patients").json()[0]["id"]
    r = client.post(f"/api/patients/{pid}/ask", json={"question": "Summarise the last week", "clock": 1500}).json()
    assert "time in range" in r["answer"].lower()
    log = client.get("/api/audit").json()
    assert any(row["patient"] == pid for row in log)
