import numpy as np
import pandas as pd
import pytest

from twin.cohort import generate_cohort
from twin.data.windows import H, build_patient
from twin.ehr.fhir import patient_bundle, risk_assessment
from twin.eval.metrics import clarke_summary, clarke_zones, excursion_onsets, false_alarm_episodes, lead_times
from twin.ingest.schema import diabetes_status
from twin.record import PatientRecord
from twin.twin import DigitalTwin


@pytest.fixture(scope="module")
def cohort():
    return generate_cohort(n=6, days=4, seed=3, chunk=6, verbose=False)


def test_clarke_zones_reference_points():
    ref = np.array([100, 100, 60, 250, 200, 50])
    pred = np.array([110, 150, 65, 150, 60, 200])
    assert list(clarke_zones(ref, pred)) == ["A", "B", "A", "D", "E", "E"]
    assert clarke_summary(np.array([100.0]), np.array([100.0]))["A"] == 100.0


def test_diabetes_status_follows_ada_cutoffs():
    assert diabetes_status(6.5) == "t2d"
    assert diabetes_status(5.9) == "prediabetes"
    assert diabetes_status(5.2, 92) == "normal"
    assert diabetes_status(5.4, 110) == "prediabetes"


def test_excursions_and_lead_time():
    cgm = np.array([150] * 10 + [190] * 4 + [150] * 10, dtype=float)
    onsets = excursion_onsets(cgm, high=True)
    assert list(onsets) == [10]
    anchors = np.arange(0, 10)
    prob = np.where(anchors >= 4, 0.9, 0.1)
    assert lead_times(anchors, prob, onsets, 0.5)[0] == 30.0  # first alarm at bin 4, onset at bin 10
    assert false_alarm_episodes(anchors, prob, np.ones(10, bool), 0.5) == 0


def test_cohort_streams_are_consistent(cohort):
    st, se, tr = cohort.static, cohort.series, cohort.truth
    assert len(st) == 6 and se.patient_id.nunique() == 6
    m = se.merge(tr, on=["patient_id", "ts"])
    for pid, g in m.groupby("patient_id"):
        a1c = st.set_index("patient_id").loc[pid, "hba1c_pct"]
        est = (g.glucose_true.mean() + 46.7) / 28.7  # ADAG: HbA1c is measured from the simulated body
        assert abs(a1c - est) < 0.8
    mard = (np.abs(m.cgm - m.glucose_true) / m.glucose_true).mean()
    assert 0.03 < mard < 0.15  # realistic CGM error


def test_windows_and_twin_forecast(cohort):
    pid = cohort.static.patient_id.iloc[0]
    row = cohort.static.set_index("patient_id").loc[pid].to_dict() | {"patient_id": pid}
    se = cohort.series[cohort.series.patient_id == pid]
    ev = cohort.events[cohort.events.patient_id == pid]
    arr = build_patient(row, se)
    assert arr.y.shape == (len(arr.anchors), H)
    assert len(arr.spike) == len(arr.spike_ok) == len(arr.anchors)
    rec = PatientRecord.from_frames(row, se, ev)
    tw = DigitalTwin.from_record(rec)
    tw.personalize(2 * 1440)
    tw.sync()
    fc = tw.forecast(arr.anchors[:20] * 5 + 2)
    assert fc.gi.shape == (20, 120) and np.isfinite(fc.gi).all()
    wi = tw.what_if(int(arr.anchors[10] * 5 + 2), walk_minutes=20)
    assert np.isfinite(wi["scenario"]).all()


def test_fhir_bundle_validates(cohort):
    p = cohort.profiles[0]
    b = patient_bundle(p, cohort.static.set_index("patient_id").loc[p.patient_id], cohort.lab_history, cohort.series)
    kinds = {e["resource"]["resourceType"] for e in b["entry"]}
    assert {"Patient", "Observation"} <= kinds
    if p.conditions:
        assert "Condition" in kinds
    ra = risk_assessment(p.patient_id, "2026-09-08T08:00:00", 0.7, 0.05, [150.0, 160.0])
    assert ra["prediction"][0]["probabilityDecimal"] == 0.7


def test_lab_history_is_quarterly(cohort):
    lh = cohort.lab_history
    assert lh.groupby("patient_id").size().eq(8).all()
    assert pd.api.types.is_datetime64_any_dtype(lh["date"])
