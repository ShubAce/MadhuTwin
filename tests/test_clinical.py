import numpy as np
import pandas as pd

from twin.eval.clinical import (
    calibration_curve,
    crossfit_recalibrate,
    decision_curve,
    platt_apply,
    platt_fit,
    subgroup_table,
)
from twin.ingest.bigideas import rmssd_5min


def test_calibration_curve_of_calibrated_scores_follows_diagonal():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = rng.uniform(0, 1, p.size) < p
    pts = calibration_curve(y, p)
    assert len(pts) == 10
    assert max(abs(q["predicted"] - q["observed"]) for q in pts) < 0.03


def test_decision_curve_net_benefit():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    perfect = decision_curve(y, y.astype(float), thresholds=np.array([0.2, 0.5]))
    assert all(abs(r["model"] - 0.2) < 1e-12 for r in perfect)  # perfect alerts: net benefit = prevalence
    assert abs(perfect[0]["alert_all"] - (0.2 - 0.8 * 0.25)) < 1e-12  # alert everyone at pt = 0.2
    useless = decision_curve(y, np.ones(10), thresholds=np.array([0.5]))
    assert useless[0]["model"] == useless[0]["alert_all"]


def test_platt_recalibration_fixes_overconfidence():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0.01, 0.6, 20_000)
    y = rng.uniform(0, 1, p_true.size) < p_true
    over = 1 / (1 + np.exp(-2.0 * np.log(p_true / (1 - p_true)) - 1.0))  # an overconfident model
    fold = np.arange(p_true.size) % 5
    fixed = crossfit_recalibrate(over, y, np.ones(p_true.size, bool), fold)
    err = lambda s: max(abs(q["predicted"] - q["observed"]) for q in calibration_curve(y, s))  # noqa: E731
    assert err(over) > 0.1 and err(fixed) < 0.03
    for f in range(5):  # monotone within each fold, so a fold's alerts are unchanged
        m = fold == f
        assert np.all(np.diff(fixed[m][np.argsort(over[m])]) >= -1e-12)
    assert platt_fit(y[:10], over[:10]) == [1.0, 0.0]  # too little data: identity map
    assert np.allclose(platt_apply(np.array([0.2, 0.7]), [1.0, 0.0]), [0.2, 0.7])


def test_subgroup_table_pools_squared_errors():
    df = pd.DataFrame({"sex": ["F", "F", "M"], "se60": [4.0, 16.0, 9.0], "n60": [1, 1, 1], "se120": [0.0, 0.0, 0.0], "n120": [1, 1, 1]})
    rows = {r["group"]: r for r in subgroup_table(df, ["sex"])}
    assert rows["F"]["patients"] == 2 and abs(rows["F"]["rmse_60"] - np.sqrt(10)) < 1e-12
    assert abs(rows["M"]["rmse_60"] - 3.0) < 1e-12


def test_rmssd_from_beat_intervals():
    # alternating 800 / 850 ms beats: every successive difference is 50 ms, so RMSSD = 50 ms
    ibi = np.tile([0.80, 0.85], 200)
    ts = pd.Timestamp("2020-02-13 08:00") + pd.to_timedelta(np.cumsum(ibi), unit="s")
    r = rmssd_5min(pd.DataFrame({"ts": ts, "v": ibi})).dropna()
    assert len(r) >= 1 and np.allclose(r.to_numpy(), 50.0, atol=1e-6)
    # a dropped beat (gap of two intervals) is not counted as a successive difference
    ts_gap = ts.delete(100)
    r2 = rmssd_5min(pd.DataFrame({"ts": ts_gap, "v": np.delete(ibi, 100)})).dropna()
    assert np.allclose(r2.to_numpy(), 50.0, atol=1e-6)
