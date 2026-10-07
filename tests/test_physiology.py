import numpy as np
import pytest

from twin.physiology.inputs import Dose, Meal, build_inputs
from twin.physiology.model import default_physiology, simulate


def _scenario(n: int = 3, T: int = 600):
    p = default_physiology(n, Gb=[90, 120, 170], Ib=[6, 10, 12], SI=[6e-4, 3e-4, 1.5e-4], beta=[0.13, 0.09, 0.03],
                           renal_thr_base=[180, 180, 90])
    meals = [[Meal(30, 75, gi=100)], [Meal(60, 85, 12, 9, 8, 72)], [Meal(20, 60, 10, 20, 5, 55)]]
    doses = [[], [Dose(240, "glimepiride", 2.0)], [Dose(10, "insulin_premix_30_70", 14.0)]]
    mets = np.ones((n, T))
    mets[0, 120:150] = 3.5
    u = build_inputs(p, T, 7 * 60, meals, mets=mets, doses=doses, therapy=[set(), {"dpp4"}, {"sglt2"}])
    return p, u


def test_fast_integrator_matches_reference():
    p, u = _scenario()
    fast = simulate(p, u)
    ref = simulate(p, u, full=True)
    np.testing.assert_allclose(fast["G"], ref["G"], rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(fast["Gi"], ref["Gi"], rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(fast["final_state"], ref["final_state"], rtol=1e-9, atol=1e-7)


def test_fasting_equilibrium_is_stable():
    p = default_physiology(1, Gb=130.0, dawn=0.0)
    u = build_inputs(p, 1440, 0, [[]])
    g = simulate(p, u)["G"][0]
    assert abs(g[-1] - 130.0) < 0.5


@pytest.mark.parametrize("phenotype,lo,hi", [("normal", 70, 140), ("t2d", 200, 330)])
def test_ogtt_two_hour_glucose_in_clinical_range(phenotype, lo, hi):
    kw = {"normal": dict(Gb=88, Ib=6, SI=6.5e-4, beta=0.13, k_inc=0.9, SG=0.018),
          "t2d": dict(Gb=150, Ib=12, SI=1.8e-4, beta=0.03, k_inc=0.5, SG=0.012)}[phenotype]
    p = default_physiology(1, dawn=0.0, **kw)
    u = build_inputs(p, 200, 8 * 60, [[Meal(0, 75, gi=100)]])
    g = simulate(p, u)["G"][0]
    assert lo <= g[119] <= hi  # ADA: 2-h OGTT < 140 normal, >= 200 diabetes


def test_post_meal_walk_blunts_spike():
    p = default_physiology(2)
    mets = np.ones((2, 300))
    mets[1, 40:70] = 3.5
    u = build_inputs(p, 300, 13 * 60, [[Meal(10, 80, gi=70)]] * 2, mets=mets)
    g = simulate(p, u)["G"]
    assert g[1].max() < g[0].max() - 5


def test_low_gi_meal_peaks_lower_than_high_gi():
    p = default_physiology(2)
    u = build_inputs(p, 300, 13 * 60, [[Meal(10, 80, gi=85)], [Meal(10, 80, gi=40, fiber=12)]])
    g = simulate(p, u)["G"]
    assert g[1].max() < g[0].max()


def test_insulin_without_meal_lowers_glucose():
    p = default_physiology(2, Gb=110, SI=4e-4)
    u = build_inputs(p, 360, 8 * 60, [[], []], doses=[[], [Dose(0, "insulin_premix_30_70", 20)]])
    g = simulate(p, u)["G"]
    assert g[1].min() < g[0].min() - 15
