"""Gradient-boosted forecaster: one LightGBM per horizon and quantile, plus event classifiers.

Predicts the *change* in glucose from now (mg/dL) at 30/60/90/120 min, with P10/P50/P90
quantiles, and the probability of a sustained spike (>180) or hypo (<70) within 2 hours.
SHAP values from the event classifiers power the plain-language "why" on the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from twin.data.windows import HORIZONS

QUANTILES = (0.1, 0.5, 0.9)
REG_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=-1)
CLF_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=-1)


@dataclass
class GBMForecaster:
    names: list[str]
    quantile_models: dict = field(default_factory=dict)   # (h, q) -> model
    event_models: dict = field(default_factory=dict)      # "spike" / "hypo" -> model

    @classmethod
    def fit(cls, d: dict, quantiles: tuple[float, ...] = QUANTILES, max_rows: int = 600_000, seed: int = 0,
            events: bool = True) -> GBMForecaster:
        rng = np.random.default_rng(seed)
        m = cls(d["names"])
        for h in HORIZONS:
            y = d["Y"][:, h - 1]
            ok = np.flatnonzero(np.isfinite(y))
            if len(ok) > max_rows:
                ok = rng.choice(ok, max_rows, replace=False)
            for q in quantiles:
                params = dict(REG_PARAMS, objective="quantile", alpha=q) if q != 0.5 else dict(REG_PARAMS, objective="l2")
                m.quantile_models[(h, q)] = lgb.LGBMRegressor(**params, random_state=seed).fit(d["X"][ok], y[ok])
        if events:
            for ev in ("spike", "hypo"):
                ok = np.flatnonzero(d[f"{ev}_ok"])
                if len(ok) > max_rows:
                    ok = rng.choice(ok, max_rows, replace=False)
                lab = d[ev][ok]
                if lab.sum() < 20:
                    continue
                spw = float(np.clip((len(lab) - lab.sum()) / max(lab.sum(), 1), 1, 20))
                m.event_models[ev] = lgb.LGBMClassifier(**CLF_PARAMS, scale_pos_weight=spw, random_state=seed).fit(d["X"][ok], lab)
        return m

    def predict(self, X: np.ndarray, now: np.ndarray) -> dict:
        """Absolute glucose quantiles (n, len(HORIZONS), 3) and event probabilities."""
        qs = sorted({q for _, q in self.quantile_models})
        out = np.full((len(X), len(HORIZONS), len(qs)), np.nan)
        for i, h in enumerate(HORIZONS):
            for j, q in enumerate(qs):
                if (h, q) in self.quantile_models:
                    out[:, i, j] = now + self.quantile_models[(h, q)].booster_.predict(X)
        out.sort(axis=2)
        res = {"quantiles": out, "q": qs}
        for ev, mdl in self.event_models.items():
            res[ev] = mdl.booster_.predict(X)
        return res

    def explain(self, X: np.ndarray, event: str = "spike", top: int = 5) -> list[list[tuple[str, float]]]:
        """Top SHAP contributions (log-odds) per row for an event classifier."""
        contrib = self.event_models[event].booster_.predict(X, pred_contrib=True)[:, :-1]
        order = np.argsort(-np.abs(contrib), axis=1)[:, :top]
        return [[(self.names[k], float(contrib[r, k])) for k in order[r]] for r in range(len(X))]
