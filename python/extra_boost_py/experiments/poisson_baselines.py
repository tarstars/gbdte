"""Poisson-mode model wrappers. predict() returns expected count per object-bin row.

Engine constraint (see test_poisson_semantics.py): the legacy Poisson booster is only
consistent in extra mode with n_stages=1, so GBDTE variants are single trees with depth
as the capacity knob; predictions are already in per-bin count units.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..poisson_booster import PoissonLegacyBooster, PoissonLegacyParams
from .baselines import Model, _xy
from .poisson_bench import PoissonBench


class GBDTEPoissonModel(Model):
    def __init__(self, roles: str = "oracle"):
        self.roles = roles
        self.name = {"oracle": "gbdte_pois", "wrong": "gbdte_pois_wrong",
                     "const": "gbdte_pois_const"}[roles]
        self._booster = None

    def param_grid(self) -> dict:
        # single stage only (engine constraint); depth is the capacity knob
        return {"max_depth": [2, 3, 5, 7]}

    def _inter_cols(self, bench: PoissonBench) -> List[str]:
        if self.roles == "wrong":
            return ["f_m", "t"]          # mis-specified partitioning
        return bench.partition_cols

    def fit(self, bench: PoissonBench, params: Optional[dict] = None) -> None:
        p = params or {}
        self._cols = self._inter_cols(bench)
        self._use_extra = self.roles != "const"
        self._extra_cols = bench.extra_cols
        bjids, freqs, _, f_extra, psi = bench.event_train()
        f_inter = _xy(bench.train, self._cols)
        kwargs = {}
        if self._use_extra:
            kwargs = dict(features_extra=f_extra, psi=psi)
        self._booster = PoissonLegacyBooster.train(
            bjids=bjids, freqs=freqs, features_inter=f_inter,
            params=PoissonLegacyParams(
                n_stages=1, max_depth=p.get("max_depth", 5),
                learning_rate=1.0, reg_lambda=1e-6),
            **kwargs)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        f_inter = _xy(df, self._cols)
        if self._use_extra:
            preds = self._booster.predict(f_inter, _xy(df, self._extra_cols))
        else:
            preds = self._booster.predict(f_inter)
        return np.clip(preds, 0.0, None)  # already per-bin expected counts


class _SKPoisson(Model):
    def _cols(self, bench: PoissonBench) -> List[str]:
        return bench.partition_cols + ["t"]

    def fit(self, bench: PoissonBench, params: Optional[dict] = None) -> None:
        self._fit_cols = self._cols(bench)
        tr = bench.train
        self._fit_impl(_xy(tr, self._fit_cols), tr["y"].to_numpy(), params or {})

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(np.asarray(self._predict_impl(_xy(df, self._fit_cols)),
                                  dtype=np.float64), 0.0, None)


class LightGBMPoisson(_SKPoisson):
    name = "lgbm_pois"

    def available(self) -> bool:
        try:
            import lightgbm  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, p):
        import lightgbm as lgb
        self._m = lgb.LGBMRegressor(objective="poisson", verbose=-1,
                                    n_estimators=p.get("n_estimators", 100),
                                    max_depth=p.get("max_depth", 5),
                                    learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)

    def _predict_impl(self, X):
        return self._m.predict(X)


class XGBoostPoisson(_SKPoisson):
    name = "xgb_pois"

    def available(self) -> bool:
        try:
            import xgboost  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, p):
        import xgboost as xgb
        self._m = xgb.XGBRegressor(objective="count:poisson",
                                   n_estimators=p.get("n_estimators", 100),
                                   max_depth=p.get("max_depth", 5),
                                   learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)

    def _predict_impl(self, X):
        return self._m.predict(X)


class CatBoostPoisson(_SKPoisson):
    name = "catboost_pois"

    def available(self) -> bool:
        try:
            import catboost  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"iterations": [50, 100, 200], "depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, p):
        import catboost as cb
        self._m = cb.CatBoostRegressor(loss_function="Poisson", verbose=0,
                                       iterations=p.get("iterations", 100),
                                       depth=p.get("depth", 5),
                                       learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)

    def _predict_impl(self, X):
        return self._m.predict(X)


class PoissonOracleModel(Model):
    name = "oracle_pois"

    def fit(self, bench: PoissonBench, params: Optional[dict] = None) -> None:
        self._delta = bench.delta

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["lam_true"].to_numpy() * self._delta


def make_poisson_models(include_oracle: bool = True) -> Dict[str, Model]:
    models: List[Model] = [
        GBDTEPoissonModel("oracle"), GBDTEPoissonModel("wrong"),
        GBDTEPoissonModel("const"),
        LightGBMPoisson(), XGBoostPoisson(), CatBoostPoisson(),
    ]
    if include_oracle:
        models.append(PoissonOracleModel())
    return {m.name: m for m in models}


__all__ = ["GBDTEPoissonModel", "LightGBMPoisson", "XGBoostPoisson",
           "CatBoostPoisson", "PoissonOracleModel", "make_poisson_models"]
