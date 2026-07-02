"""E7: unified model wrappers. predict() returns raw score (logloss) or value (mse)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..booster import BoosterParams, ExtraBooster
from .benchgen import Bench
from .rolestat import auto_roles


class Model:
    name = "base"

    def available(self) -> bool:
        return True

    def param_grid(self) -> dict:
        return {}

    def fit(self, bench: Bench, params: Optional[dict] = None) -> None:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


def _xy(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    return np.ascontiguousarray(df[cols].to_numpy(dtype=np.float64))


class GBDTEModel(Model):
    def __init__(self, roles: str = "oracle"):
        self.roles = roles
        self.name = {"oracle": "gbdte", "auto": "gbdte_auto", "wrong": "gbdte_wrong",
                     "all_in_leaf": "gbdte_all_in_leaf", "const": "gbdte_const"}[roles]
        self._booster = None
        self._inter_cols: List[str] = []
        self._extra_cols: List[str] = []

    def param_grid(self) -> dict:
        return {"n_stages": [24, 48, 96], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _blocks(self, bench: Bench):
        p, e = bench.partition_cols, bench.extra_cols
        if self.roles == "oracle":
            return p, e
        if self.roles == "wrong":
            return e, ["e_0"] + p          # blocks swapped (e_0 is the constant term)
        if self.roles == "all_in_leaf":
            return p, e + p                # tg_072 failure mode
        if self.roles == "const":
            return p, ["e_0"]              # constant leaves, no separation
        if self.roles == "auto":
            part, leaf = auto_roles(bench)
            return (part if part else ["e_0"]), ["e_0"] + leaf
        raise ValueError(self.roles)

    def fit(self, bench: Bench, params: Optional[dict] = None) -> None:
        self._inter_cols, self._extra_cols = self._blocks(bench)
        p = params or {}
        bp = BoosterParams(loss=bench.task,
                           n_stages=p.get("n_stages", 48),
                           max_depth=p.get("max_depth", 5),
                           learning_rate=p.get("learning_rate", 0.1))
        tr = bench.train
        self._booster = ExtraBooster.train(
            _xy(tr, self._inter_cols), _xy(tr, self._extra_cols),
            np.ascontiguousarray(tr["y"].to_numpy(dtype=np.float64)), bp)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._booster.predict(_xy(df, self._inter_cols), _xy(df, self._extra_cols))


class _SKStyleModel(Model):
    """Shared logic for external boosters: X = partition candidates + t."""

    def _cols(self, bench: Bench) -> List[str]:
        return bench.partition_cols + ["t"]

    def fit(self, bench: Bench, params: Optional[dict] = None) -> None:
        self._fit_cols = self._cols(bench)
        tr = bench.train
        self._fit_impl(_xy(tr, self._fit_cols), tr["y"].to_numpy(), bench.task,
                       params or {})

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._predict_impl(_xy(df, self._fit_cols)), dtype=np.float64)


class LightGBMModel(_SKStyleModel):
    def __init__(self, linear_tree: bool):
        self.linear_tree = linear_tree
        self.name = "lgbm_linear" if linear_tree else "lgbm"

    def available(self) -> bool:
        try:
            import lightgbm  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, task, p):
        import lightgbm as lgb
        cls = lgb.LGBMRegressor if task == "mse" else lgb.LGBMClassifier
        self._m = cls(linear_tree=self.linear_tree, verbose=-1,
                      n_estimators=p.get("n_estimators", 100),
                      max_depth=p.get("max_depth", 5),
                      learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)
        self._task = task

    def _predict_impl(self, X):
        if self._task == "mse":
            return self._m.predict(X)
        return self._m.predict_proba(X, raw_score=True)


class XGBoostModel(_SKStyleModel):
    name = "xgb"

    def available(self) -> bool:
        try:
            import xgboost  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, task, p):
        import xgboost as xgb
        cls = xgb.XGBRegressor if task == "mse" else xgb.XGBClassifier
        self._m = cls(n_estimators=p.get("n_estimators", 100),
                      max_depth=p.get("max_depth", 5),
                      learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)
        self._task = task

    def _predict_impl(self, X):
        if self._task == "mse":
            return self._m.predict(X)
        return self._m.predict(X, output_margin=True)


class CatBoostModel(_SKStyleModel):
    name = "catboost"

    def available(self) -> bool:
        try:
            import catboost  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"iterations": [50, 100, 200], "depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, task, p):
        import catboost as cb
        cls = cb.CatBoostRegressor if task == "mse" else cb.CatBoostClassifier
        self._m = cls(verbose=0, iterations=p.get("iterations", 100),
                      depth=p.get("depth", 5),
                      learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)
        self._task = task

    def _predict_impl(self, X):
        if self._task == "mse":
            return self._m.predict(X)
        return self._m.predict(X, prediction_type="RawFormulaVal")


class OracleModel(Model):
    """Bayes-optimal score provided by the generator (logloss benches only)."""
    name = "oracle"

    def fit(self, bench: Bench, params: Optional[dict] = None) -> None:
        pass

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["oracle"].to_numpy()


class GroupLinearModel(Model):
    """Per-true-group least squares on the extrapolation basis (mse upper bound)."""
    name = "group_linear"

    def fit(self, bench: Bench, params: Optional[dict] = None) -> None:
        self._extra_cols = bench.extra_cols
        self._coef = {}
        tr = bench.train
        for g, part in tr.groupby("group"):
            E = part[self._extra_cols].to_numpy()
            self._coef[g], *_ = np.linalg.lstsq(E, part["y"].to_numpy(), rcond=None)
        self._default = np.mean(list(self._coef.values()), axis=0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        E = df[self._extra_cols].to_numpy()
        coefs = np.array([self._coef.get(g, self._default) for g in df["group"]])
        return np.einsum("ij,ij->i", E, coefs)


def make_models(task: str, include_oracle: bool = True) -> Dict[str, Model]:
    models: List[Model] = [
        GBDTEModel("oracle"), GBDTEModel("auto"), GBDTEModel("wrong"),
        GBDTEModel("all_in_leaf"), GBDTEModel("const"),
        LightGBMModel(linear_tree=True), LightGBMModel(linear_tree=False),
        XGBoostModel(), CatBoostModel(),
    ]
    if task == "mse":
        models.append(GroupLinearModel())
    elif include_oracle:
        models.append(OracleModel())
    return {m.name: m for m in models}


__all__ = ["Model", "GBDTEModel", "LightGBMModel", "XGBoostModel", "CatBoostModel",
           "OracleModel", "GroupLinearModel", "make_models"]
