from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .bridge import ExtraBoostBridge

_LOSS_KIND_MAP = {
    "mse": 0,
    "logloss": 1,
}


@dataclass(slots=True)
class BoosterParams:
    n_stages: int = 200
    reg_lambda: float = 1e-4
    max_depth: int = 6
    learning_rate: float = 0.3
    loss: str = "mse"
    threads_num: int = 1
    unbalanced_loss: float = 0.0

    def to_bridge_dict(self) -> Dict[str, Any]:
        try:
            loss_kind = _LOSS_KIND_MAP[self.loss.lower()]
        except KeyError as err:
            raise ValueError(f"Unsupported loss '{self.loss}'.") from err
        return {
            "n_stages": int(self.n_stages),
            "reg_lambda": float(self.reg_lambda),
            "max_depth": int(self.max_depth),
            "learning_rate": float(self.learning_rate),
            "loss_kind": loss_kind,
            "threads_num": int(self.threads_num),
            "unbalanced_loss": float(self.unbalanced_loss),
        }


class ExtraBooster:
    """High-level API mirroring the Go booster."""

    def __init__(
        self,
        handle: int,
        features_inter_dim: int,
        features_extra_dim: int,
        bridge: Optional[ExtraBoostBridge] = None,
    ) -> None:
        self._handle = handle
        self._bridge = bridge or ExtraBoostBridge()
        self._inter_dim = features_inter_dim
        self._extra_dim = features_extra_dim
        self._closed = False

    # ------------------------------------------------------------------
    # lifecycle helpers

    def close(self) -> None:
        if not self._closed and self._handle:
            self._bridge.free(self._handle)
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # construction helpers

    @staticmethod
    def _ensure_f64(array: np.ndarray, ndim: int) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float64)
        if arr.ndim != ndim:
            raise ValueError(f"Expected {ndim}D array, got shape {arr.shape}.")
        return np.ascontiguousarray(arr)

    @classmethod
    def train(
        cls,
        features_inter: np.ndarray,
        features_extra: np.ndarray,
        target: np.ndarray,
        params: BoosterParams | None = None,
        bridge: Optional[ExtraBoostBridge] = None,
    ) -> "ExtraBooster":
        bridge = bridge or ExtraBoostBridge()
        params = params or BoosterParams()

        f_inter = cls._ensure_f64(features_inter, 2)
        f_extra = cls._ensure_f64(features_extra, 2)
        if f_inter.shape[0] != f_extra.shape[0]:
            raise ValueError("features_inter and features_extra must share the same number of rows.")
        f_target = cls._ensure_f64(target, 1)
        if f_target.shape[0] != f_inter.shape[0]:
            raise ValueError("Target length must match number of rows.")

        handle = bridge.train(f_inter, f_extra, f_target, params.to_bridge_dict())
        return cls(handle, f_inter.shape[1], f_extra.shape[1], bridge=bridge)

    @classmethod
    def load(cls, path: str | Path, bridge: Optional[ExtraBoostBridge] = None) -> "ExtraBooster":
        bridge = bridge or ExtraBoostBridge()
        handle = bridge.load(path)
        # Without metadata, we cannot infer feature dimensions; require caller to set manually
        return cls(handle, features_inter_dim=-1, features_extra_dim=-1, bridge=bridge)

    # ------------------------------------------------------------------
    # inference & persistence

    def predict(
        self,
        features_inter: np.ndarray,
        features_extra: np.ndarray,
        tree_limit: Optional[int] = None,
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Booster handle already freed.")

        f_inter = self._ensure_f64(features_inter, 2)
        f_extra = self._ensure_f64(features_extra, 2)
        if f_inter.shape[0] != f_extra.shape[0]:
            raise ValueError("features_inter and features_extra must share the same number of rows.")

        return self._bridge.predict(self._handle, f_inter, f_extra, tree_limit)

    def save(self, path: str | Path) -> None:
        if self._closed:
            raise RuntimeError("Booster handle already freed.")
        self._bridge.save(self._handle, path)

    def dump_learning_curves(self, path: str | Path) -> None:
        if self._closed:
            raise RuntimeError("Booster handle already freed.")
        self._bridge.dump_learning_curves(self._handle, path)


__all__ = ["ExtraBooster", "BoosterParams"]
