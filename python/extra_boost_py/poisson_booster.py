from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .poisson_bridge import PoissonLegacyBridge


@dataclass(slots=True)
class PoissonLegacyParams:
    n_stages: int = 10
    reg_lambda: float = 1e-4
    max_depth: int = 6
    learning_rate: float = 0.3
    unbalanced_penalty: float = 0.0
    check_zero: bool = True

    def to_bridge_dict(self) -> dict:
        return {
            "n_stages": int(self.n_stages),
            "reg_lambda": float(self.reg_lambda),
            "max_depth": int(self.max_depth),
            "learning_rate": float(self.learning_rate),
            "unbalanced_penalty": float(self.unbalanced_penalty),
            "check_zero": bool(self.check_zero),
        }


class PoissonLegacyBooster:
    """High-level API for the legacy Poisson booster."""

    def __init__(self, handle: int, bridge: Optional[PoissonLegacyBridge] = None) -> None:
        self._handle = handle
        self._bridge = bridge or PoissonLegacyBridge()
        self._closed = False

    def close(self) -> None:
        if not self._closed and self._handle:
            self._bridge.free(self._handle)
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _ensure_f64(array: np.ndarray, ndim: int) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float64)
        if arr.ndim != ndim:
            raise ValueError(f"Expected {ndim}D array, got shape {arr.shape}.")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _ensure_i32(array: np.ndarray, ndim: int) -> np.ndarray:
        arr = np.asarray(array, dtype=np.int32)
        if arr.ndim != ndim:
            raise ValueError(f"Expected {ndim}D array, got shape {arr.shape}.")
        return np.ascontiguousarray(arr)

    @classmethod
    def train(
        cls,
        bjids: np.ndarray,
        freqs: np.ndarray,
        features_inter: np.ndarray,
        features_extra: Optional[np.ndarray] = None,
        psi: Optional[np.ndarray] = None,
        params: Optional[PoissonLegacyParams] = None,
        bridge: Optional[PoissonLegacyBridge] = None,
    ) -> "PoissonLegacyBooster":
        bridge = bridge or PoissonLegacyBridge()
        params = params or PoissonLegacyParams()

        bjid_arr = cls._ensure_i32(bjids, 1)
        freqs_arr = cls._ensure_f64(freqs, 1)
        f_inter = cls._ensure_f64(features_inter, 2)
        if bjid_arr.shape[0] != f_inter.shape[0]:
            raise ValueError("bjids length must match features_inter rows.")
        if freqs_arr.shape[0] != f_inter.shape[0]:
            raise ValueError("freqs length must match features_inter rows.")

        f_extra = None
        psi_arr = None
        if features_extra is not None:
            f_extra = cls._ensure_f64(features_extra, 2)
            if f_extra.shape[0] != f_inter.shape[0]:
                raise ValueError("features_extra rows must match features_inter rows.")
            if psi is None:
                raise ValueError("psi is required when features_extra is provided.")
            psi_arr = cls._ensure_f64(psi, 1)
            if psi_arr.shape[0] != f_extra.shape[1]:
                raise ValueError("psi length must match features_extra columns.")

        handle = bridge.train(
            bjid_arr,
            freqs_arr,
            f_inter,
            f_extra,
            psi_arr,
            params.to_bridge_dict(),
        )
        return cls(handle, bridge=bridge)

    def predict(
        self,
        features_inter: np.ndarray,
        features_extra: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Booster handle already freed.")
        f_inter = self._ensure_f64(features_inter, 2)
        f_extra = None
        if features_extra is not None:
            f_extra = self._ensure_f64(features_extra, 2)
            if f_extra.shape[0] != f_inter.shape[0]:
                raise ValueError("features_extra rows must match features_inter rows.")
        return self._bridge.predict(self._handle, f_inter, f_extra)


__all__ = ["PoissonLegacyBooster", "PoissonLegacyParams"]
