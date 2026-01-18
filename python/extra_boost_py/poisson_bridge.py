from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .go_lib import DEFAULT_POISSON_LIBRARY_NAME


class PoissonLegacyBridge:
    """ctypes wrapper for the legacy Poisson Go library."""

    def __init__(self, library_path: Optional[os.PathLike[str] | str] = None) -> None:
        path = self._resolve_library_path(library_path)
        self._lib = ctypes.CDLL(str(path))
        self._configure_signatures()

    @staticmethod
    def _resolve_library_path(library_path: Optional[os.PathLike[str] | str]) -> Path:
        if library_path is not None:
            return Path(library_path)
        env_path = os.environ.get("EXTRA_POISSON_LEGACY_LIB")
        if env_path:
            return Path(env_path)
        default_path = Path(__file__).resolve().parent / DEFAULT_POISSON_LIBRARY_NAME
        if not default_path.exists():
            raise FileNotFoundError(
                f"Shared library not found. Looked for {default_path}. "
                "Build it via extra_boost_py.go_lib.build_poisson_legacy_shared()."
            )
        return default_path

    def _configure_signatures(self) -> None:
        c_double_p = ctypes.POINTER(ctypes.c_double)
        c_int_p = ctypes.POINTER(ctypes.c_int)

        self._lib.PoissonTrainModel.restype = ctypes.c_ulonglong
        self._lib.PoissonTrainModel.argtypes = [
            c_int_p,
            ctypes.c_int,
            c_double_p,
            c_double_p,
            ctypes.c_int,
            c_double_p,
            ctypes.c_int,
            c_double_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]

        self._lib.PoissonPredict.restype = ctypes.c_int
        self._lib.PoissonPredict.argtypes = [
            ctypes.c_ulonglong,
            c_double_p,
            ctypes.c_int,
            ctypes.c_int,
            c_double_p,
            ctypes.c_int,
            c_double_p,
        ]

        self._lib.PoissonFreeModel.restype = None
        self._lib.PoissonFreeModel.argtypes = [ctypes.c_ulonglong]

        self._lib.GetLastError.restype = ctypes.c_char_p
        self._lib.GetLastError.argtypes = []

        self._lib.FreeCString.restype = None
        self._lib.FreeCString.argtypes = [ctypes.c_char_p]

    def _consume_error(self) -> str:
        err_ptr = self._lib.GetLastError()
        if not err_ptr:
            return ""
        try:
            return ctypes.cast(err_ptr, ctypes.c_char_p).value.decode("utf-8")
        finally:
            self._lib.FreeCString(err_ptr)

    def _raise_on_zero(self, handle: int, action: str) -> int:
        if handle == 0:
            msg = self._consume_error() or f"Go bridge failed during {action}"
            raise RuntimeError(msg)
        return handle

    def _check_status(self, status: int, action: str) -> None:
        if status != 0:
            msg = self._consume_error() or f"Go bridge reported error code {status} during {action}"
            raise RuntimeError(msg)

    def train(
        self,
        bjids: np.ndarray,
        freqs: np.ndarray,
        features_inter: np.ndarray,
        features_extra: Optional[np.ndarray],
        psi: Optional[np.ndarray],
        params: dict,
    ) -> int:
        rows, inter_cols = features_inter.shape

        extra_cols = 0
        if features_extra is not None:
            extra_cols = features_extra.shape[1]

        if extra_cols > 0 and psi is None:
            raise ValueError("psi is required when features_extra is provided.")

        handle = self._lib.PoissonTrainModel(
            bjids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(rows),
            freqs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            features_inter.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(inter_cols),
            None if features_extra is None else features_extra.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(extra_cols),
            None if psi is None else psi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(params["n_stages"]),
            ctypes.c_int(params["max_depth"]),
            ctypes.c_double(params["learning_rate"]),
            ctypes.c_double(params["reg_lambda"]),
            ctypes.c_double(params["unbalanced_penalty"]),
            ctypes.c_int(1 if params.get("check_zero", True) else 0),
        )
        return self._raise_on_zero(handle, "training")

    def predict(
        self,
        handle: int,
        features_inter: np.ndarray,
        features_extra: Optional[np.ndarray],
    ) -> np.ndarray:
        rows, inter_cols = features_inter.shape
        extra_cols = 0
        if features_extra is not None:
            extra_cols = features_extra.shape[1]
        output = np.empty(rows, dtype=np.float64)
        status = self._lib.PoissonPredict(
            ctypes.c_ulonglong(handle),
            features_inter.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(rows),
            ctypes.c_int(inter_cols),
            None if features_extra is None else features_extra.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(extra_cols),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        self._check_status(status, "prediction")
        return output

    def free(self, handle: int) -> None:
        self._lib.PoissonFreeModel(ctypes.c_ulonglong(handle))


__all__ = ["PoissonLegacyBridge"]
