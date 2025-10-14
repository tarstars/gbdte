from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .go_lib import DEFAULT_LIBRARY_NAME


class ExtraBoostBridge:
    """Thin ctypes wrapper over the Go shared library."""

    def __init__(self, library_path: Optional[os.PathLike[str] | str] = None) -> None:
        path = self._resolve_library_path(library_path)
        self._lib = ctypes.CDLL(str(path))
        self._configure_signatures()

    @staticmethod
    def _resolve_library_path(library_path: Optional[os.PathLike[str] | str]) -> Path:
        if library_path is not None:
            return Path(library_path)
        env_path = os.environ.get("EXTRA_BOOST_LIB")
        if env_path:
            return Path(env_path)
        default_path = Path(__file__).resolve().parent / DEFAULT_LIBRARY_NAME
        if not default_path.exists():
            raise FileNotFoundError(
                f"Shared library not found. Looked for {default_path}. "
                "Build it via extra_boost_py.go_lib.build_shared()."
            )
        return default_path

    def _configure_signatures(self) -> None:
        c_double_p = ctypes.POINTER(ctypes.c_double)

        self._lib.TrainModel.restype = ctypes.c_ulonglong
        self._lib.TrainModel.argtypes = [
            c_double_p,
            ctypes.c_int,
            ctypes.c_int,
            c_double_p,
            ctypes.c_int,
            c_double_p,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
        ]

        self._lib.Predict.restype = ctypes.c_int
        self._lib.Predict.argtypes = [
            ctypes.c_ulonglong,
            c_double_p,
            ctypes.c_int,
            ctypes.c_int,
            c_double_p,
            ctypes.c_int,
            c_double_p,
            ctypes.c_int,
        ]

        self._lib.SaveModel.restype = ctypes.c_int
        self._lib.SaveModel.argtypes = [ctypes.c_ulonglong, ctypes.c_char_p]

        self._lib.LoadModel.restype = ctypes.c_ulonglong
        self._lib.LoadModel.argtypes = [ctypes.c_char_p]

        self._lib.FreeModel.restype = None
        self._lib.FreeModel.argtypes = [ctypes.c_ulonglong]

        self._lib.DumpLearningCurves.restype = ctypes.c_int
        self._lib.DumpLearningCurves.argtypes = [ctypes.c_ulonglong, ctypes.c_char_p]

        self._lib.GetLastError.restype = ctypes.c_char_p
        self._lib.GetLastError.argtypes = []

        self._lib.FreeCString.restype = None
        self._lib.FreeCString.argtypes = [ctypes.c_char_p]

    # ------------------------------------------------------------------
    # error handling helpers

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

    # ------------------------------------------------------------------
    # public API

    def train(
        self,
        features_inter: np.ndarray,
        features_extra: np.ndarray,
        target: np.ndarray,
        params: dict,
    ) -> int:
        rows, inter_cols = features_inter.shape
        _, extra_cols = features_extra.shape

        handle = self._lib.TrainModel(
            features_inter.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(rows),
            ctypes.c_int(inter_cols),
            features_extra.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(extra_cols),
            target.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(params["n_stages"]),
            ctypes.c_double(params["reg_lambda"]),
            ctypes.c_int(params["max_depth"]),
            ctypes.c_double(params["learning_rate"]),
            ctypes.c_int(params["loss_kind"]),
            ctypes.c_int(params.get("threads_num", 1)),
            ctypes.c_double(params.get("unbalanced_loss", 0.0)),
        )
        return self._raise_on_zero(handle, "training")

    def predict(
        self,
        handle: int,
        features_inter: np.ndarray,
        features_extra: np.ndarray,
        tree_limit: Optional[int] = None,
    ) -> np.ndarray:
        rows, inter_cols = features_inter.shape
        _, extra_cols = features_extra.shape
        output = np.empty(rows, dtype=np.float64)
        status = self._lib.Predict(
            ctypes.c_ulonglong(handle),
            features_inter.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(rows),
            ctypes.c_int(inter_cols),
            features_extra.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(extra_cols),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(tree_limit or 0),
        )
        self._check_status(status, "prediction")
        return output

    def save(self, handle: int, path: os.PathLike[str] | str) -> None:
        status = self._lib.SaveModel(ctypes.c_ulonglong(handle), str(path).encode("utf-8"))
        self._check_status(status, "save")

    def dump_learning_curves(self, handle: int, path: os.PathLike[str] | str) -> None:
        status = self._lib.DumpLearningCurves(ctypes.c_ulonglong(handle), str(path).encode("utf-8"))
        self._check_status(status, "dump learning curves")

    def load(self, path: os.PathLike[str] | str) -> int:
        handle = self._lib.LoadModel(str(path).encode("utf-8"))
        return self._raise_on_zero(handle, "load")

    def free(self, handle: int) -> None:
        self._lib.FreeModel(ctypes.c_ulonglong(handle))


__all__ = ["ExtraBoostBridge"]
