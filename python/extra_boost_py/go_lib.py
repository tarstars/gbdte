from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent

_PLATFORM_SUFFIX = {
    "Linux": ".so",
    "Darwin": ".dylib",
    "Windows": ".dll",
}


def _default_library_name() -> str:
    system = platform.system()
    suffix = _PLATFORM_SUFFIX.get(system, ".so")
    return f"libextra_boost{suffix}"

def _default_poisson_library_name() -> str:
    system = platform.system()
    suffix = _PLATFORM_SUFFIX.get(system, ".so")
    return f"libextra_poisson_legacy{suffix}"

DEFAULT_LIBRARY_NAME = _default_library_name()
DEFAULT_HEADER_NAME = "libextra_boost.h"
DEFAULT_POISSON_LIBRARY_NAME = _default_poisson_library_name()
DEFAULT_POISSON_HEADER_NAME = "libextra_poisson_legacy.h"


def build_shared(output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Builds the Go shared library used by the Python bindings.

    Parameters
    ----------
    output_dir:
        Destination directory for the compiled artefacts. Defaults to the
        package directory so wheels/sdist can include the binary.

    Returns
    -------
    Tuple[pathlib.Path, pathlib.Path]
        Paths to the compiled shared library and its C header.
    """

    if output_dir is None:
        output_dir = _PACKAGE_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lib_path = output_dir / DEFAULT_LIBRARY_NAME
    header_path = output_dir / DEFAULT_HEADER_NAME

    env = os.environ.copy()
    if "GOCACHE" not in env:
        env["GOCACHE"] = tempfile.mkdtemp(prefix="go-build-cache-")
    env.setdefault("CGO_ENABLED", "1")

    cmd = [
        "go",
        "build",
        "-buildmode=c-shared",
        "-o",
        str(lib_path),
        "./golang/extra_boost/pybridge",
    ]

    subprocess.check_call(cmd, cwd=_REPO_ROOT, env=env)

    # go build places the header next to the library with the same basename
    generated_header = lib_path.with_suffix(".h")
    if generated_header != header_path:
        generated_header.replace(header_path)

    return lib_path, header_path


def build_poisson_legacy_shared(output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Builds the Go shared library for the legacy Poisson bridge."""

    if output_dir is None:
        output_dir = _PACKAGE_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lib_path = output_dir / DEFAULT_POISSON_LIBRARY_NAME
    header_path = output_dir / DEFAULT_POISSON_HEADER_NAME

    env = os.environ.copy()
    if "GOCACHE" not in env:
        env["GOCACHE"] = tempfile.mkdtemp(prefix="go-build-cache-")
    env.setdefault("CGO_ENABLED", "1")

    cmd = [
        "go",
        "build",
        "-buildmode=c-shared",
        "-o",
        str(lib_path),
        "./pybridge",
    ]

    subprocess.check_call(cmd, cwd=_REPO_ROOT / "golang" / "poisson_legacy", env=env)

    generated_header = lib_path.with_suffix(".h")
    if generated_header != header_path:
        generated_header.replace(header_path)

    return lib_path, header_path


__all__ = [
    "build_shared",
    "build_poisson_legacy_shared",
    "DEFAULT_LIBRARY_NAME",
    "DEFAULT_HEADER_NAME",
    "DEFAULT_POISSON_LIBRARY_NAME",
    "DEFAULT_POISSON_HEADER_NAME",
]
