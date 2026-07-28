"""Safe helpers for loading tensor-only PyTorch files."""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import Any

import torch

MINIMUM_SAFE_TORCH = (2, 6, 0)


def public_filename(path: str | PathLike[str]) -> str:
    """Return a filename without leaking POSIX or Windows parent paths."""
    name = os.fspath(path).replace('\\', '/').rstrip('/').rsplit('/', 1)[-1]
    if not name:
        raise ValueError("Path must include a filename")
    return name


def _version_tuple(version: str) -> tuple[int, int, int]:
    """Parse the numeric part of a PyTorch version string."""
    numeric = version.split('+', 1)[0].split('-', 1)[0]
    parts = numeric.split('.')
    try:
        values = [int(part) for part in parts[:3]]
    except ValueError as exc:
        raise RuntimeError(f"Could not parse PyTorch version {version!r}") from exc
    return tuple((values + [0, 0, 0])[:3])


def require_safe_torch_load() -> None:
    """Reject versions affected by the weights-only loading vulnerability."""
    if _version_tuple(torch.__version__) < MINIMUM_SAFE_TORCH:
        required = '.'.join(map(str, MINIMUM_SAFE_TORCH))
        raise RuntimeError(
            f"PyTorch {required} or newer is required for safe tensor loading; "
            f"found {torch.__version__}. Install the pinned requirements before "
            "opening checkpoints or feature files."
        )


def safe_torch_load(
    path: str | Path,
    *,
    map_location: str | torch.device = 'cpu',
) -> Any:
    """Load tensors and built-in containers without enabling pickle objects."""
    require_safe_torch_load()
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(
            f"Could not load {public_filename(path)!r} in tensor-only mode. "
            "Legacy checkpoints containing Python objects must be converted in "
            "an isolated, trusted environment before use."
        ) from exc
