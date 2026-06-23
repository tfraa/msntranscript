"""
OutputManager: no-pickle persistence + sha256 manifest.
Phase 1, Task T1.7.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("msnpip.io.writers")

_MSNPIP_VERSION = "2.0.0"
_BANNED_SUFFIXES = {".pkl", ".pickle"}


class OutputManager:
    """Manages the msnpip output directory tree with sha256 artifact tracking.

    All tabular data is written as CSV or Parquet, arrays as NPZ, and
    structured metadata as JSON.  Pickle files are never created — an
    attempt to ``record`` one raises ``ValueError``.

    Usage::

        mgr = OutputManager(output_dir, engine_commit="e6a2c237", seed=1234)
        mgr.write_table(df, "merged_data")
        mgr.write_array(arr, "strength_maps")
        manifest_path = mgr.finalize(resolved_config)
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        engine_commit: str = "e6a2c237fc74a0b2072a6d58efaf9d1c22cc08e1",
        seed: int = 1234,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._engine_commit = engine_commit
        self._seed = seed
        self._artifacts: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Sub-directory scoping
    # ------------------------------------------------------------------

    def subdir(self, *parts: str) -> OutputManager:
        """Return a child ``OutputManager`` rooted at ``output_dir/parts``.

        Artifacts written by the child are recorded in the child's manifest,
        not the parent's.  Call :meth:`record` on the parent if you want
        the child's directory tracked in the parent manifest.
        """
        child = OutputManager(
            self.output_dir.joinpath(*parts),
            engine_commit=self._engine_commit,
            seed=self._seed,
        )
        return child

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def write_table(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        fmt: str = "csv",
    ) -> Path:
        """Write *df* to CSV or Parquet (no pickle).

        Parameters
        ----------
        df
            DataFrame to write.
        name
            Filename stem (no extension).
        fmt
            ``"csv"`` or ``"parquet"``.

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        fmt = fmt.lower()
        if fmt == "parquet":
            path = self.output_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = self.output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
        self.record(path)
        return path

    def write_array(self, arr: np.ndarray, name: str) -> Path:
        """Write *arr* as a compressed NPZ file (``{name}.npz``)."""
        path = self.output_dir / f"{name}.npz"
        np.savez_compressed(path, data=np.asarray(arr))
        self.record(path)
        return path

    def write_json(self, data: Any, name: str) -> Path:
        """Write *data* as a JSON file (``{name}.json``)."""
        path = self.output_dir / f"{name}.json"
        path.write_text(
            json.dumps(data, indent=2, default=_json_default),
            encoding="utf-8",
        )
        self.record(path)
        return path

    def record(self, path: str | Path) -> None:
        """Register an externally-created artifact for sha256 tracking.

        Parameters
        ----------
        path
            Absolute path of an existing file.

        Raises
        ------
        ValueError
            If *path* ends with ``.pkl`` or ``.pickle``.
        """
        path = Path(path)
        if path.suffix.lower() in _BANNED_SUFFIXES:
            raise ValueError(
                f"Pickle files are not allowed in msnpip v2 outputs: '{path}'. "
                "Use write_array() for numpy data or write_table() for DataFrames."
            )
        sha = _sha256(path) if path.exists() else "missing"
        self._artifacts.append({"path": str(path.relative_to(self.output_dir)), "sha256": sha})

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def finalize(self, resolved_config: dict | None = None) -> Path:
        """Write ``manifest.json`` and return its path.

        Parameters
        ----------
        resolved_config
            The fully resolved pipeline config dict (serialized as-is).

        Returns
        -------
        Path
            Path to the written ``manifest.json``.
        """
        manifest: dict[str, Any] = {
            "msnpip_version": _MSNPIP_VERSION,
            "engine_commit": self._engine_commit,
            "seed": self._seed,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "resolved_config": resolved_config or {},
            "artifacts": self._artifacts,
        }
        path = self.write_json(manifest, "manifest")
        logger.info("manifest written: %s (%d artifacts)", path, len(self._artifacts))
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(obj: Any) -> Any:
    """JSON serializer for types that are not natively serializable."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
