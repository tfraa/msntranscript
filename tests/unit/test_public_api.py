"""The top-level public API is importable and consistent (I4)."""

from __future__ import annotations

import msnpip


def test_public_api_surface():
    assert msnpip.__version__ == "2.0.0"
    assert hasattr(msnpip, "Pipeline")
    assert hasattr(msnpip, "run_pipeline")
    assert hasattr(msnpip, "PipelineConfig")
    for name in ("Pipeline", "run_pipeline", "PipelineConfig"):
        assert name in msnpip.__all__
